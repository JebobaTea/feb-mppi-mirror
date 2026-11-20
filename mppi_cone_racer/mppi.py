import functools

import numpy as np
import jax
import jax.numpy as jnp
import time
import math
from .halfspaceutils import generate_rectangular_halfspaces
from .vehicle import AXLE_TO_CENTRE, CORNER_DRAG, DRAG, MAX_STEER, STEER_RATE, WHEELBASE, AXLE_TO_LIDAR

# FIXME: should NOT be a magic number, should be a config option
M_MAX_CONES = 16  # unfuck JAX kernel with static obstruction array dimensions

class OUContourKinematicMPPI:
    def __init__(self, p=None):
        p = {} if p is None else p
        self.ok = True
        self.u_min = (-float(p.get("mpc_a_brake", 5.4)), -STEER_RATE)
        self.u_max = (float(p.get("mpc_a_acc", 5.4)), STEER_RATE)
        self.delta_min, self.delta_max = -MAX_STEER, MAX_STEER
        self.DT = float(p.get("mppi_dt", 0.05))
        self.N = int(p.get("mppi_horizon", 32))
        self.num_samples = int(p.get("mppi_num_samples", 2048))
        self.temperature = float(p.get("mppi_temperature", 3.5)) # adjust to keep K_eff @ 5 - 20%
        self.steer_sigma = float(p.get("mppi_steer_sigma", 0.35))
        self.accel_sigma = float(p.get("mppi_accel_sigma", 1.5))
        self.beta = float(p.get("mppi_beta", 0.85)) # see comments @ generate_colored_noise

        self._warm = None
        self.status = "init"
        self.track_ready = False
        self.uses_centerline = False
        self.key = jax.random.key(6767)

        self.w_progress = float(p.get("w_progress", 6.0))
        self.w_contour = float(p.get("w_contour", 8.0))
        self.w_heading = float(p.get("w_heading", 15.0))
        self.w_speed = float(p.get("w_speed", 10.0)) # enforce speed cap
        self.w_accel = float(p.get("w_accel", 8.0))
        self.w_effort = float(p.get("w_effort", 0.5))
        self.w_effort_omega = float(p.get("w_effort_omega", 5.0))
        self.w_domega = float(p.get("w_domega", 2.0))
        self.w_collisions = float(p.get("w_collision", 2000.0)) # proximity-based penalty
        self.w_collision_hard = float(p.get("w_collision_hard", 5000.0)) # actual intersection
        self.w_boundary = float(p.get("w_boundary", 1e5)) # stop exiting the track

        self.default_track_width = float(p.get("track_width", 2.2))
        self.track_width_left_table = None
        self.track_width_right_table = None
        self.track_width_table = None
        self.width_table = None

        self.max_a_net = float(p.get("max_a_net", 5.8)) # FIXME: name changes halfway through the kernel
        self.v_max = float(p.get("v_max", 7.0))

        half_width = float(p.get("car_half_width", 0.135))
        front_hang = float(p.get("car_front_overhang", 0.100))
        rear_hang = float(p.get("car_rear_overhang", 0.100))
        # 4-plane bounding box (or use generate_chamfered_halfspaces for 8 planes)
        self.A, self.b = generate_rectangular_halfspaces(
            wheelbase=WHEELBASE,
            front_overhang=front_hang,
            rear_overhang=rear_hang,
            half_width=half_width,
            as_jax=True,
            inflation=0.0 # FIXME: MAGIC NUMBER
        )
        self.collision_margin = float(p.get("collision_margin", 0.15)) # start penalizing cone proximity at this distance

        print(f"[MPPI] Initialized: temperature={self.temperature}, horizon={self.N}, dt={self.DT}, num_samples={self.num_samples}")
        print(f"[MPPI] Weights: progress={self.w_progress}, contour={self.w_contour}, heading={self.w_heading}, speed={self.w_speed}, effort_a={self.w_effort}, effort_w={self.w_effort_omega}, domega={self.w_domega}, boundary={self.w_boundary}, accel={self.w_accel}")

    @property
    def warm(self):
        return self._warm

    @warm.setter
    def warm(self, val):
        self._warm = val
        if val is None:
            self.plan = jnp.zeros((self.N, 2))

    # NOTE:
    # JAX handles objects and attributes very poorly; everything MUST be static and immutable at compilation time
    # so we pull + freeze reference line data immediately after the mapping lap into a fixed array
    # instead of using controller.py's Raceline, we maintain our own pure functions for cartesian-frenet conversions
    # and our own tables
    # XXX: unreadable slop
    def set_centerline(self, centerline, half_width=None):
        """Prepare Frenet lookup tables from track centerline and boundary widths."""
        if not self.ok:
            return
        if isinstance(centerline, dict):
            if half_width is None:
                half_width = centerline.get("half_width", None)
            centerline = centerline.get("centre", centerline.get("line", centerline))
        if hasattr(centerline, "line"):
            pts = np.asarray(centerline.line[:, :2], float)
        else:
            pts = np.asarray(centerline, float)
            if pts.ndim == 2 and pts.shape[1] > 2:
                pts = pts[:, :2]

        seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        total_length = float(np.sum(seg)) # perimeter calculation
        s_cum = np.concatenate([[0.0], np.cumsum(seg)])
        pts_closed = np.vstack([pts, pts[0:1]])

        num_pts = max(int(total_length / 0.1), 32) # XXX: s_grid ds is ostensibly hardcoded here...
        s_grid = np.linspace(0.0, total_length, num_pts, endpoint=False)
        x_grid = np.interp(s_grid, s_cum, pts_closed[:, 0])
        y_grid = np.interp(s_grid, s_cum, pts_closed[:, 1])

        # Interpolate and build track width tables along s_grid
        if isinstance(half_width, (tuple, list)) and len(half_width) == 2:
            hw_l = np.asarray(half_width[0], float)
            hw_r = np.asarray(half_width[1], float)
        elif isinstance(half_width, np.ndarray) and half_width.ndim == 2 and half_width.shape[1] >= 2:
            hw_l = half_width[:, 0]
            hw_r = half_width[:, 1]
        elif half_width is not None and np.ndim(half_width) == 1 and len(half_width) == len(pts):
            hw_l = np.asarray(half_width, float)
            hw_r = np.asarray(half_width, float)
        elif half_width is not None and np.isscalar(half_width):
            hw_l = np.full(len(pts), float(half_width))
            hw_r = np.full(len(pts), float(half_width))
        else:
            default_hw = self.default_track_width / 2.0
            hw_l = np.full(len(pts), default_hw)
            hw_r = np.full(len(pts), default_hw)

        hw_l_closed = np.concatenate([hw_l, hw_l[:1]])
        hw_r_closed = np.concatenate([hw_r, hw_r[:1]])
        w_left_grid = np.interp(s_grid, s_cum, hw_l_closed)
        w_right_grid = np.interp(s_grid, s_cum, hw_r_closed)

        # Smooth waypoints and width tables cyclically to drop piecewise-linear interpolation kinks
        k = 3  # 7-point circular moving average (~0.7 m window) # FIXME: magic number, multiple dependent usages elsewhere
        pad_x = np.concatenate([x_grid[-k:], x_grid, x_grid[:k]])
        pad_y = np.concatenate([y_grid[-k:], y_grid, y_grid[:k]])
        pad_wl = np.concatenate([w_left_grid[-k:], w_left_grid, w_left_grid[:k]])
        pad_wr = np.concatenate([w_right_grid[-k:], w_right_grid, w_right_grid[:k]])
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        x_grid = np.convolve(pad_x, ker, mode="valid")
        y_grid = np.convolve(pad_y, ker, mode="valid")
        w_left_grid = np.maximum(np.convolve(pad_wl, ker, mode="valid"), 0.2)
        w_right_grid = np.maximum(np.convolve(pad_wr, ker, mode="valid"), 0.2)

        # Tangents and headings via central finite differences (with cyclic wrapping)
        dx = np.roll(x_grid, -1) - np.roll(x_grid, 1)
        dy = np.roll(y_grid, -1) - np.roll(y_grid, 1)
        headings = np.arctan2(dy, dx)

        # Curvature dphi / ds
        dphi = np.arctan2(np.sin(np.roll(headings, -1) - headings),
                          np.cos(np.roll(headings, -1) - headings))
        ds = total_length / num_pts
        curvatures = np.clip(dphi / ds, -1.5, 1.5) # XXX: why?

        self.track_len = total_length
        self.s_table = jnp.array(s_grid)
        self.x_table = jnp.array(x_grid)
        self.y_table = jnp.array(y_grid)
        self.phi_table = jnp.array(headings)
        self.kappa_table = jnp.array(curvatures)
        self.track_width_left_table = jnp.array(w_left_grid, dtype=jnp.float32)
        self.track_width_right_table = jnp.array(w_right_grid, dtype=jnp.float32)
        self.track_width_table = jnp.array(0.5 * (w_left_grid + w_right_grid), dtype=jnp.float32)
        self.width_table = self.track_width_table
        self.uses_centerline = True
        self.track_ready = True
        self.plan = jnp.zeros((self.N, 2))
        print(f"[MPPI] Centerline configured: length={total_length:.2f} m, {num_pts} waypoints, ds={ds:.3f} m, track_width=[{w_left_grid.min() + w_right_grid.min():.2f}..{w_left_grid.max() + w_right_grid.max():.2f}] m")

    # XXX: unreadable slop
    def set_raceline(self, raceline):
        """Prepare Frenet lookup tables from controller.Raceline."""
        if not self.ok:
            return
        # FIXME: why this wack ass solution for prioritizing centerline over raceline?
        if getattr(self, "uses_centerline", False):
            # Centerline is already configured; ignore raceline to keep tracking centerline
            return
        s_grid = np.linspace(0.0, raceline.length, max(int(raceline.length / 0.1), 32), endpoint=False)
        pts = np.array([raceline.at(s) for s in s_grid])  # (N, 4): x, y, h, v

        self.track_len = float(raceline.length)
        self.s_table = jnp.array(s_grid)
        self.x_table = jnp.array(pts[:, 0])
        self.y_table = jnp.array(pts[:, 1])
        self.phi_table = jnp.array(pts[:, 2])

        # Heading-difference curvature
        dphi = np.arctan2(np.sin(np.roll(pts[:, 2], -1) - pts[:, 2]),
                          np.cos(np.roll(pts[:, 2], -1) - pts[:, 2]))
        ds = raceline.length / len(s_grid)
        self.kappa_table = jnp.array(np.clip(dphi / ds, -1.5, 1.5))
        default_hw = self.default_track_width / 2.0
        self.track_width_left_table = jnp.full(len(s_grid), default_hw, dtype=jnp.float32)
        self.track_width_right_table = jnp.full(len(s_grid), default_hw, dtype=jnp.float32)
        self.track_width_table = jnp.full(len(s_grid), default_hw, dtype=jnp.float32)
        self.width_table = self.track_width_table
        self.track_ready = True
        self.plan = jnp.zeros((self.N, 2))
        print(f"[MPPI] Raceline configured: length={self.track_len:.2f} m, {len(s_grid)} waypoints, ds={ds:.3f} m")

    # ref, uprev unused (JAX-friendly versions maintained in this file instead)
    # and these parameters kept around for consistency's sake
    def solve(self, z0, ref, uprev, obstructions):
        """Solve MPPI and return (U, Z) meeting RaceController contract."""
        if not self.ok or not self.track_ready:
            return None
        # prepare static padded cone array in world coordinates as JAX requires static array dimensions;
        # variable amount of obstructions will cause recompilation at every timestep
        # should be very fast
        pt0 = time.process_time_ns()
        cones_world = np.full((M_MAX_CONES, 2), 10000.0, dtype=np.float32)
        if obstructions:
            ahead = np.array([math.cos(z0[2]), math.sin(z0[2])])
            pos_lidar = z0[:2] + AXLE_TO_LIDAR * ahead
            pts = []
            for obs in obstructions:
                # obs is (x_l, y_l, ...) in lidar frame
                xl, yl = obs[0], obs[1]
                xw = pos_lidar[0] + xl * ahead[0] - yl * ahead[1]
                yw = pos_lidar[1] + xl * ahead[1] + yl * ahead[0]
                pts.append((xw, yw))
            if pts:
                pts = np.array(pts, dtype=np.float32)
                # Take M_MAX_CONES closest
                dists = np.linalg.norm(pts - z0[:2], axis=1)
                idx = np.argsort(dists)[:M_MAX_CONES]
                cones_world[:len(idx)] = pts[idx]
        pt1 = time.process_time_ns()
        print(f"[MPPI BadDebuggingPractices] prepare obstruction array process time {(pt1 - pt0) // 1_000}us")

        pt0 = time.process_time_ns()
        # monstrous kernel "cascades" config options into the necessary pure utility functions; keeping everything
        # in one place squeezes out as much time savings as possible
        new_ctrl, Z, costs, self.key, k_eff, new_plan = jax.block_until_ready(mppi_step_kernel(
            state=jnp.asarray(z0, dtype=jnp.float32),
            nominal_ctrl=self.plan,
            key=self.key,
            cones=jnp.asarray(cones_world),
            line_tables=(
                self.s_table,
                self.x_table,
                self.y_table,
                self.phi_table,
                self.kappa_table,
                self.track_width_left_table,
                self.track_width_right_table,
            ),
            track_len=self.track_len,
            A=self.A, b=self.b, collision_margin=self.collision_margin,
            u_min=jnp.array(self.u_min), u_max=jnp.array(self.u_max),
            wheel_limits=(self.delta_min, self.delta_max),
            dt=self.DT, horizon=self.N, num_samples=self.num_samples,
            sigmas=jnp.array([self.accel_sigma, self.steer_sigma]),
            beta=self.beta, temperature=self.temperature,
            w_progress=self.w_progress,
            w_contour=self.w_contour,
            w_heading=self.w_heading,
            w_accel=self.w_accel,
            w_effort=self.w_effort,
            w_effort_omega=self.w_effort_omega,
            w_domega=self.w_domega,
            w_collision=self.w_collisions,
            w_collision_hard=self.w_collision_hard,
            w_speed=self.w_speed,
            w_boundary=self.w_boundary,
            v_max=self.v_max,
            wheelbase=WHEELBASE,
            track_width=self.track_width_table,
            drag=DRAG,
            corner_drag=CORNER_DRAG,
            max_friction_accel=self.max_a_net,
        ))
        pt1 = time.process_time_ns()
        print(f"[MPPI BadDebuggingPractices] step_kernel solve process time {(pt1 - pt0) // 1_000}us")

        # advance plan
        self.plan = new_plan
        self.status = f"K_eff: {float(k_eff):.2f}%"

        # return (U, Z) where U is shape (2, N) with [a, omega]
        U = np.array(new_ctrl).T

        return U, np.array(Z)

# horizon, num_samples are compile-time constants
@functools.partial(jax.jit, static_argnames=['horizon', 'num_samples'])
def mppi_step_kernel(state, nominal_ctrl, key,
                     cones, A, b, track_len,
                     w_progress, w_contour, w_heading, w_accel, w_effort, w_effort_omega, w_domega,
                     w_collision, w_collision_hard, w_speed, w_boundary,
                     v_max, beta,
                     max_friction_accel, collision_margin,
                     sigmas, temperature,
                     u_min, u_max, wheel_limits, wheelbase,
                     dt, horizon, num_samples, line_tables, track_width, drag, corner_drag):

    key, subkey = jax.random.split(key)
    noise = generate_colored_noise(subkey, num_samples, horizon, sigmas, beta)

    # noise shape: (K, H, 2)
    all_perturbed = nominal_ctrl[None, :, :] + noise
    min_ctrl = jnp.array([u_min[0], u_min[1]])  # [accel, steer_rate]
    max_ctrl = jnp.array([u_max[0], u_max[1]])  # [accel, steer_rate]
    all_clipped_ctrl = jnp.clip(all_perturbed, min_ctrl, max_ctrl)

    # locate anchor waypoint in cartesian tables
    cx0 = state[0] + AXLE_TO_CENTRE * jnp.cos(state[2])
    cy0 = state[1] + AXLE_TO_CENTRE * jnp.sin(state[2])
    base_idx = jnp.argmin((line_tables[1] - cx0) ** 2 + (line_tables[2] - cy0) ** 2)

    # restrict search area for each cartesian to frenet conversion call (16 waypoints behind, 112 waypoints ahead)
    # currently, the order of operations is for each vmapped rollout cost call, updated poses are computed in
    # cartesian coordinates (obviously), for which we must query the frenet equivalent for to obtain the correct
    # progress and cross-track error for a certain rollout; each pose's coordinate conversion takes the argmin
    # using the entire given table, so it is more effective to gather a fixed slice of waypoints we know no
    # rollout will possibly rocket out of bounds for and use it for every rollout
    # that is to say, tables ideally should not be gathered within the conversion call...
    # the window size should be adjusted based on the ds between waypoints and the expected travel distance per rollout
    N_TRACK = line_tables[0].shape[0]
    WINDOW_SIZE = 128 # FIXME: move to unified config area, stop leaving levers in random places
    BACKWARD_MARGIN = 16
    local_indices = (base_idx - BACKWARD_MARGIN + jnp.arange(WINDOW_SIZE)) % N_TRACK
    # Slice all tables to static shape (128,)
    local_line_tables = tuple(table[local_indices] for table in line_tables)

    costs = jax.vmap(lambda u: rollout_cost_frenet(state, u, local_line_tables, wheelbase, dt,
                                                   wheel_limits[0], wheel_limits[1], track_len, cones, A, b,
                                                   w_progress, w_contour, w_heading, w_accel,
                                                   w_effort, w_effort_omega, w_domega,
                                                   w_collision, w_collision_hard, w_speed, w_boundary,
                                                   v_max,
                                                   max_friction_accel, collision_margin, track_width, drag,
                                                   corner_drag))(all_clipped_ctrl)

    delta_costs = costs - jnp.min(costs)
    weights = jax.nn.softmax(-delta_costs / temperature)

    new_ctrl = jnp.sum(weights[:, None, None] * all_clipped_ctrl, axis=0)
    traj_steps = rollout(state, new_ctrl, wheelbase, dt, wheel_limits[0], wheel_limits[1], drag, corner_drag)
    Z = jnp.vstack([state[None, :], traj_steps]).T

    # weights shape: (num_samples,) after softmax
    K_eff = 1.0 / jnp.sum(weights ** 2)
    # As an effective percentage of total samples:
    K_eff_pct = (K_eff / num_samples) * 100.0

    new_plan = jnp.roll(new_ctrl, shift=-1, axis=0).at[-1].set(new_ctrl[-1])

    return new_ctrl, Z, costs, key, K_eff_pct, new_plan

# generate correlated noise along the horizon axis
# steering is a rate, so gaussian noise will cause jittering
# time-correlated noise will broaden exploration space by
# pushing out further across timesteps
# beta can be understood as temporal persistence, where the memory time constant
# tau ~= dt / 1 - beta
# https://www.mdpi.com/2504-446X/10/2/96#Methods_a5
def generate_colored_noise(key, num_samples, horizon, sigmas, beta=0.85):
    raw_noise = jax.random.normal(key, (num_samples, horizon, 2)) * sigmas
    def step(prev, current):
        val = beta * prev + jnp.sqrt(1.0 - beta**2) * current
        return val, val
    _, colored_noise = jax.lax.scan(step, jnp.zeros((num_samples, 2)), raw_noise.swapaxes(0, 1))
    return colored_noise.swapaxes(0, 1)

def check_trajectory_collisions(trajectory, cones, A, b, margin,
                                dt, w_collision, w_collision_hard):
    """
    Args:
        trajectory: (H, 5) representing [x, y, phi, v, theta]
        cones: (M, 2) array of nearby cone coordinates
        A: (K, 2) half-space normals
        b: (K,) half-space offsets
    Returns:
        total_collision_cost: scalar
    """
    # Extract predicted poses across horizon: shapes (H, 1)
    x = trajectory[:, 0:1]
    y = trajectory[:, 1:2]
    phi = trajectory[:, 2:3]

    # Translate cones relative to each trajectory step: shape (H, M, 2)
    # cones shape is (1, M, 2), poses shape is (H, 1, 2)
    dx = cones[None, :, 0] - x
    dy = cones[None, :, 1] - y

    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    # Rotate into car body frame: (H, M)
    p_body_x = cos_phi * dx + sin_phi * dy
    p_body_y = -sin_phi * dx + cos_phi * dy
    p_body = jnp.stack([p_body_x, p_body_y], axis=-1) # (H, M, 2)

    # Half-space evaluations: (H, M, K)
    # A @ p_body - b
    ineq = jnp.einsum('k d, h m d -> h m k', A, p_body) - b # < 0 inside body

    # hard collision: all half-spaces satisfied (inside car)
    is_inside = jnp.all(ineq < 0, axis=-1) # (H, M)
    hard_cost = jnp.sum(is_inside) * w_collision_hard

    # soft repulsion: penalize cones within safety margin of any edge
    # Maximum signed distance to boundary across half-spaces for each cone
    dist_to_boundary = jnp.max(ineq, axis=-1) # (H, M)
    soft_violation = jnp.maximum(0.0, margin - dist_to_boundary)
    soft_cost = jnp.sum(soft_violation ** 2) * w_collision * dt

    return hard_cost + soft_cost

def bicycle_step(state, ctrl,
                 wheelbase, dt,
                 steer_min, steer_max,
                 drag, corner_drag):
    x, y, phi, v, delta = state
    a, omega = ctrl  # ctrl[0] = force/accel (m/s^2), ctrl[1] = steering rate (rad/s)

    x_new = x + v * dt * jnp.cos(phi)
    y_new = y + v * dt * jnp.sin(phi)

    # Kinematic yaw rate and lateral acceleration demand
    r_kin = (v / wheelbase) * jnp.tan(delta)
    ay_demand = jnp.abs(v * r_kin)

    # Tire adhesion saturation (6.3 m/s^2 peak, 5.0 m/s^2 sliding)
    # stolen from vehicle.Plant
    # FIXME: magic numbers...
    r_eff = jnp.where(
        ay_demand > 6.3,
        jnp.sign(r_kin) * (5.0 / jnp.maximum(v, 0.4)),
        r_kin
    )
    phi_new = phi + r_eff * dt

    # Friction ellipse limits available drive force during cornering
    # Using lateral demand ay_demand ensures available drive force drops smoothly to 0 at grip limit
    grip_ratio = jnp.clip(ay_demand / 6.3, 0.0, 1.0)
    a_long_max = 6.3 * jnp.sqrt(1.0 - grip_ratio ** 2)
    # Forward drive acceleration is bounded by available grip; braking is uninhibited
    a_effective = jnp.where(a > 0.0, jnp.minimum(a, a_long_max), a)

    dv = a_effective - drag * v - corner_drag * (v ** 2) * jnp.abs(delta * jnp.tan(delta)) / wheelbase
    v_new = jnp.maximum(0.0, v + dv * dt)

    delta_new = jnp.clip(delta + omega * dt, steer_min, steer_max)

    return jnp.array([x_new, y_new, phi_new, v_new, delta_new])

def rollout(state_init, ctrl, wheelbase, dt, theta_min, theta_max, drag, corner_drag):
    def step(state, u):
        next_state = bicycle_step(state, u, wheelbase, dt, theta_min, theta_max, drag, corner_drag)
        return next_state, next_state
    _, trajectory = jax.lax.scan(step, state_init, ctrl)
    return trajectory # (H, 5) for one sample

def cartesian_to_frenet(pose, s_table, x_table, y_table, phi_table, kappa_table, track_len,
                        width_left_table=None, width_right_table=None):
    """
    Transforms a Cartesian pose [x, y, phi] to Frenet coordinates [s, e_y, e_phi, kappa]
    (and optionally [w_left, w_right] if track width tables are provided).

    Args:
        pose: (3,) array [x, y, phi]
    Returns:
        s: Arc length along centerline (wrapped [0, track_len])
        e_y: Cross-track lateral error (meters, positive = left of travel)
        e_phi: Heading error relative to track tangent (radians)
        kappa: Track curvature at projection point (1/m)
        (optional) w_left: distance to left track boundary (meters)
        (optional) w_right: distance to right track boundary (meters)
    """
    px, py, p_phi = pose[0], pose[1], pose[2]

    # global nearest waypoint index
    dists_sq = (x_table - px) ** 2 + (y_table - py) ** 2
    idx = jnp.argmin(dists_sq)

    # continuous interpolation along local tangent segment
    s_base = s_table[idx]
    x_base = x_table[idx]
    y_base = y_table[idx]
    phi_base = phi_table[idx]
    kappa = kappa_table[idx]

    dx = px - x_base
    dy = py - y_base

    # Tangent and normal vectors
    tx = jnp.cos(phi_base)
    ty = jnp.sin(phi_base)
    nx = -ty
    ny = tx

    # Longitudinal offset along tangent and lateral offset along normal
    delta_s = dx * tx + dy * ty
    e_y = dx * nx + dy * ny

    # Continuous arc length with cyclic track wrapping
    s = jnp.mod(s_base + delta_s, track_len)

    # Heading error with angular wrap-around
    e_phi = jnp.arctan2(jnp.sin(p_phi - phi_base), jnp.cos(p_phi - phi_base))

    if width_left_table is not None and width_right_table is not None:
        w_left = width_left_table[idx]
        w_right = width_right_table[idx]
        return s, e_y, e_phi, kappa, w_left, w_right

    return s, e_y, e_phi, kappa

def rollout_cost_frenet(state_init, ctrl, line_tables, wheelbase, dt,
                        theta_min, theta_max, track_len, cones, A, b,
                        w_progress, w_contour, w_heading,
                        w_accel, w_effort, w_effort_omega, w_domega,
                        w_collision, w_collision_hard, w_speed, w_boundary,
                        v_max, max_friction_accel, collision_margin,
                        track_width, drag, corner_drag):
    """
    Computes total trajectory cost in the Frenet frame for a single sample.
    """
    trajectory = rollout(state_init, ctrl, wheelbase, dt, theta_min, theta_max, drag, corner_drag)
    # FIXME: gemini's magic numbers, probably to account for 7-pt circular MA?
    if len(line_tables) >= 7:
        s_table, x_table, y_table, phi_table, kappa_table, width_left_table, width_right_table = line_tables[:7]
    elif len(line_tables) == 6:
        s_table, x_table, y_table, phi_table, kappa_table, width_table = line_tables
        width_left_table = width_table
        width_right_table = width_table
    else:
        s_table, x_table, y_table, phi_table, kappa_table = line_tables
        width_left_table = jnp.full_like(s_table, track_width)
        width_right_table = jnp.full_like(s_table, track_width)

    # Transform predicted trajectory to Frenet frame at CENTRE OF MASS (matching controller & raceline)
    centre_x = trajectory[:, 0] + AXLE_TO_CENTRE * jnp.cos(trajectory[:, 2])
    centre_y = trajectory[:, 1] + AXLE_TO_CENTRE * jnp.sin(trajectory[:, 2])
    poses = jnp.stack([centre_x, centre_y, trajectory[:, 2]], axis=-1)  # (H, 3)
    s, e_y, e_phi, kappa, w_left, w_right = jax.vmap(
        lambda p: cartesian_to_frenet(p, s_table, x_table, y_table, phi_table, kappa_table, track_len,
                                     width_left_table, width_right_table)
    )(poses)

    v = trajectory[:, 3]
    theta = trajectory[:, 4]
    a_tangential = ctrl[:, 0]
    omega = ctrl[:, 1]

    # Longitudinal velocity progress:
    progress_cost = -w_progress * jnp.sum(v * jnp.cos(e_phi)) * dt

    # Lateral deviation and yaw alignment (w_heading >= w_contour provides lateral damping):
    contour_cost = w_contour * jnp.sum(e_y ** 2) * dt
    heading_cost = w_heading * jnp.sum(e_phi ** 2) * dt

    # Boundary violation using the appropriate local track width (left & right):
    violation_left = jnp.maximum(0.0, e_y - w_left)
    violation_right = jnp.maximum(0.0, -e_y - w_right)
    hard_violation = jnp.maximum(violation_left, violation_right)
    boundary_cost = (
            jnp.any(hard_violation > 0.0) * w_boundary
            + w_boundary * jnp.sum(hard_violation ** 2)
    )

    # Net acceleration = sqrt(a_tangential^2 + a_centripetal^2)
    # Centripetal acceleration considers both steering deflection and local track curvature
    a_centripetal = jnp.maximum((v ** 2 / wheelbase) * jnp.abs(jnp.tan(theta)), v ** 2 * jnp.abs(kappa))
    a_net = jnp.sqrt(a_tangential ** 2 + a_centripetal ** 2 + 1e-4)
    friction_violation = jnp.maximum(0.0, a_net - max_friction_accel)
    friction_cost = w_accel * jnp.sum(friction_violation ** 2) * dt

    # Separate acceleration and steering rate effort penalties:
    effort_a = w_effort * jnp.sum(a_tangential ** 2) * dt
    effort_omega = w_effort_omega * jnp.sum(omega ** 2) * dt

    # Steering rate rate-of-change penalty (suppresses high-frequency chatter):
    domega = omega[1:] - omega[:-1]
    domega_cost = w_domega * jnp.sum(domega ** 2) * dt

    # Soft speed ceiling penalty at v_max (MPPI regulates speed autonomously without raceline profile):
    v_excess = jnp.maximum(0.0, v - v_max)
    speed_cost = w_speed * jnp.sum(v_excess ** 2) * dt

    # Collision cost:
    collision_cost = check_trajectory_collisions(trajectory, cones, A, b, collision_margin, dt, w_collision, w_collision_hard)

    return (progress_cost
            + contour_cost
            + heading_cost
            + friction_cost
            + effort_a
            + effort_omega
            + domega_cost
            + collision_cost
            + speed_cost
            + boundary_cost)