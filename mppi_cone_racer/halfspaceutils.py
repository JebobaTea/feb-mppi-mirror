import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

def generate_rectangular_halfspaces(
    wheelbase,
    front_overhang,
    rear_overhang,
    half_width,
    inflation: float = 0.0,
    as_jax: bool = True,
):
    """Generates 4 half-spaces for an axis-aligned bounding box centered at the rear axle.

    Bounds in body frame:
        x_min = -rear_overhang - inflation
        x_max = (wheelbase + front_overhang) + inflation
        y_min = -half_width - inflation
        y_max =  half_width + inflation

    Returns:
        A: (4, 2) array of unit normal vectors pointing outwards.
        b: (4,) array of offsets such that A @ p_body <= b inside the vehicle.
    """
    x_max = wheelbase + front_overhang + inflation
    x_min = -rear_overhang - inflation
    y_max = half_width + inflation
    y_min = -half_width - inflation

    # Outward unit normals:
    # 1. Front:  +1*x + 0*y <= x_max
    # 2. Rear:   -1*x + 0*y <= -x_min (since x >= x_min => -x <= -x_min)
    # 3. Left:    0*x + 1*y <= y_max
    # 4. Right:   0*x - 1*y <= -y_min (since y >= y_min => -y <= -y_min)
    A = np.array([
        [ 1.0,  0.0],  # Front
        [-1.0,  0.0],  # Rear
        [ 0.0,  1.0],  # Left
        [ 0.0, -1.0],  # Right
    ], dtype=np.float32)

    b = np.array([
        x_max,
        -x_min,
        y_max,
        -y_min,
    ], dtype=np.float32)

    if as_jax and jnp is not None:
        return jnp.array(A), jnp.array(b)
    return A, b


def generate_chamfered_halfspaces(
    wheelbase,
    front_overhang,
    rear_overhang,
    half_width,
    chamfer_front: float = 0.06,  # 6 cm chamfer cut at front corners
    chamfer_rear: float = 0.04,   # 4 cm chamfer cut at rear corners
    inflation: float = 0.0,
    as_jax: bool = True,
):
    """Generates 8 half-spaces (octagonal bounding envelope).

    Chamfering cuts off the 90-degree front and rear corners, matching the curved
    bumper profile and preventing false collision penalties when turning near cones.
    """
    # 8 vertices ordered counter-clockwise (CCW)
    xf = wheelbase + front_overhang + inflation
    xr = -rear_overhang - inflation
    yw = half_width + inflation

    vertices = np.array([
        [xf - chamfer_front, -yw],                  # Front-right lower
        [xf,                 -yw + chamfer_front],  # Front-right nose
        [xf,                  yw - chamfer_front],  # Front-left nose
        [xf - chamfer_front,  yw],                  # Front-left upper
        [xr + chamfer_rear,   yw],                  # Rear-left upper
        [xr,                  yw - chamfer_rear],   # Rear-left tail
        [xr,                 -yw + chamfer_rear],   # Rear-right tail
        [xr + chamfer_rear,  -yw],                  # Rear-right lower
    ], dtype=np.float32)

    return polygon_to_halfspaces(vertices, as_jax=as_jax)


def polygon_to_halfspaces(vertices: np.ndarray, as_jax: bool = True):
    """Converts any 2D convex polygon (N, 2) in CCW order into normalized (A, b).

    For an edge from v_i to v_{i+1} with tangent (dx, dy):
        Outward unit normal n = (dy, -dx) / ||(dx, dy)||
        Offset b = n @ v_i
    """
    v = np.asarray(vertices, dtype=np.float32)
    N = len(v)
    v_next = np.roll(v, -1, axis=0)

    # Edge vector: v_{i+1} - v_i
    edges = v_next - v

    # Outward normal for CCW winding: (dy, -dx)
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    A = normals / norms  # Unit outward normals

    # b_k = a_k @ v_k (evaluated at any vertex on the edge)
    b = np.einsum("ij,ij->i", A, v)

    if as_jax and jnp is not None:
        return jnp.array(A), jnp.array(b)
    return A, b