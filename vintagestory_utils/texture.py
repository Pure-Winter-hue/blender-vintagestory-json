import math
import numpy as np
import logging
import bpy
import bmesh
from mathutils import Vector

log = logging.getLogger(__name__)

# single pre-defined axes vectors, for convenience
X_AXIS = np.array([1.0, 0.0, 0.0])
X_NEG_AXIS = np.array([-1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Y_NEG_AXIS = np.array([0.0, -1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])
Z_NEG_AXIS = np.array([0.0, 0.0, -1.0])

# pre-defined axes array block
AXIS = np.array([
    [1.0, 0.0, 0.0], # X
    [-1.0, 0.0, 0.0], # -X
    [0.0, 1.0, 0.0], # Y
    [0.0, -1.0, 0.0], # -Y
    [0.0, 0.0, 1.0], # Z
    [0.0, 0.0, -1.0], # -Z
])

# indices into AXIS array
IDX_X_AXIS = 0
IDX_X_NEG_AXIS = 1
IDX_Y_AXIS = 2
IDX_Y_NEG_AXIS = 3
IDX_Z_AXIS = 4
IDX_Z_NEG_AXIS = 5


def _find_view3d_region_data(context):
    """Best-effort find a VIEW_3D RegionView3D (region_3d).

    Notes:
    - Operators invoked from the 3D View N-panel will usually have
      context.region_data set.
    - If invoked from elsewhere, we scan screen areas for a 3D View.
    """
    region_3d = getattr(context, "region_data", None)
    if region_3d is not None and hasattr(region_3d, "view_rotation"):
        return region_3d

    win = getattr(context, "window", None)
    screen = getattr(win, "screen", None) if win else None
    if not screen:
        return None

    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type == "VIEW_3D" and getattr(space, "region_3d", None) is not None:
                return space.region_3d
    return None


def _view_basis_world(context):
    """Return an orthonormal view-aligned basis in world space.

    Returns (x_right, y_view_dir, z_up, mat4_world_to_basis).

    Convention:
    - x_right points to the viewport's right.
    - z_up points to the viewport's up.
    - y_view_dir points *into the scene* (the direction the view looks).

    This is chosen so that the face *facing the viewer* has normal aligned to
    -y in this basis (matching the addon's default front face = "-y").
    """
    region_3d = _find_view3d_region_data(context)
    if region_3d is None:
        return None

    # view_rotation rotates from view space into world space.
    q = region_3d.view_rotation

    x_right = (q @ Vector((1.0, 0.0, 0.0))).normalized()
    z_up = (q @ Vector((0.0, 1.0, 0.0))).normalized()
    y_view_dir = (q @ Vector((0.0, 0.0, -1.0))).normalized()  # forward/look dir

    # Orthonormalize defensively (rare edge cases).
    x_right = x_right.normalized()
    z_up = (z_up - x_right * z_up.dot(x_right)).normalized()
    # Use a right-handed basis: up × right = forward (matches Blender view dir).
    y_view_dir = z_up.cross(x_right).normalized()

    # world_to_basis is a rotation matrix with basis vectors as rows:
    # p_basis = world_to_basis @ p_world
    m3 = np.array([
        [x_right.x, x_right.y, x_right.z],
        [y_view_dir.x, y_view_dir.y, y_view_dir.z],
        [z_up.x, z_up.y, z_up.z],
    ], dtype=np.float64)
    m4 = np.eye(4, dtype=np.float64)
    m4[0:3, 0:3] = m3
    return x_right, y_view_dir, z_up, m4



def _get_active_image_size(context):
    """Try to find an image size (w, h) for pixel-relative UV scaling."""
    # 1) Image editor
    try:
        screen = context.window.screen if context and context.window else None
        if screen:
            for area in screen.areas:
                if area.type == "IMAGE_EDITOR":
                    space = area.spaces.active
                    img = getattr(space, "image", None)
                    if img and getattr(img, "size", None):
                        w, h = int(img.size[0]), int(img.size[1])
                        if w > 0 and h > 0:
                            return w, h
    except Exception:
        pass

    # 2) Active material image texture nodes
    obj = getattr(context, "active_object", None)
    if obj and getattr(obj, "type", "") == "MESH":
        for slot in getattr(obj, "material_slots", []) or []:
            mat = getattr(slot, "material", None)
            if not mat or not getattr(mat, "use_nodes", False):
                continue
            nt = getattr(mat, "node_tree", None)
            if not nt:
                continue
            for node in nt.nodes:
                if getattr(node, "type", "") == "TEX_IMAGE":
                    img = getattr(node, "image", None)
                    if img and getattr(img, "size", None):
                        w, h = int(img.size[0]), int(img.size[1])
                        if w > 0 and h > 0:
                            return w, h
    return None


def _selected_cuboid_objects(context):
    """Return list of objects to unwrap.

    Behavior:
    - Object mode: unwrap selected objects.
    - Edit mode (including multi-object edit): unwrap only objects whose
      *entire* cuboid (all 6 faces) are selected. This enables unwrapping
      body vs legs separately without touching unselected cuboids.
    """
    mode = getattr(context, "mode", "OBJECT")
    if mode == "OBJECT":
        return list(getattr(bpy.context, "selected_objects", []) or [])

    if mode != "EDIT_MESH":
        # Fallback: behave like object mode selection.
        return list(getattr(bpy.context, "selected_objects", []) or [])

    objs = []
    edit_objs = list(getattr(context, "objects_in_mode", []) or [])
    if not edit_objs and getattr(context, "edit_object", None) is not None:
        edit_objs = [context.edit_object]

    for obj in edit_objs:
        mesh = getattr(obj, "data", None)
        if not isinstance(mesh, bpy.types.Mesh):
            continue
        if len(mesh.vertices) != 8 or len(mesh.polygons) != 6:
            continue

        try:
            bm = bmesh.from_edit_mesh(mesh)
        except Exception:
            continue

        total_faces = len(bm.faces)
        selected_faces = sum(1 for f in bm.faces if f.select)
        if total_faces == 6 and selected_faces == 6:
            objs.append(obj)

    return objs


# transform/projection for cuboid faces along each axis into an XY plane
# hard-coded based on pre-defined ways for how we should
# look at each face (which determines projection axes)
MAT_Y_AX_TO_XY = np.array([ # +Y axis
    [-1.0, 0.0, 0.0], # x <- -x
    [0.0, 0.0, 1.0],  # y <- z
    [0.0, 0.0, 0.0],  # z <- 0
])
MAT_Y_NEG_AX_TO_XY = np.array([ # -Y axis
    [1.0, 0.0, 0.0], # x <- x
    [0.0, 0.0, 1.0], # y <- z
    [0.0, 0.0, 0.0], # z <- 0
])
MAT_X_AX_TO_XY = np.array([ # -X axis
    [0.0, 1.0, 0.0], # x <- y
    [0.0, 0.0, 1.0], # y <- z
    [0.0, 0.0, 0.0], # z <- 0
])
MAT_X_NEG_AX_TO_XY = np.array([ # +X axis
    [0.0, -1.0, 0.0], # x <- -y
    [0.0, 0.0, 1.0],  # y <- z
    [0.0, 0.0, 0.0],  # z <- 0
])
# for Z faces, we need to do 90 deg rotations depending on front face
# for notation, this will be CCW rotations around the Z axis
# e.g. rotate X axis counterclockwise by angle
MAT_Z_AX_TO_XY = np.array([ #  +Z axis
    [1.0, 0.0, 0.0], # x <- x
    [0.0, 1.0, 0.0], # y <- y
    [0.0, 0.0, 0.0], # z <- 0
])
MAT_Z_AX_ROT90_TO_XY = np.array([ #  +Z axis
    [0.0, -1.0, 0.0], # x <- -y
    [1.0, 0.0, 0.0],  # y <- x
    [0.0, 0.0, 0.0],  # z <- 0
])
MAT_Z_AX_ROT180_TO_XY = np.array([ #  +Z axis
    [-1.0, 0.0, 0.0], # x <- -x
    [0.0, -1.0, 0.0], # y <- -y
    [0.0, 0.0, 0.0],  # z <- 0
])
MAT_Z_AX_ROT270_TO_XY = np.array([ #  +Z axis
    [0.0, 1.0, 0.0],  # x <- y
    [-1.0, 0.0, 0.0], # y <- -x
    [0.0, 0.0, 0.0],  # z <- 0
])

MAT_Z_NEG_AX_TO_XY = np.array([ # -Z axis
    [1.0, 0.0, 0.0],  # x <- x
    [0.0, -1.0, 0.0], # y <- -y
    [0.0, 0.0, 0.0],  # z <- 0
])
MAT_Z_NEG_AX_ROT90_TO_XY = np.array([ # -Z axis
    [0.0, 1.0, 0.0], # x <- y
    [1.0, 0.0, 0.0], # y <- x
    [0.0, 0.0, 0.0], # z <- 0
])
MAT_Z_NEG_AX_ROT180_TO_XY = np.array([ # -Z axis
    [-1.0, 0.0, 0.0], # x <- -x
    [0.0, 1.0, 0.0],  # y <- y
    [0.0, 0.0, 0.0],  # z <- 0
])
MAT_Z_NEG_AX_ROT270_TO_XY = np.array([ # -Z axis
    [0.0, -1.0, 0.0], # x <- y
    [-1.0, 0.0, 0.0], # y <- x
    [0.0, 0.0, 0.0],  # z <- 0
])

def loop_is_clockwise(coords):
    """Detect if loop of 2d coordinates is clockwise or counterclockwise.
    Inputs:
    - coords: List of 2d array indexed coords, [p0, p1, p2, ... pN]
              where each is array indexed as p[0] = p0.x, p[1] = p0.y
    Returns:
    - True if clockwise, False if counterclockwise
    """
    num_coords = len(coords)
    area = 0
    
    # use polygon winding area to detect if loop is clockwise or counterclockwise
    for i in range(num_coords):
        # next index
        k = i + 1 if i < num_coords - 1 else 0
        area += (coords[k][0] - coords[i][0]) * (coords[k][1] + coords[i][1])
    
    # clockwise if area positive
    return area > 0


def index_of_vmin(
    face_verts: np.ndarray, # shape (4, 3)
):
    """Determine index and vertex of vmin or "v0" (bottom left) vertex of a
    face in the XY plane as defined in format below:
    
             ^ y axis
             |
             |  v2
      v3..---^^\
        \     / \
         \   x---\---------> x axis
          \       \
           \       \
            \..---^^^ v1
            v0
          = vmin
    
    Note the face is not necessarily axis-aligned which is
    the challenge. Current method:
    1. Find two points with smallest x values. If any points have same x value,
       use the one with smallest y value.
    2. Between those two points, pick the one with smallest y value.
       Use that as the "vmin" point.
    Returns index of point.
    """
    idx_min_x = 0
    idx_2nd_min_x = math.inf
    x_min = face_verts[0,0]
    y_min = face_verts[0,1]
    x_2nd_min = math.inf
    y_2nd_min = math.inf

    for i in range(1, 4):
        x = face_verts[i,0]
        y = face_verts[i,1]
        if x < x_min:
            idx_2nd_min_x = idx_min_x
            x_2nd_min = x_min
            y_2nd_min = y_min
            idx_min_x = i
            x_min = x
            y_min = y
        elif x == x_min:
            if y < y_min:
                idx_2nd_min_x = idx_min_x
                x_2nd_min = x_min
                y_2nd_min = y_min
                idx_min_x = i
                x_min = x
                y_min = y
            else:
                idx_2nd_min_x = i
                x_2nd_min = x
                y_2nd_min = y
        elif x < x_2nd_min:
            idx_2nd_min_x = i
            x_2nd_min = x
            y_2nd_min = y
        elif x == x_2nd_min:
            if y < y_2nd_min:
                idx_2nd_min_x = i
                x_2nd_min = x
                y_2nd_min = y
        else:
            pass # x > x_min and x > x_2nd_min, do nothing
    
    # return x_min or x_2nd_min, whichever has smaller y value
    if y_min <= y_2nd_min:
        return idx_min_x
    else:
        return idx_2nd_min_x


def _unwrap_cuboid_objects(
    *,
    context,
    objects,
    front_face: str,
    use_local_space_normals: bool,
    scale_to_unit: bool,
    world_to_basis_4x4: np.ndarray | None = None,
):
    """Core cuboid UV unwrap implementation.

    If world_to_basis_4x4 is provided, all world-space vertex coordinates and
    normals are additionally rotated into that basis before face classification
    and projection. This enables a "project from view"-style unwrap while still
    using the addon's VS-style cuboid layout.
    """

    # map `front_face` string arg to integer axis index
    if front_face == "+x":
        front_axis = X_AXIS
        mat_front_face_to_xy = MAT_X_AX_TO_XY
        mat_back_face_to_xy = MAT_X_NEG_AX_TO_XY
        mat_left_face_to_xy = MAT_Y_NEG_AX_TO_XY
        mat_right_face_to_xy = MAT_Y_AX_TO_XY
        mat_up_face_to_xy = MAT_Z_AX_ROT270_TO_XY
        mat_down_face_to_xy = MAT_Z_NEG_AX_ROT90_TO_XY
    elif front_face == "-x":
        front_axis = X_NEG_AXIS
        mat_front_face_to_xy = MAT_X_NEG_AX_TO_XY
        mat_back_face_to_xy = MAT_X_AX_TO_XY
        mat_left_face_to_xy = MAT_Y_AX_TO_XY
        mat_right_face_to_xy = MAT_Y_NEG_AX_TO_XY
        mat_up_face_to_xy = MAT_Z_AX_ROT90_TO_XY
        mat_down_face_to_xy = MAT_Z_NEG_AX_ROT270_TO_XY
    elif front_face == "+y":
        front_axis = Y_AXIS
        mat_front_face_to_xy = MAT_Y_AX_TO_XY
        mat_back_face_to_xy = MAT_Y_NEG_AX_TO_XY
        mat_left_face_to_xy = MAT_X_AX_TO_XY
        mat_right_face_to_xy = MAT_X_NEG_AX_TO_XY
        mat_up_face_to_xy = MAT_Z_AX_ROT180_TO_XY
        mat_down_face_to_xy = MAT_Z_NEG_AX_ROT180_TO_XY
    elif front_face == "-y":
        front_axis = Y_NEG_AXIS
        mat_front_face_to_xy = MAT_Y_NEG_AX_TO_XY
        mat_back_face_to_xy = MAT_Y_AX_TO_XY
        mat_left_face_to_xy = MAT_X_NEG_AX_TO_XY
        mat_right_face_to_xy = MAT_X_AX_TO_XY
        mat_up_face_to_xy = MAT_Z_AX_TO_XY
        mat_down_face_to_xy = MAT_Z_NEG_AX_TO_XY
    # z-axis are different: depends on if we want aligned to y- or x-axis,
    # matrices here are hard-coded since the rotations do not follow an
    # easy pattern for re-use
    elif front_face == "+z,-x":
        front_axis = Z_AXIS
        mat_front_face_to_xy = MAT_Z_AX_ROT270_TO_XY
        mat_back_face_to_xy = MAT_Z_NEG_AX_ROT270_TO_XY
        mat_left_face_to_xy = np.array([
            [0.0, 0.0, 1.0], # x <- z
            [-1.0, 0.0, 0.0], # y <- -x
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_right_face_to_xy = np.array([
            [0.0, 0.0, -1.0], # x <- -z
            [-1.0, 0.0, 0.0], # y <- -x
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_up_face_to_xy = np.array([
            [0.0, 1.0, 0.0], # x <- y
            [0.0, 0.0, -1.0], # y <- -z
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_down_face_to_xy = np.array([
            [0.0, 1.0, 0.0], # x <- y
            [0.0, 0.0, 1.0], # y <- z
            [0.0, 0.0, 0.0], # z <- 0
        ])
    elif front_face == "-z,+x":
        front_axis = Z_NEG_AXIS
        mat_front_face_to_xy = MAT_Z_NEG_AX_ROT90_TO_XY
        mat_back_face_to_xy = MAT_Z_AX_ROT90_TO_XY
        mat_left_face_to_xy = np.array([
            [0.0, 0.0, -1.0], # x <- -z
            [1.0, 0.0, 0.0], # y <- x
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_right_face_to_xy = np.array([
            [0.0, 0.0, 1.0], # x <- z
            [1.0, 0.0, 0.0], # y <- x
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_up_face_to_xy = np.array([
            [0.0, 1.0, 0.0], # x <- y
            [0.0, 0.0, 1.0], # y <- z
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_down_face_to_xy = np.array([
            [0.0, 1.0, 0.0], # x <- y
            [0.0, 0.0, -1.0], # y <- -z
            [0.0, 0.0, 0.0], # z <- 0
        ])
    elif front_face == "+z,+y":
        front_axis = Z_AXIS
        mat_front_face_to_xy = MAT_Z_AX_TO_XY
        mat_back_face_to_xy = MAT_Z_NEG_AX_ROT180_TO_XY
        mat_left_face_to_xy = np.array([
            [0.0, 0.0, 1.0], # x <- z
            [0.0, 1.0, 0.0], # y <- y
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_right_face_to_xy = np.array([
            [0.0, 0.0, -1.0], # x <- -z
            [0.0, 1.0, 0.0], # y <- y
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_up_face_to_xy = np.array([
            [1.0, 0.0, 0.0], # x <- x
            [0.0, 0.0, -1.0], # y <- -z
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_down_face_to_xy = np.array([
            [1.0, 0.0, 0.0], # x <- x
            [0.0, 0.0, 1.0], # y <- z
            [0.0, 0.0, 0.0], # z <- 0
        ])
    elif front_face == "-z,-y":
        front_axis = Z_NEG_AXIS
        mat_front_face_to_xy = np.array([ #  -Z axis
            [1.0, 0.0, 0.0], # x <- x
            [0.0, -1.0, 0.0], # y <- -y
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_back_face_to_xy = np.array([ #  +Z axis
            [1.0, 0.0, 0.0], # x <- x
            [0.0, 1.0, 0.0], # y <- y
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_left_face_to_xy = np.array([
            [0.0, 0.0, -1.0], # x <- -z
            [0.0, -1.0, 0.0], # y <- -y
            [0.0, 0.0, 0.0],  # z <- 0
        ])
        mat_right_face_to_xy = np.array([
            [0.0, 0.0, 1.0],  # x <- z
            [0.0, -1.0, 0.0], # y <- -y
            [0.0, 0.0, 0.0],  # z <- 0
        ])
        mat_up_face_to_xy = np.array([
            [1.0, 0.0, 0.0], # x <- x
            [0.0, 0.0, 1.0], # y <- z
            [0.0, 0.0, 0.0], # z <- 0
        ])
        mat_down_face_to_xy = np.array([
            [1.0, 0.0, 0.0],  # x <- x
            [0.0, 0.0, -1.0], # y <- -z
            [0.0, 0.0, 0.0],  # z <- 0
        ])
    else:
        raise Exception(
            f"Invalid front_face: {front_face}, must be one of: +x, -x, +y, -y, +z,-x, -z,+x, +z,+y, -z,-y"
        )

    # uv face format indices
    IDX_UV_FACE_LEFT = 0
    IDX_UV_FACE_FRONT = 1
    IDX_UV_FACE_RIGHT = 2
    IDX_UV_FACE_BACK = 3
    IDX_UV_FACE_UP = 4
    IDX_UV_FACE_DOWN = 5

    # For mapping direction indices into the LEFT-FRONT-RIGHT-BACK-UP-DOWN format.
    # These are stable constants across front_face modes.
    # (We keep them here to avoid relying on outer class scope.)

    for obj in objects:
        try:
            mesh = obj.data
            if not isinstance(mesh, bpy.types.Mesh):
                continue

            # skip non cuboid meshes, print warning
            if len(mesh.vertices) != 8 or len(mesh.polygons) != 6:
                log.warning(f"Skipping UV unwrap of non-cuboid mesh: {obj.name}")
                continue

            # Ensure there is an active UV map.
            if mesh.uv_layers.active is None:
                mesh.uv_layers.new(name="UVMap")
            uv_layer = mesh.uv_layers.active.data

            vertices_local = np.ones((4, 8))  # 8 verts, (x,y,z,1)
            for i, v in enumerate(mesh.vertices):
                vertices_local[0:3, i] = v.co

            matrix_world = np.asarray(obj.matrix_world, dtype=np.float64)
            vertices_world = matrix_world @ vertices_local

            if world_to_basis_4x4 is not None:
                vertices = world_to_basis_4x4 @ vertices_world
            else:
                vertices = vertices_world

            if use_local_space_normals:
                normal_matrix = np.identity(3)
            else:
                try:
                    normal_matrix = np.transpose(np.linalg.inv(matrix_world[0:3, 0:3]))
                except Exception:
                    log.warning(f"Non-invertible matrix for: {obj.name}, using its world matrix instead")
                    normal_matrix = matrix_world[0:3, 0:3]

            # gather original mesh face vertices and normals
            mesh_face_uv_loop_start = np.zeros((6,), dtype=int)  # (face,)
            mesh_face_vert_indices = np.zeros((6, 4), dtype=int)  # (face, vert)
            mesh_face_vertices = np.zeros((6, 4, 4), dtype=np.float64)  # (face, vert, xyzw)
            mesh_face_normals = np.zeros((6, 3), dtype=np.float64)  # (face, xyz)

            for i, face in enumerate(mesh.polygons):
                mesh_face_uv_loop_start[i] = face.loop_start
                mesh_face_vertices[i, :, :] = np.stack(
                    [vertices[:, v] for v in face.vertices],
                    axis=0,
                )
                mesh_face_vert_indices[i, :] = face.vertices
                mesh_face_normals[i, :] = np.array(face.normal)

            # world space face normals
            mesh_face_normals_world = normal_matrix @ mesh_face_normals.transpose()
            mesh_face_normals_world = mesh_face_normals_world.transpose()

            if world_to_basis_4x4 is not None:
                m3 = world_to_basis_4x4[0:3, 0:3]
                mesh_face_normals_world = (m3 @ mesh_face_normals_world.transpose()).transpose()

            # detect index of closest matching mesh front face
            front_index = np.argmax(np.sum(mesh_face_normals_world * front_axis, axis=1), axis=0)

            # Determine which faces are adjacent to the front face.
            # (This is the original addon's robust method for assigning
            # LEFT/RIGHT/UP/DOWN/BACK in a unique way.)

            # face local indices f0..f3 -> mesh vertex indices v0..v3
            mesh_face_f0_f1_f2_f3 = np.zeros((6, 4), dtype=int)
            mesh_face_v0_v1_v2_v3 = np.zeros((6, 4), dtype=int)
            # direction indices for each mesh face (into LEFT/FRONT/... array)
            mesh_face_directions = -1 * np.ones((6,), dtype=int)

            # helper: compute face projected coords for determining vmin
            def _project_face_xy(face_idx: int, mat_to_xy: np.ndarray):
                return mat_to_xy @ mesh_face_vertices[face_idx, :, :3].transpose()

            # mark front face
            mesh_face_directions[front_index] = IDX_UV_FACE_FRONT

            # compute v0-v3 ordering for each face based on its "look" matrix
            for i in range(0, 6):
                if i == front_index:
                    face_verts_xy = _project_face_xy(i, mat_front_face_to_xy)
                else:
                    # pick the matrix by the face normal direction later
                    # (we will overwrite below once direction is known)
                    face_verts_xy = _project_face_xy(i, mat_front_face_to_xy)

                f0 = index_of_vmin(face_verts_xy.transpose())
                is_cw = loop_is_clockwise(face_verts_xy[0:3, 0:3].transpose())
                if is_cw:
                    f3 = (f0 + 1) % 4
                    f2 = (f0 + 2) % 4
                    f1 = (f0 + 3) % 4
                else:
                    f1 = (f0 + 1) % 4
                    f2 = (f0 + 2) % 4
                    f3 = (f0 + 3) % 4

                v0 = mesh_face_vert_indices[i, f0]
                v1 = mesh_face_vert_indices[i, f1]
                v2 = mesh_face_vert_indices[i, f2]
                v3 = mesh_face_vert_indices[i, f3]
                mesh_face_f0_f1_f2_f3[i, :] = [f0, f1, f2, f3]
                mesh_face_v0_v1_v2_v3[i, :] = [v0, v1, v2, v3]

            # Use vertex adjacency against the front face's ordered verts.
            # Identify left/right/up/down.
            f_front_v0, f_front_v1, f_front_v2, f_front_v3 = mesh_face_v0_v1_v2_v3[front_index, :]

            def _face_has_verts(face_idx: int, a: int, b: int) -> bool:
                vv = set(mesh_face_vert_indices[face_idx, :].tolist())
                return (a in vv) and (b in vv)

            # Find faces adjacent to front via the shared edges.
            for i in range(0, 6):
                if i == front_index:
                    continue
                if mesh_face_directions[i] != -1:
                    continue

                if _face_has_verts(i, f_front_v0, f_front_v3):
                    mesh_face_directions[i] = IDX_UV_FACE_LEFT
                elif _face_has_verts(i, f_front_v1, f_front_v2):
                    mesh_face_directions[i] = IDX_UV_FACE_RIGHT
                elif _face_has_verts(i, f_front_v2, f_front_v3):
                    mesh_face_directions[i] = IDX_UV_FACE_UP
                elif _face_has_verts(i, f_front_v0, f_front_v1):
                    mesh_face_directions[i] = IDX_UV_FACE_DOWN

            # Remaining unassigned face is the back.
            back_candidates = [i for i in range(0, 6) if mesh_face_directions[i] == -1]
            if len(back_candidates) != 1:
                raise Exception(
                    "Invalid cuboid mesh: could not uniquely determine back face from adjacency"
                )
            mesh_face_directions[back_candidates[0]] = IDX_UV_FACE_BACK

            # Now we have per-face directions. Fix the v0..v3 ordering for each
            # face using the correct projection matrix for that direction.
            for i in range(0, 6):
                direction_index = mesh_face_directions[i]
                if direction_index == IDX_UV_FACE_LEFT:
                    face_verts_xy = _project_face_xy(i, mat_left_face_to_xy)
                elif direction_index == IDX_UV_FACE_FRONT:
                    face_verts_xy = _project_face_xy(i, mat_front_face_to_xy)
                elif direction_index == IDX_UV_FACE_RIGHT:
                    face_verts_xy = _project_face_xy(i, mat_right_face_to_xy)
                elif direction_index == IDX_UV_FACE_BACK:
                    face_verts_xy = _project_face_xy(i, mat_back_face_to_xy)
                elif direction_index == IDX_UV_FACE_UP:
                    face_verts_xy = _project_face_xy(i, mat_up_face_to_xy)
                elif direction_index == IDX_UV_FACE_DOWN:
                    face_verts_xy = _project_face_xy(i, mat_down_face_to_xy)
                else:
                    raise Exception("Invalid face direction assignment")

                f0 = index_of_vmin(face_verts_xy.transpose())
                is_cw = loop_is_clockwise(face_verts_xy[0:3, 0:3].transpose())
                if is_cw:
                    f3 = (f0 + 1) % 4
                    f2 = (f0 + 2) % 4
                    f1 = (f0 + 3) % 4
                else:
                    f1 = (f0 + 1) % 4
                    f2 = (f0 + 2) % 4
                    f3 = (f0 + 3) % 4

                v0 = mesh_face_vert_indices[i, f0]
                v1 = mesh_face_vert_indices[i, f1]
                v2 = mesh_face_vert_indices[i, f2]
                v3 = mesh_face_vert_indices[i, f3]
                mesh_face_f0_f1_f2_f3[i, :] = [f0, f1, f2, f3]
                mesh_face_v0_v1_v2_v3[i, :] = [v0, v1, v2, v3]

            # Create uvs based on face width/height and place them in
            # LEFT-FRONT-RIGHT-BACK-UP-DOWN layout.
            face_uv_width_height = np.zeros((6, 2))
            face_uv_xy = np.zeros((6, 4, 2))
            face_uv_loop_start = np.zeros((6,), dtype=int)

            for i in range(0, 6):
                direction_index = mesh_face_directions[i]
                f0, f1, f2, f3 = mesh_face_f0_f1_f2_f3[i, :]
                v0, v1, v2, v3 = mesh_face_v0_v1_v2_v3[i, :]

                face_width = np.linalg.norm(vertices[:3, v0] - vertices[:3, v1])
                face_height = np.linalg.norm(vertices[:3, v1] - vertices[:3, v2])

                face_uv_width_height[direction_index, 0] = face_width
                face_uv_width_height[direction_index, 1] = face_height

                face_uv_xy[direction_index, f0, :] = (0.0, 0.0)
                face_uv_xy[direction_index, f1, :] = (face_width, 0.0)
                face_uv_xy[direction_index, f2, :] = (face_width, face_height)
                face_uv_xy[direction_index, f3, :] = (0.0, face_height)

                face_uv_loop_start[direction_index] = mesh_face_uv_loop_start[i]

            uv_offset = np.zeros((6, 2))
            uv_offset[IDX_UV_FACE_FRONT, :] = (face_uv_width_height[IDX_UV_FACE_LEFT, 0], 0.0)
            uv_offset[IDX_UV_FACE_RIGHT, :] = (
                face_uv_width_height[IDX_UV_FACE_LEFT, 0] + face_uv_width_height[IDX_UV_FACE_FRONT, 0],
                0.0,
            )
            uv_offset[IDX_UV_FACE_BACK, :] = (
                face_uv_width_height[IDX_UV_FACE_LEFT, 0]
                + face_uv_width_height[IDX_UV_FACE_FRONT, 0]
                + face_uv_width_height[IDX_UV_FACE_RIGHT, 0],
                0.0,
            )
            uv_offset[IDX_UV_FACE_UP, :] = (
                face_uv_width_height[IDX_UV_FACE_LEFT, 0],
                face_uv_width_height[IDX_UV_FACE_LEFT, 1],
            )
            uv_offset[IDX_UV_FACE_DOWN, :] = (
                face_uv_width_height[IDX_UV_FACE_LEFT, 0] + face_uv_width_height[IDX_UV_FACE_UP, 0],
                face_uv_width_height[IDX_UV_FACE_LEFT, 1],
            )

            uv_xy = uv_offset[:, np.newaxis, :] + face_uv_xy

            if scale_to_unit:
                uv_scale = 1.0 / uv_xy.max(axis=(0, 1, 2))
                uv_xy = uv_scale * uv_xy

            # update face uvs
            for i in range(0, 6):
                idx = face_uv_loop_start[i]
                uv_layer[idx].uv = (uv_xy[i, 0, 0], uv_xy[i, 0, 1])
                uv_layer[idx + 1].uv = (uv_xy[i, 1, 0], uv_xy[i, 1, 1])
                uv_layer[idx + 2].uv = (uv_xy[i, 2, 0], uv_xy[i, 2, 1])
                uv_layer[idx + 3].uv = (uv_xy[i, 3, 0], uv_xy[i, 3, 1])

        except Exception as e:
            import traceback

            traceback.print_exc()
            err_msg = f"Error unwrapping cuboid: {obj.name}, {e}"
            log.error(err_msg)
            # Prefer reporting through operator if available.
            op = getattr(context, "active_operator", None)
            if op is not None and hasattr(op, "report"):
                op.report({"ERROR"}, err_msg)



# =============================================================================
# View-oriented "project from view" cuboid unwrap
# =============================================================================

def _unwrap_project_from_view_groups(*, context, objects, scale_to_unit: bool, mode: str = "AUTO", margin_px: int = 2):
    """Project selected cuboid faces into 6 view-relative groups.

    Groups (relative to the active 3D View):
      - FRONT: faces pointing toward the viewer (normal ~ -view_dir)
      - BACK:  faces pointing away (normal ~ +view_dir)
      - LEFT/RIGHT: faces pointing to screen-left / screen-right
      - UP/DOWN: faces pointing to screen-up / screen-down

    Within each group, faces are projected so they appear with the same
    orientation as they would in that corresponding orthographic view.

    Finally, the 6 groups are arranged into a VS/Minecraft-style cuboid net:

             [  UP  ]
    [ LEFT ][FRONT][RIGHT][ BACK ]
             [ DOWN ]
    """
    vb = _view_basis_world(context)
    if vb is None:
        raise RuntimeError("No active VIEW_3D region found")
    X, Y, Z, _m4 = vb  # X=screen right, Y=into scene, Z=screen up (all in world space)

    # Virtual view projection axes for each group: (screen_right, screen_up) in world space
    proj_axes = {
        "FRONT": (X, Z),          # looking along +Y, visible normals ~ -Y
        "BACK": (-X, Z),          # looking along -Y (mirror X)
        "RIGHT": (-Y, Z),         # looking along -X
        "LEFT": (Y, Z),           # looking along +X
        "UP": (X, -Y),            # looking along -Z
        "DOWN": (X, Y),           # looking along +Z
    }

    # Net placement (col, row) in a 4x3 grid (row 0 = bottom)
    net_pos = {
        "LEFT": (0, 1),
        "FRONT": (1, 1),
        "RIGHT": (2, 1),
        "BACK": (3, 1),
        "UP": (1, 2),
        "DOWN": (1, 0),
    }

    # Accumulate raw projections per loop so we can normalize + lay out after bounds are known.
    # Each entry: (mesh, loop_index, group, u_raw, v_raw)
    records = []
    bounds = {k: [math.inf, -math.inf, math.inf, -math.inf] for k in proj_axes.keys()}  # umin, umax, vmin, vmax

    def _ensure_uv(mesh):
        if mesh.uv_layers.active is None:
            mesh.uv_layers.new(name="UVMap")
        return mesh.uv_layers.active.data

    for obj in objects:
        if obj is None or obj.type != "MESH":
            continue
        mesh = obj.data
        if mesh is None:
            continue

        uv_data = _ensure_uv(mesh)

        mw = obj.matrix_world
        mwn = mw.to_3x3()

        # Heuristic: treat as cuboid-like only if it has at least 6 faces.
        # We still categorize faces individually (doesn't assume perfect boxes).
        for face in mesh.polygons:
            # World-space normal
            n_world = (mwn @ face.normal).normalized()
            # Components in view basis
            nx = n_world.dot(X)
            ny = n_world.dot(Y)
            nz = n_world.dot(Z)

            ax = abs(nx); ay = abs(ny); az = abs(nz)
            if ay >= ax and ay >= az:
                group = "FRONT" if ny < 0.0 else "BACK"
            elif ax >= ay and ax >= az:
                group = "RIGHT" if nx > 0.0 else "LEFT"
            else:
                group = "UP" if nz > 0.0 else "DOWN"

            xr, zu = proj_axes[group]

            # Project each loop vertex into the group's virtual screen plane
            ls = face.loop_start
            le = ls + face.loop_total
            for li in range(ls, le):
                vi = mesh.loops[li].vertex_index
                wco = mw @ mesh.vertices[vi].co
                u = wco.dot(xr)
                v = wco.dot(zu)
                records.append((mesh, li, group, u, v))

                b = bounds[group]
                if u < b[0]: b[0] = u
                if u > b[1]: b[1] = u
                if v < b[2]: b[2] = v
                if v > b[3]: b[3] = v

    if not records:
        return

    # Compute group sizes
    sizes = {}
    max_dim = 0.0
    for g, (umin, umax, vmin, vmax) in bounds.items():
        if not math.isfinite(umin):
            sizes[g] = (0.0, 0.0)
            continue
        w = max(umax - umin, 1e-9)
        h = max(vmax - vmin, 1e-9)
        sizes[g] = (w, h)
        max_dim = max(max_dim, w, h)

    # Cell size and padding in raw units
    cell = max(max_dim, 1e-6)
    pad = cell * 0.05

    # Layout extents in raw units
    total_w = 4 * cell + 5 * pad
    total_h = 3 * cell + 4 * pad

    # Determine scaling / placement strategy
    mode_eff = mode.upper().strip() if isinstance(mode, str) else "AUTO"
    if mode_eff == "AUTO":
        mode_eff = "UNIT" if scale_to_unit else "RAW"

    img_size = None
    if mode_eff in {"PIXEL", "BOUNDS"}:
        img_size = _get_active_image_size(context)
        if img_size is None:
            # Fallback: still produce something usable without an image
            mode_eff = "UNIT"

    scale_u = 1.0
    scale_v = 1.0
    off_global_u = 0.0
    off_global_v = 0.0

    if mode_eff == "UNIT":
        s = 1.0 / max(total_w, total_h)
        # center the net in [0,1]
        off_global_u = (1.0 - total_w * s) * 0.5
        off_global_v = (1.0 - total_h * s) * 0.5
        scale_u = s
        scale_v = s

    elif mode_eff == "BOUNDS":
        if img_size:
            img_w, img_h = img_size
            margin_u = float(margin_px) / float(max(img_w, 1))
            margin_v = float(margin_px) / float(max(img_h, 1))
        else:
            margin_u = margin_v = 0.02

        avail_w = max(1.0 - 2.0 * margin_u, 1e-6)
        avail_h = max(1.0 - 2.0 * margin_v, 1e-6)
        s = min(avail_w / max(total_w, 1e-9), avail_h / max(total_h, 1e-9))
        off_global_u = margin_u + (avail_w - total_w * s) * 0.5
        off_global_v = margin_v + (avail_h - total_h * s) * 0.5
        scale_u = s
        scale_v = s

    elif mode_eff == "PIXEL":
        img_w, img_h = img_size
        scale_u = 1.0 / float(max(img_w, 1))
        scale_v = 1.0 / float(max(img_h, 1))
        margin_u = float(margin_px) / float(max(img_w, 1))
        margin_v = float(margin_px) / float(max(img_h, 1))

        # Place near bottom-left by default.
        off_global_u = margin_u
        off_global_v = margin_v

        # If it wouldn't fit in the 0..1 tile, shrink uniformly (preserving proportions).
        net_w = total_w * scale_u
        net_h = total_h * scale_v
        avail_w = max(1.0 - 2.0 * margin_u, 1e-6)
        avail_h = max(1.0 - 2.0 * margin_v, 1e-6)
        if net_w > avail_w or net_h > avail_h:
            k = min(avail_w / max(net_w, 1e-9), avail_h / max(net_h, 1e-9))
            scale_u *= k
            scale_v *= k

    else:  # RAW
        scale_u = 1.0
        scale_v = 1.0
        off_global_u = 0.0
        off_global_v = 0.0

    # Per-group offsets (raw units), with per-group centering in its cell
    group_offset = {}
    for g, (col, row) in net_pos.items():
        w, h = sizes.get(g, (0.0, 0.0))
        # If group is empty, just place at cell origin
        cx = (cell - w) * 0.5
        cy = (cell - h) * 0.5
        u0 = pad + col * (cell + pad) + max(cx, 0.0)
        v0 = pad + row * (cell + pad) + max(cy, 0.0)
        group_offset[g] = (u0, v0)

    # Apply UVs
    for mesh, li, g, u, v in records:
        umin, _umax, vmin, _vmax = bounds[g]
        u_local = (u - umin) + group_offset[g][0]
        v_local = (v - vmin) + group_offset[g][1]
        uv = mesh.uv_layers.active.data[li].uv
        uv[0] = off_global_u + u_local * scale_u
        uv[1] = off_global_v + v_local * scale_v

    # Update meshes
    for obj in objects:
        if obj and obj.type == "MESH":
            obj.data.update()


class OpUVCuboidUnwrap(bpy.types.Operator):
    """Specialized VS cuboid UV unwrap"""

    """
    This unwraps cuboid UVs into format (in Blender axes):
             __________ __________
            |    UP    |  DOWN    |
            |   (+z)   |  (-z)    |
     _______|__________|__________|_______
    | LEFT  |  FRONT   | RIGHT |  BACK    |
    | (-x)  |  (-y)    | (+x)  |  (+y)    |
    |_______|__________|_______|__________|

    The notation is the FRONT, BACK, LEFT, RIGHT, UP, DOWN faces.
    Example using "-y" as the FRONT axis is shown above.

    The front-facing side of the model, is generally most important, so
    we want unwrap format to create a strip across the sides of the cuboid
    as these are usually the most viewed location, so we want artist to most
    easily blend together the texture across these sides.
    
    We will always operate on arrays of face sides stored in format:
        faces = [
            LEFT,
            FRONT,
            RIGHT,
            BACK,
            UP,
            DOWN,
        ]
    This is so at the last step
    """
    bl_idname = "vintagestory.uv_cuboid_unwrap"
    bl_label = "Cuboid UV Unwrap (VS)"
    bl_options = {"REGISTER", "UNDO"}

    front_face: bpy.props.EnumProperty(
        items=[ # (identifier, name, description)
            ("-y", "-Y", "Front face is -Y"),
            ("+y", "+Y", "Front face is +Y"),
            ("-x", "-X", "Front face is -X"),
            ("+x", "+X", "Front face is +X"),
            ("+z,-x", "+Z (-X Up)", "Front face is +Z (-X Up)"),
            ("-z,+x", "-Z (+X Up)", "Front face is -Z (+X Up)"),
            ("+z,+y", "+Z (+Y Up)", "Front face is +Z (+Y Up)"),
            ("-z,-y", "-Z (-Y Up)", "Front face is -Z (-Y Up)"),
        ],
        default="-y",
        name="Front Face",
        description="Front face of cuboid for unwraping",
    )

    use_local_normals: bpy.props.BoolProperty(
        default=False,
        name="Use Local Space Normals",
        description="Use local space normals instead of world space normals",
    )

    scale_to_unit: bpy.props.BoolProperty(
        default=False,
        name="Scale to [0, 1]",
        description="Scale UVs to fit into [0, 1] square",
    )

    def execute(self, context):
        args = self.as_keywords()

        # unpack args

        # use local normals to find front face instead of world normals
        # BUG TODO: does not handle face rotations properly, e.g. up/down faces
        # are rotated the wrong way, need to adjust their rotations
        use_local_space_normals = args.get("use_local_normals", False)

        # scale uv to unit square [0, 1]
        scale_to_unit = args.get("scale_to_unit", False)

        # map `front_face` string arg to integer axis index
        front_face = args.get("front_face", "-y")
        if front_face == "+x":
            front_axis = X_AXIS
            mat_front_face_to_xy = MAT_X_AX_TO_XY
            mat_back_face_to_xy = MAT_X_NEG_AX_TO_XY
            mat_left_face_to_xy = MAT_Y_NEG_AX_TO_XY
            mat_right_face_to_xy = MAT_Y_AX_TO_XY
            mat_up_face_to_xy = MAT_Z_AX_ROT270_TO_XY
            mat_down_face_to_xy = MAT_Z_NEG_AX_ROT90_TO_XY
        elif front_face == "-x":
            front_axis = X_NEG_AXIS
            mat_front_face_to_xy = MAT_X_NEG_AX_TO_XY
            mat_back_face_to_xy = MAT_X_AX_TO_XY
            mat_left_face_to_xy = MAT_Y_AX_TO_XY
            mat_right_face_to_xy = MAT_Y_NEG_AX_TO_XY
            mat_up_face_to_xy = MAT_Z_AX_ROT90_TO_XY
            mat_down_face_to_xy = MAT_Z_NEG_AX_ROT270_TO_XY
        elif front_face == "+y":
            front_axis = Y_AXIS
            mat_front_face_to_xy = MAT_Y_AX_TO_XY
            mat_back_face_to_xy = MAT_Y_NEG_AX_TO_XY
            mat_left_face_to_xy = MAT_X_AX_TO_XY
            mat_right_face_to_xy = MAT_X_NEG_AX_TO_XY
            mat_up_face_to_xy = MAT_Z_AX_ROT180_TO_XY
            mat_down_face_to_xy = MAT_Z_NEG_AX_ROT180_TO_XY
        elif front_face == "-y":
            front_axis = Y_NEG_AXIS
            mat_front_face_to_xy = MAT_Y_NEG_AX_TO_XY
            mat_back_face_to_xy = MAT_Y_AX_TO_XY
            mat_left_face_to_xy = MAT_X_NEG_AX_TO_XY
            mat_right_face_to_xy = MAT_X_AX_TO_XY
            mat_up_face_to_xy = MAT_Z_AX_TO_XY
            mat_down_face_to_xy = MAT_Z_NEG_AX_TO_XY
        # z-axis are different: depends on if we want aligned to y- or x-axis,
        # matrices here are hard-coded since the rotations do not follow an
        # easy pattern for re-use
        elif front_face == "+z,-x":
            front_axis = Z_AXIS
            mat_front_face_to_xy = MAT_Z_AX_ROT270_TO_XY
            mat_back_face_to_xy = MAT_Z_NEG_AX_ROT270_TO_XY
            mat_left_face_to_xy = np.array([
                [0.0, 0.0, 1.0], # x <- z
                [-1.0, 0.0, 0.0], # y <- -x
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_right_face_to_xy = np.array([
                [0.0, 0.0, -1.0], # x <- -z
                [-1.0, 0.0, 0.0], # y <- -x
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_up_face_to_xy = np.array([
                [0.0, 1.0, 0.0], # x <- y
                [0.0, 0.0, -1.0], # y <- -z
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_down_face_to_xy = np.array([
                [0.0, 1.0, 0.0], # x <- y
                [0.0, 0.0, 1.0], # y <- z
                [0.0, 0.0, 0.0], # z <- 0
            ])
        elif front_face == "-z,+x":
            front_axis = Z_NEG_AXIS
            mat_front_face_to_xy = MAT_Z_NEG_AX_ROT90_TO_XY
            mat_back_face_to_xy = MAT_Z_AX_ROT90_TO_XY
            mat_left_face_to_xy = np.array([
                [0.0, 0.0, -1.0], # x <- -z
                [1.0, 0.0, 0.0], # y <- x
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_right_face_to_xy = np.array([
                [0.0, 0.0, 1.0], # x <- z
                [1.0, 0.0, 0.0], # y <- x
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_up_face_to_xy = np.array([
                [0.0, 1.0, 0.0], # x <- y
                [0.0, 0.0, 1.0], # y <- z
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_down_face_to_xy = np.array([
                [0.0, 1.0, 0.0], # x <- y
                [0.0, 0.0, -1.0], # y <- -z
                [0.0, 0.0, 0.0], # z <- 0
            ])
        elif front_face == "+z,+y":
            front_axis = Z_AXIS
            mat_front_face_to_xy = MAT_Z_AX_TO_XY
            mat_back_face_to_xy = MAT_Z_NEG_AX_ROT180_TO_XY
            mat_left_face_to_xy = np.array([
                [0.0, 0.0, 1.0], # x <- z
                [0.0, 1.0, 0.0], # y <- y
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_right_face_to_xy = np.array([
                [0.0, 0.0, -1.0], # x <- -z
                [0.0, 1.0, 0.0], # y <- y
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_up_face_to_xy = np.array([
                [1.0, 0.0, 0.0], # x <- x
                [0.0, 0.0, -1.0], # y <- -z
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_down_face_to_xy = np.array([
                [1.0, 0.0, 0.0], # x <- x
                [0.0, 0.0, 1.0], # y <- z
                [0.0, 0.0, 0.0], # z <- 0
            ])
        elif front_face == "-z,-y":
            front_axis = Z_NEG_AXIS
            mat_front_face_to_xy = np.array([ #  -Z axis
                [1.0, 0.0, 0.0], # x <- x
                [0.0, -1.0, 0.0], # y <- -y
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_back_face_to_xy = np.array([ #  +Z axis
                [1.0, 0.0, 0.0], # x <- x
                [0.0, 1.0, 0.0], # y <- y
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_left_face_to_xy = np.array([
                [0.0, 0.0, -1.0], # x <- -z
                [0.0, -1.0, 0.0], # y <- -y
                [0.0, 0.0, 0.0],  # z <- 0
            ])
            mat_right_face_to_xy = np.array([
                [0.0, 0.0, 1.0],  # x <- z
                [0.0, -1.0, 0.0], # y <- -y
                [0.0, 0.0, 0.0],  # z <- 0
            ])
            mat_up_face_to_xy = np.array([
                [1.0, 0.0, 0.0], # x <- x
                [0.0, 0.0, 1.0], # y <- z
                [0.0, 0.0, 0.0], # z <- 0
            ])
            mat_down_face_to_xy = np.array([
                [1.0, 0.0, 0.0],  # x <- x
                [0.0, 0.0, -1.0], # y <- -z
                [0.0, 0.0, 0.0],  # z <- 0
            ])
        else:
            err_msg = f"Invalid front_face: {front_face}, must be one of: +x, -x, +y, -y, +z, -z"
            self.report({"ERROR"}, err_msg)
            raise Exception(err_msg)

        # uv face format indices
        IDX_UV_FACE_LEFT = 0
        IDX_UV_FACE_FRONT = 1
        IDX_UV_FACE_RIGHT = 2
        IDX_UV_FACE_BACK = 3
        IDX_UV_FACE_UP = 4
        IDX_UV_FACE_DOWN = 5

        # need to be in object mode to access context selected objects
        user_mode = context.active_object.mode
        if user_mode != "OBJECT":
            need_to_switch_mode_back = True
            bpy.ops.object.mode_set(mode="OBJECT")
        else:
            need_to_switch_mode_back = False
        
        # only perform on selected objects
        objects = bpy.context.selected_objects
        for obj in objects:
            try:
                mesh = obj.data
                if not isinstance(mesh, bpy.types.Mesh):
                    continue
                
                # skip non cuboid meshes, print warning
                if len(mesh.vertices) != 8:
                    log.warning(f"Skipping UV unwrap of non-cuboid mesh: {obj.name}")
                    continue

                uv_layer = mesh.uv_layers.active.data
                vertices_local = np.ones((4, 8)) # 8 vertices, each as (x,y,z,1) for 4x4 matrix multiplication
                for i, v in enumerate(mesh.vertices):
                    vertices_local[0:3,i] = v.co
                
                # transform vertices to world space
                matrix_world = np.asarray(obj.matrix_world)
                vertices = matrix_world @ vertices_local

                if use_local_space_normals: # just use identity
                    normal_matrix = np.identity(3)
                else:
                    # normal matrix = tranpose of inverse of upper left 3x3 of world matrix
                    try:
                        normal_matrix = np.transpose(np.linalg.inv(matrix_world[0:3,0:3]))
                    except:
                        log.warning(f"Non-invertible matrix for: {obj.name}, using its world matrix instead")
                        normal_matrix = matrix_world
                
                # gather original mesh face vertices and normals
                mesh_face_uv_loop_start = np.zeros((6,), dtype=int)   # (face,)
                mesh_face_vert_indices = np.zeros((6, 4), dtype=int)  # (face, vert)
                mesh_face_vertices = np.zeros((6, 4, 4), dtype=np.float64) # (face, vert, xyzw)
                mesh_face_normals = np.zeros((6, 3), dtype=np.float64)     # (face, xyz)

                for i, face in enumerate(mesh.polygons):
                    mesh_face_uv_loop_start[i] = face.loop_start

                    # note face vertices contains indices pointers into mesh vertices
                    mesh_face_vertices[i,:,:] = np.stack(
                        [ vertices[:,v] for v in face.vertices ],
                        axis=0,
                    )
                    mesh_face_vert_indices[i,:] = face.vertices
                    mesh_face_normals[i,:] = np.array(face.normal)
                    
                # world space face normals
                mesh_face_normals_world = normal_matrix @ mesh_face_normals.transpose()
                mesh_face_normals_world = mesh_face_normals_world.transpose()

                # First find the "front face" by finding the face normal that
                # is closest matching to the front face normal
                
                # determine index of closest matching mesh front face: 
                # detect world space face normal closest to
                # axis-aligned front face normal
                front_index = np.argmax(np.sum(mesh_face_normals_world * front_axis, axis=1), axis=0)

                # Next, determine which faces are adjacent to the front face
                # transform the front face to XY plane and define:
                #
                #          |    up    |
                #     ____v3__________v2_____          ^ +y
                #          |          |                |
                #    left  |   front  | right          +---> +x
                #          |          |
                #    _____v0__________v1_____
                #          |          |
                #          |   down   |
                # 
                # Then check all faces and assign other sides based on
                # which faces contain the same vertices as the front face:
                # - left: v0, v3
                # - right: v1, v2
                # - up: v2, v3
                # - down: v0, v1
                # - back: (none)
                # This method ensures each face is assigned only once
                # (for properly defined cuboids).
                # 
                # NOTATION:
                # - f0, f1, f2, f3 are face local indices in values {0, 1, 2, 3}
                # - v0, v1, v2, v3 are corresponding global mesh vertex indices
                #   these are pointers into the mesh.vertices
                
                # map face vertices indices to standard v0, v1, v2, v3 format
                # 1. transform front face coords into an XY plane
                #    (use face normal specific transform)
                # 2. find v0 (bottom left) as "min" vertex
                # 3. determine if loop is clockwise or counterclockwise
                # 4. assign v1, v2, v3 based on loop order
                mesh_face_vertices_xy = mat_front_face_to_xy @ mesh_face_vertices[front_index,:,:3].transpose()
                mesh_face_vertices_xy = mesh_face_vertices_xy.transpose()
                u0 = mesh_face_vertices_xy[0,:3]
                u1 = mesh_face_vertices_xy[1,:3]
                u2 = mesh_face_vertices_xy[2,:3]
                idx_v0 = index_of_vmin(mesh_face_vertices_xy)
                is_cw = loop_is_clockwise([u0, u1, u2])

                # print(f"Mesh front face {front_index}: {mesh_face_vertices[front_index,:,:]}")
                # print(f"Mesh front face XY: {mesh_face_vertices_xy}")
                # print(f"idx_v0 = {idx_v0}")
                # print(f"is_cw = {is_cw}")

                if is_cw:
                    idx_v1 = (idx_v0 + 3) % 4
                    idx_v2 = (idx_v0 + 2) % 4
                    idx_v3 = (idx_v0 + 1) % 4
                else:
                    idx_v1 = (idx_v0 + 1) % 4
                    idx_v2 = (idx_v0 + 2) % 4
                    idx_v3 = (idx_v0 + 3) % 4

                # get mesh global vertex indices for front face
                front_v0 = mesh_face_vert_indices[front_index, idx_v0]
                front_v1 = mesh_face_vert_indices[front_index, idx_v1]
                front_v2 = mesh_face_vert_indices[front_index, idx_v2]
                front_v3 = mesh_face_vert_indices[front_index, idx_v3]

                # print(f"FRONT: idx_v0 = {idx_v0}, idx_v1 = {idx_v1}, idx_v2 = {idx_v2}, idx_v3 = {idx_v3}")
                # print(f"FRONT: v0 = {front_v0}, v1 = {front_v1}, v2 = {front_v2}, v3 = {front_v3}")
                
                mesh_face_f0_f1_f2_f3 = np.zeros((6, 4), dtype=int)
                mesh_face_v0_v1_v2_v3 = np.zeros((6, 4), dtype=int)
                mesh_face_f0_f1_f2_f3[front_index,:] = [idx_v0, idx_v1, idx_v2, idx_v3]
                mesh_face_v0_v1_v2_v3[front_index,:] = [front_v0, front_v1, front_v2, front_v3]

                # detect face directions relative to the front face.
                # do this by matching shared points between faces
                mesh_face_directions = np.full((6,), -1, dtype=int)
                mesh_face_directions[front_index] = IDX_UV_FACE_FRONT
                for i in range(0, 6): # for face in mesh.polygons
                    if i == front_index:
                        continue
                    
                    face_vert_indices = mesh_face_vert_indices[i,:]

                    if front_v0 in face_vert_indices and front_v3 in face_vert_indices:
                        mesh_face_directions[i] = IDX_UV_FACE_LEFT
                        # left f2 is index of front f3
                        f2 = np.argwhere(face_vert_indices == front_v3)[0][0]
                        f0 = (f2 + 2) % 4
                        face_verts_xy = mat_left_face_to_xy @ mesh_face_vertices[i,:,:3].transpose()
                    elif front_v1 in face_vert_indices and front_v2 in face_vert_indices:
                        mesh_face_directions[i] = IDX_UV_FACE_RIGHT
                        # right f0 is index of front f1
                        f0 = np.argwhere(face_vert_indices == front_v1)[0][0]
                        face_verts_xy = mat_right_face_to_xy @ mesh_face_vertices[i,:,:3].transpose()
                    elif front_v2 in face_vert_indices and front_v3 in face_vert_indices:
                        mesh_face_directions[i] = IDX_UV_FACE_UP
                        # up f0 is index of front f3
                        f0 = np.argwhere(face_vert_indices == front_v3)[0][0]
                        face_verts_xy = mat_up_face_to_xy @ mesh_face_vertices[i,:,:3].transpose()
                    elif front_v0 in face_vert_indices and front_v1 in face_vert_indices:
                        mesh_face_directions[i] = IDX_UV_FACE_DOWN
                        # down f2 is index of front f1 
                        f2 = np.argwhere(face_vert_indices == front_v1)[0][0]
                        f0 = (f2 + 2) % 4
                        face_verts_xy = mat_down_face_to_xy @ mesh_face_vertices[i,:,:3].transpose()
                    else:
                        mesh_face_directions[i] = IDX_UV_FACE_BACK
                        face_verts_xy = mat_back_face_to_xy @ mesh_face_vertices[i,:,:3].transpose()
                        f0 = index_of_vmin(face_verts_xy.transpose())
                    
                    is_cw = loop_is_clockwise(face_verts_xy[0:3,0:3].transpose())
                    if is_cw:
                        f3 = (f0 + 1) % 4
                        f2 = (f0 + 2) % 4
                        f1 = (f0 + 3) % 4
                    else:
                        f1 = (f0 + 1) % 4
                        f2 = (f0 + 2) % 4
                        f3 = (f0 + 3) % 4
                    # maps face vertices in f in {0,1,2,3}
                    # to v in global mesh vertices indices
                    v0 = mesh_face_vert_indices[i, f0]
                    v1 = mesh_face_vert_indices[i, f1]
                    v2 = mesh_face_vert_indices[i, f2]
                    v3 = mesh_face_vert_indices[i, f3]
                    mesh_face_f0_f1_f2_f3[i,:] = [f0, f1, f2, f3] # local mesh loop indices {0, 1, 2, 3}
                    mesh_face_v0_v1_v2_v3[i,:] = [v0, v1, v2, v3] # pointers to mesh.vertices

                if -1 in mesh_face_directions:
                    raise Exception("Invalid cuboid mesh, some faces are not properly defined, could not determine face directions")

                # we have assigned all face directions and mapped all
                # face vertices to their v0, v1, v2, v3 face uv order.
                # now, determine face width (v0 -> v1) and height (v1 -> v2)

                # 6 faces, each as 
                face_uv_width_height = np.zeros((6, 2))        # (face, width/height)
                face_uv_xy = np.zeros((6, 4, 2))               # (face, vert, xy)
                face_uv_loop_start = np.zeros((6,), dtype=int) # (face,)

                # creates uvs based on face width/height and 
                # maps faces index to uv index in LEFT-FRONT-RIGHT-BACK-UP-DOWN
                for i in range(0, 6):
                    direction_index = mesh_face_directions[i]
                    f0, f1, f2, f3 = mesh_face_f0_f1_f2_f3[i,:]
                    v0, v1, v2, v3 = mesh_face_v0_v1_v2_v3[i,:]

                    face_width = np.linalg.norm(vertices[:3,v0] - vertices[:3,v1])
                    face_height = np.linalg.norm(vertices[:3,v1] - vertices[:3,v2])
                    
                    face_uv_width_height[direction_index,0] = face_width
                    face_uv_width_height[direction_index,1] = face_height

                    face_uv_xy[direction_index,f0,:] = (0.0, 0.0)
                    face_uv_xy[direction_index,f1,:] = (face_width, 0.0)
                    face_uv_xy[direction_index,f2,:] = (face_width, face_height)
                    face_uv_xy[direction_index,f3,:] = (0.0, face_height)

                    face_uv_loop_start[direction_index] = mesh_face_uv_loop_start[i]
                
                # original uv_xy are local sizes of each face
                # first translate each face to its unwrapped position
                uv_offset = np.zeros((6, 2))
                # LEFT: x=0, y=0, no change
                # FRONT
                uv_offset[1,:] = (face_uv_width_height[0,0], 0.0)
                # RIGHT
                uv_offset[2,:] = (face_uv_width_height[0,0] + face_uv_width_height[1,0], 0.0)
                # BACK
                uv_offset[3,:] = (face_uv_width_height[0,0] + face_uv_width_height[1,0] + face_uv_width_height[2,0], 0.0)
                # UP
                uv_offset[4,:] = (face_uv_width_height[0,0], face_uv_width_height[0,1])
                # DOWN, x = x_left + x_up
                uv_offset[5,:] = (face_uv_width_height[0,0] + face_uv_width_height[4,0], face_uv_width_height[0,1])
                
                # shape broadcasting:
                # (6,4,2)   =     (6,1,2)         +   (6,4,2)
                uv_xy = uv_offset[:,np.newaxis,:] + face_uv_xy
                
                # finally, scale the entire uv map to fit into the (0, 1) square
                if scale_to_unit:
                    uv_scale = 1.0 / uv_xy.max(axis=(0, 1, 2))
                    uv_xy_normalized = uv_scale * uv_xy

                    uv_x = uv_xy_normalized[:,:,0]
                    uv_y = uv_xy_normalized[:,:,1]
                else:
                    uv_x = uv_xy[:,:,0]
                    uv_y = uv_xy[:,:,1]
                
                # update face uvs
                for i in range(0, 6):
                    idx = face_uv_loop_start[i]
                    uv_layer[idx].uv = (uv_x[i,0], uv_y[i,0])
                    uv_layer[idx+1].uv = (uv_x[i,1], uv_y[i,1])
                    uv_layer[idx+2].uv = (uv_x[i,2], uv_y[i,2])
                    uv_layer[idx+3].uv = (uv_x[i,3], uv_y[i,3])
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_msg = f"Error unwrapping cuboid: {obj.name}, {e}"
                log.error(err_msg)
                self.report({"ERROR"}, err_msg)
            
        if need_to_switch_mode_back:
            bpy.ops.object.mode_set(mode=user_mode)
        
        return {"FINISHED"}


class OpUVCuboidUnwrapProjectFromView(bpy.types.Operator):
    """VS cuboid unwrap that orients the UVs to the current 3D View.

    The face most "visible" in the viewport becomes the FRONT face of the
    unwrap layout. This makes it much easier to texture from dorsal,
    underbelly, or side views because the UV island orientation matches what
    you're seeing.

    In edit mode, only cuboids whose *entire* 6 faces are selected are unwrapped.
    """

    bl_idname = "vintagestory.uv_cuboid_unwrap_project_from_view"
    bl_label = "Cuboid UV Unwrap (Project From View)"
    bl_options = {"REGISTER", "UNDO"}

    scale_to_unit: bpy.props.BoolProperty(
        default=False,
        name="Scale to [0, 1]",
        description="Scale UVs to fit into [0, 1] square",
    )

    def execute(self, context):
        vb = _view_basis_world(context)
        if vb is None:
            self.report({"ERROR"}, "Could not read active 3D View orientation (need a VIEW_3D area).")
            return {"CANCELLED"}

        _x_right, _y_view_dir, _z_up, world_to_basis_4x4 = vb

        # Determine target objects:
        # - Object mode: selected objects
        # - Edit mode: only cuboids with all 6 faces selected
        objects = _selected_cuboid_objects(context)
        if not objects:
            self.report({"WARNING"}, "No fully-selected cuboids found to unwrap.")
            return {"CANCELLED"}

        # Need to be in object mode to write uv_layer efficiently.
        user_mode = context.active_object.mode if context.active_object else "OBJECT"
        need_to_switch_mode_back = user_mode != "OBJECT"
        if need_to_switch_mode_back:
            bpy.ops.object.mode_set(mode="OBJECT")

        try:
            # We align the view basis so that "front" is always -Y in that basis.
            _unwrap_project_from_view_groups(
                context=context,
                objects=objects,
                scale_to_unit=bool(self.scale_to_unit),
                mode="PIXEL",
                margin_px=int(getattr(self, "margin_px", 2)),
            )
        finally:
            if need_to_switch_mode_back:
                bpy.ops.object.mode_set(mode=user_mode)

        return {"FINISHED"}



class OpUVCuboidUnwrapProjectFromViewToBounds(bpy.types.Operator):
    """View-oriented cuboid unwrap that fits the net into the 0..1 UV tile.

    Same face grouping/orientation as 'Cuboid UV Unwrap (View)', but scaled and
    centered to stay inside UV bounds (with a small margin).
    """

    bl_idname = "vintagestory.uv_cuboid_unwrap_project_from_view_to_bounds"
    bl_label = "Cuboid UV Unwrap (View to Bounds)"
    bl_options = {"REGISTER", "UNDO"}

    margin_px: bpy.props.IntProperty(
        default=2,
        min=0,
        name="Margin (px)",
        description="Margin in pixels to keep inside the 0..1 UV tile when an image is available",
    )

    def execute(self, context):
        vb = _view_basis_world(context)
        if vb is None:
            self.report({"ERROR"}, "Could not read active 3D View orientation (need a VIEW_3D area).")
            return {"CANCELLED"}

        objects = _selected_cuboid_objects(context)
        if not objects:
            self.report({"WARNING"}, "No fully-selected cuboids found to unwrap.")
            return {"CANCELLED"}

        user_mode = context.active_object.mode if context.active_object else "OBJECT"
        need_to_switch_mode_back = user_mode != "OBJECT"
        if need_to_switch_mode_back:
            bpy.ops.object.mode_set(mode="OBJECT")

        try:
            _unwrap_project_from_view_groups(
                context=context,
                objects=objects,
                scale_to_unit=True,
                mode="BOUNDS",
                margin_px=int(self.margin_px),
            )
        finally:
            if need_to_switch_mode_back:
                bpy.ops.object.mode_set(mode=user_mode)

        return {"FINISHED"}




class OpUVCuboidMakeCuboidUV(bpy.types.Operator):
    """Rectify existing UVs on selected cuboids into proper per-face rectangles.

    This is meant as a "re-cuboid" step before export to tools that expect
    cuboid-style UV rectangles (e.g. VSMC).

    What it does:
    - Ensures the mesh has an active UV map.
    - For each of the 6 quad faces, computes the current UV bounding box.
    - Reassigns the 4 UV corners to match a canonical, axis-aligned rectangle,
      *rotating* the UV tile as needed so the face orientation is consistent.

    It preserves the existing UV rectangle *position/size* (the bbox) so your
    view-based grouping layout stays intact.

    In edit mode, only cuboids whose entire 6 faces are selected are processed.
    """

    bl_idname = "vintagestory.uv_make_cuboid_uv"
    bl_label = "Make Cuboid UV (Rectify)"
    bl_options = {"REGISTER", "UNDO"}

    snap_to_pixels: bpy.props.BoolProperty(
        default=False,
        name="Snap to Pixels",
        description="Snap UV rectangle bounds to the active image pixel grid (if an image is available)",
    )

    def execute(self, context):
        objects = _selected_cuboid_objects(context)
        if not objects:
            self.report({"WARNING"}, "No fully-selected cuboids found.")
            return {"CANCELLED"}

        user_mode = context.active_object.mode if context.active_object else "OBJECT"
        need_to_switch_mode_back = user_mode != "OBJECT"
        if need_to_switch_mode_back:
            bpy.ops.object.mode_set(mode="OBJECT")

        try:
            img_size = _get_active_image_size(context)
            for obj in objects:
                mesh = obj.data
                if not isinstance(mesh, bpy.types.Mesh):
                    continue
                if len(mesh.vertices) != 8 or len(mesh.polygons) != 6:
                    continue

                # Ensure UV map exists and has data.
                if mesh.uv_layers.active is None:
                    mesh.uv_layers.new(name="UVMap")
                # In some Blender 4.x cases, the wrapper exists but data can be empty until update.
                try:
                    mesh.update()
                except Exception:
                    pass
                if mesh.uv_layers.active is None:
                    continue

                uv_layer = mesh.uv_layers.active.data
                if len(uv_layer) == 0 and len(mesh.loops) > 0:
                    # Create a fresh UV map.
                    mesh.uv_layers.new(name="UVMap")
                    mesh.uv_layers.active = mesh.uv_layers[-1]
                    try:
                        mesh.update()
                    except Exception:
                        pass
                    uv_layer = mesh.uv_layers.active.data

                # Fallback: if still empty, we cannot rectify.
                if len(uv_layer) == 0:
                    self.report({"WARNING"}, f"{obj.name}: UV map has no data; run an unwrap first.")
                    continue

                # Build quick local vertex lookup
                v_local = [np.array(v.co, dtype=np.float64) for v in mesh.vertices]

                for face in mesh.polygons:
                    if face.loop_total != 4:
                        continue

                    ls = face.loop_start
                    uvs = [uv_layer[ls + i].uv.copy() for i in range(4)]
                    uv_min_x = min(u[0] for u in uvs)
                    uv_max_x = max(u[0] for u in uvs)
                    uv_min_y = min(u[1] for u in uvs)
                    uv_max_y = max(u[1] for u in uvs)

                    # Degenerate UVs: skip.
                    if (uv_max_x - uv_min_x) <= 1e-12 or (uv_max_y - uv_min_y) <= 1e-12:
                        continue

                    # Optional snap to pixels
                    if self.snap_to_pixels and img_size is not None:
                        w, h = img_size
                        def _snap(v, n):
                            return round(v * n) / float(n)
                        uv_min_x = _snap(uv_min_x, w)
                        uv_max_x = _snap(uv_max_x, w)
                        uv_min_y = _snap(uv_min_y, h)
                        uv_max_y = _snap(uv_max_y, h)
                        if uv_min_x == uv_max_x or uv_min_y == uv_max_y:
                            continue

                    # Project this face's vertices to a stable 2D space based on its normal,
                    # matching the exporter's convention. This gives us a consistent "corner order".
                    n = np.array(face.normal, dtype=np.float64)
                    # Determine dominant axis
                    ax = int(np.argmax(np.abs(n)))
                    sign = 1.0 if n[ax] >= 0.0 else -1.0

                    verts = [v_local[idx] for idx in face.vertices]
                    verts2d = []
                    if ax == 0:  # +/-X
                        # +X: (y, z), -X: (-y, z)
                        for v in verts:
                            if sign > 0:
                                verts2d.append((v[1], v[2]))
                            else:
                                verts2d.append((-v[1], v[2]))
                    elif ax == 1:  # +/-Y
                        # +Y: (-x, z), -Y: (x, z)
                        for v in verts:
                            if sign > 0:
                                verts2d.append((-v[0], v[2]))
                            else:
                                verts2d.append((v[0], v[2]))
                    else:  # +/-Z
                        # +Z: (y, -x), -Z: (y, x)
                        for v in verts:
                            if sign > 0:
                                verts2d.append((v[1], -v[0]))
                            else:
                                verts2d.append((v[1], v[0]))

                    verts2d_np = np.array(verts2d, dtype=np.float64)
                    f0 = index_of_vmin(verts2d_np)
                    is_cw = loop_is_clockwise(verts2d_np)
                    if is_cw:
                        f3 = (f0 + 1) % 4
                        f2 = (f0 + 2) % 4
                        f1 = (f0 + 3) % 4
                    else:
                        f1 = (f0 + 1) % 4
                        f2 = (f0 + 2) % 4
                        f3 = (f0 + 3) % 4

                    # Assign bbox corners in a canonical order.
                    uv_for_corner = {
                        f0: (uv_min_x, uv_min_y),
                        f1: (uv_max_x, uv_min_y),
                        f2: (uv_max_x, uv_max_y),
                        f3: (uv_min_x, uv_max_y),
                    }

                    for j in range(4):
                        uv_layer[ls + j].uv = uv_for_corner[j]

        finally:
            if need_to_switch_mode_back:
                bpy.ops.object.mode_set(mode=user_mode)

        return {"FINISHED"}


class OpUVPixelUnwrap(bpy.types.Operator):
    """Unwrap all UVs into a single pixel (for single color textures)"""
    bl_idname = "vintagestory.uv_pixel_unwrap"
    bl_label = "Pixel UV Unwrap (VS)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # args = self.as_keywords()

        # need to be in object mode to access context selected objects
        user_mode = context.active_object.mode
        if user_mode != "OBJECT":
            need_to_switch_mode_back = True
            bpy.ops.object.mode_set(mode="OBJECT")
        else:
            need_to_switch_mode_back = False
        
        # only perform on selected objects
        objects = bpy.context.selected_objects
        for obj in objects:
            mesh = obj.data
            if not isinstance(mesh, bpy.types.Mesh):
                continue
            
            uv_layer = mesh.uv_layers.active.data

            for face in mesh.polygons:
                loop_start = face.loop_start
                # TODO: if object has texture, scale UVs to fit 1 pixel in texture.
                # otherwise, set to unit square (0,0) to (1,1)
                uv_layer[loop_start].uv = (0.0, 0.0)
                uv_layer[loop_start+1].uv = (1.0, 0.0)
                uv_layer[loop_start+2].uv = (1.0, 1.0)
                uv_layer[loop_start+3].uv = (0.0, 1.0)
        
        if need_to_switch_mode_back:
            bpy.ops.object.mode_set(mode=user_mode)
        
        return {"FINISHED"}


def round_cuboid_uv_island_to_pixels(
    uvs, # (faces, vertices, xy)
):
    """Round uv islands to closest pixel (integer) corners.
    Strategy assumes uv islands are already axis-aligned unwraps for
    cuboids, looking like

             __________ __________
            |    UP    |  DOWN    |
            |   (+z)   |  (-z)    |
     _______|__________|__________|_______
    | LEFT  |  FRONT   | RIGHT |  BACK    |
    | (-x)  |  (-y)    | (+x)  |  (+y)    |
    |_______|__________|_______|__________|

    We want to round face width/height so that faces that should have
    matching widths/heights (e.g. left/right or front/back) remain 
    consistent. Simple rounding does not guarantee this.

    1. Find (x_min, y_min) and offset uvs min to origin (0,0)
    
    2. First round x values on the x-axis:
        At high level, idea is to go from left-to-right "stretching" each
        face's x values to the closest x-integer. But each time we stretch
        a face, we need to also stretch everything to the right. In this way
        we are rounding face widths instead of just points.
         _______ ___________ ______
        |       |           |      |
        |       |           |      |
        |_______|___________|______|
        0----1----2----3----4----5----6----7----> x
                ^
                Stretch these points to x=2
                and update all faces to the right as well

         _______ ___________ ______
        |       | |         | |    | |
        |       |>|         |>|    |>|
        |_______|_|_________|_|____| |
        0----1----2----3----4----5----6----7----> x
                ^
         _________ ___________ ______
        |         |           |      |
        |         |           |      |
        |_________|___________|______|
        0----1----2----3----4----5----6----7----> x
                             ^       ^
                            These edges were also "stretched"
        
         _________ __________ _______
        |         |         | |    | |
        |         |         |<|    |<|
        |_________|_________|_|____|_|
        0----1----2----3----4----5----6----7----> x
                             ^ 
                            Now we want to stretch this next face to x=4
                            and stretch everything to the right back
         _________ _________ ______
        |         |         |      |
        |         |         |      |
        |_________|_________|______|
        0----1----2----3----4----5----6----7----> x

         _________ _________ ______
        |         |         |    | |
        |         |         |    |<|
        |_________|_________|____|_|
        0----1----2----3----4----5----6----7----> x

         _________ _________ ____
        |         |         |    |
        |         |         |    |
        |_________|_________|____|
        0----1----2----3----4----5----6----7----> x

        Issue: how to deal with faces coords not associated to same face:
            y2  __________________
               |                  |
            y1 |__________________|________
               |           |               |
            y0 |___________|_______________|
               ^           ^      ^       ^
            {y0,y1,y2}  {y0,y1}  {y1,y2} {y0,y1}
        
            Tag each vertex bin with rounded y values. This creates rules:
            - A bin's parent has y tags either subset or superset of
              current bin y-value tags
            - Child bins has y-tags either subset or superset of current
              bin y-value tags.
            This is a hacky way to estimate "parent-child" relationships
            without a full graph, based on the format of our cuboid uv islands.

        Exact method:
        1. Get all vertices with x values rounded to closest integer xround:
        2. Bin together vertex indices by their xround values, e.g. in island above
         points:        x       x       x    
                    x   x       x    x  x    x
                    x   x       x    x       x
                   -|---|-------|----|--|----|-----> x value (integers)
            bins:   0   4       12   17 20   25
           These bin integers are from rounding x value to closest x integer.
           We see bin 0 has 2 vertices, bin 4 has 3 vertices, etc. 
        3. Sort bins by xround value
        4. for each bin from 2nd bin, n=1, to end we find distance of each bin
           relative to "parent" bin (this is the "face width")
                parent bin = previous bin whose y value are subset of current bin
                             or are superset of current bin
                dx = x_bin(n) - x_bin(n-1)
                width_rounded = round(dx)
           Round all vertices in this bin and shift next bin vertices by dx:
                for point in x_bin(n):
                    point_x = x_bin(n-1) + width_rounded
                for point in bins n+1, n+2, ...:
                    if bin(n+1) y values are subset or superset of bin(n)
                        point_x += dx

    3. Second round y values on the y-axis:
       (Repeat same strategy above along y-axis)
       
    4. Re-translate back: find closest integer (x0, y0) to (x_min, y_min)
       and translate all uv vertices back by that distance.

    Returns new integer rounded uvs.
    """
    from collections import defaultdict
    
    num_faces = uvs.shape[0]
    num_vertices = uvs.shape[1]

    xmin_ymin = np.min(uvs, axis=(0, 1,))

    # translate uvs to origin
    uvs0 = uvs - xmin_ymin

    # rounded x, y to nearest integers (used for vertex binning below)
    uvs0_round_nearest = np.round(uvs0)

    def round_along_axis(ax, ax_ytags):
        """Round vertex values along an xy axis index ax in {0, 1}.
        Written in terms of x, but works for y as well.
        This modifies original uvs0_rounded array.
        """
        # bin vertices by rounded x values
        x_bins = defaultdict(list)
        bin_ytags = defaultdict(set)
        for f in range(num_faces):
            for v in range(num_vertices):
                x = int(uvs0_round_nearest[f, v, ax])
                x_bins[x].append((f, v, ax)) # uv vertex index
                bin_ytags[x].add(int(uvs0_round_nearest[f, v, ax_ytags]))
        
        x_sorted = sorted(x_bins)
        x_bins_sorted = [ x_bins[x] for x in x_sorted ]
        bin_ytags_sorted = [ bin_ytags[x] for x in x_sorted ]
        
        n_bins = len(x_bins_sorted)
        for i in range(1, n_bins):
            # find parent bin index
            p_parent = None
            for j in range(i-1, -1, -1):
                if bin_ytags_sorted[j].issubset(bin_ytags_sorted[i]) or \
                   bin_ytags_sorted[j].issuperset(bin_ytags_sorted[i]):
                    p_parent = x_bins_sorted[i-1][0]
                    break

            if p_parent is None: # could not find parent??
                print(f"`round_along_axis` could not find parent bin for bin {i}, defaulting to previous bin")
                p_parent = x_bins_sorted[i-1][0]

            # un-rounded dist from this to previous bin
            # assume all un-rounded x values in each bin are same
            p_curr = x_bins_sorted[i][0]
            x_parent = uvs0[p_parent]
            x_curr = uvs0[p_curr]
            dist = x_curr - x_parent

            # round the dist to closest int
            dist_rounded = np.round(dist)
            x_rounded = x_parent + dist_rounded
            
            # distance to offset to get to rounded values
            dx = x_rounded - x_curr

            # for all these current vertices, force values to val_rounded
            for p in x_bins_sorted[i]:
                uvs0[p] = x_rounded
            # for all further x vertices (to the right) offset by dx
            for j in range(i+1, n_bins):
                if bin_ytags_sorted[j].issubset(bin_ytags_sorted[i]) or \
                   bin_ytags_sorted[j].issuperset(bin_ytags_sorted[i]):
                    for p in x_bins_sorted[j]:
                        uvs0[p] += dx

    round_along_axis(ax=0, ax_ytags=1)
    round_along_axis(ax=1, ax_ytags=0)

    # translate back: round xmin, ymin to closest integer
    # also do final rounding to clean up any numerical arithmetic errors
    x0_y0 = np.round(xmin_ymin)
    uvs_rounded = np.round(uvs0 + x0_y0)
    return uvs_rounded
    

class OpUVPackSimpleBoundingBox(bpy.types.Operator):
    """Simple cuboid uv pack that treats all uv faces in a cuboid mesh as a
    connected island. Default Blender uv pack needs connected faces as an
    island, but the cuboid unwraps create disjointed faces.

    Using simple heuristic packing from Igarshi and Cosgrove 2001.
    https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/i3dg2001.pdf
    """
    bl_idname = "vintagestory.uv_pack_simple_bounding_box"
    bl_label = "Simple UV Pack"
    bl_options = {"REGISTER", "UNDO"}

    stand_up_islands: bpy.props.BoolProperty(
        default=False,
        name="Stand up UV islands",
        description="Rotates UV islands to stand up on the longest edge",
    )

    scale_to_unit: bpy.props.BoolProperty(
        default=True,
        name="Scale to [0, 1]",
        description="Scale packed UVs to fit into [0, 1] square",
    )

    round_to_pixels: bpy.props.BoolProperty(
        default=True,
        name="Round to pixels",
        description="Rounds UV vertices to closest pixel (based on texture size input)",
    )

    round_to_pixel_method: bpy.props.EnumProperty(
        items=[ # (identifier, name, description)
            ("closest", "Closest", "Round to closest pixel"),
            ("width_height", "Face Width/Height", "Round face width/heights (keeps consistent faces)"),
        ],
        default="width_height",
        name="Round to Pixel Method",
        description="Method to round UV vertices to closest pixel",
    )

    pad_pixels: bpy.props.IntProperty(
        default=1,
        name="Padding Pixels",
        description="Padding between UV islands in pixels relative to texture size",
        min=0,
    )

    padding_default_texture_size: bpy.props.IntProperty(
        default=128,
        name="Texture Size",
        description="Texture size to use for padding calculation if no texture is set",
        min=1,
    )

    len_margin: bpy.props.FloatProperty(
        default=1.2,
        name="UV Strip Length Margin",
        description="Adjusts how many UV islands fit into each row",
        min=0.0,
    )

    def execute(self, context):
        args = self.as_keywords()

        # unpack args
        stand_up_islands = args.get("stand_up_islands", False)
        scale_to_unit = args.get("scale_to_unit", True)
        round_to_pixels = args.get("round_to_pixels", True)
        round_to_pixel_method = args.get("round_to_pixel_method", "width_height")
        pad_pixels = args.get("pad_pixels", 1)
        tex_size = args.get("padding_default_texture_size", 256)
        len_margin = args.get("len_margin", 1.2)

        # need to be in object mode to access context selected objects
        user_mode = context.active_object.mode
        if user_mode != "OBJECT":
            need_to_switch_mode_back = True
            bpy.ops.object.mode_set(mode="OBJECT")
        else:
            need_to_switch_mode_back = False
        
        # only perform on selected objects
        all_objects = bpy.context.selected_objects
        
        # pre-filter objects to only include cuboid meshes
        cuboids = []
        for obj in all_objects:
            mesh = obj.data
            if not isinstance(mesh, bpy.types.Mesh):
                continue
            if len(mesh.polygons) != 6:
                continue
            cuboids.append(obj)
        
        num_cuboids = len(cuboids)
        if num_cuboids == 0: # skip if no cuboids
            return {"FINISHED"}

        # uvs for each object in shape: (num_objects, faces, vertices, xy)
        uvs = np.zeros((num_cuboids, 6, 4, 2), dtype=np.float64)
        # uv aabb for each object in shape: (num_objects, min/max, xy)
        uvs_aabb = np.zeros((num_cuboids, 2, 2), dtype=np.float64)

        for i, obj in enumerate(cuboids):
            mesh = obj.data
            if not isinstance(mesh, bpy.types.Mesh):
                continue
            
            uv_layer = mesh.uv_layers.active.data

            # gather uv faces and determine uv 2d aabb
            for f, face in enumerate(mesh.polygons):
                loop_start = face.loop_start
                uvs[i,f,0,:] = uv_layer[loop_start].uv
                uvs[i,f,1,:] = uv_layer[loop_start+1].uv
                uvs[i,f,2,:] = uv_layer[loop_start+2].uv
                uvs[i,f,3,:] = uv_layer[loop_start+3].uv

                # aabb
                uvs_aabb[i,0,:] = np.min(uvs[i,:,:,:], axis=(0, 1))
                uvs_aabb[i,1,:] = np.max(uvs[i,:,:,:], axis=(0, 1))

        # rotate each mesh uv islands by 90 deg to stand up on longest edge
        # (e.g. if width > height, TODO: option to stand up on width or height) 
        if stand_up_islands:
            # TODO
            pass

        # get widths dx and heights dy of uv island aabbs
        uv_width = uvs_aabb[:,1,0] - uvs_aabb[:,0,0]
        uv_height = uvs_aabb[:,1,1] - uvs_aabb[:,0,1]

        # print(f"uvs: {uvs}")
        # print(f"uvs_aabb: {uvs_aabb}")
        # print(f"uv_width: {uv_width}")
        # print(f"uv_height: {uv_height}")

        # uv areas
        uv_area = uv_width * uv_height

        # calculate total area of all uvs
        uv_total_area = np.sum(uv_area)
        # estimate square length needed to pack uvs as sqrt(total_area) * margin
        # where margin > 1.0 (TODO: make it a parameter)
        uv_total_area_sqrt = np.sqrt(uv_total_area)
        uv_pack_side_len = len_margin * uv_total_area_sqrt

        # sort islands by height (from tallest to shortest)
        indices_sorted = np.argsort(uv_height)[::-1]

        # iterate through height-sorted islands and split uvs into strips of
        # width up to uv_pack_side_len
        strips = []        # list of lists of uv indices
        strip_dy = []      # list of strip height offsets dy
        strip_heights = [] # list of strip height
        strip_widths = []  # list of strip width
        strip_island_width_offsets = [] # list of each strip's island offset (correspond to strip uv indices)

        curr_strip_dy = 0.0
        curr_strip_width = 0.0
        curr_strip_indices = []
        island_width_offsets = []
        for i in indices_sorted:
            island_width_offsets.append(curr_strip_width)
            curr_strip_width += uv_width[i]
            curr_strip_indices.append(i)

            if curr_strip_width >= uv_pack_side_len:
                # strip reached max width, start a new strip
                curr_height = uv_height[curr_strip_indices[0]] # first uv in strip is tallest
                strips.append(curr_strip_indices)
                strip_dy.append(curr_strip_dy)
                strip_heights.append(curr_height)
                strip_widths.append(curr_strip_width)
                strip_island_width_offsets.append(island_width_offsets)
                curr_strip_width = 0.0
                curr_strip_dy += curr_height
                curr_strip_indices = []
                island_width_offsets = []

        # append last strip (if not already captured by strip_width >= uv_pack_side_len)
        if len(curr_strip_indices) > 0:
            strips.append(curr_strip_indices)
            strip_dy.append(curr_strip_dy)
            strip_heights.append(uv_height[curr_strip_indices[0]])
            strip_widths.append(curr_strip_width)
            strip_island_width_offsets.append(island_width_offsets)
        
        # print(f"strips: {strips}")
        # print(f"strip_dy: {strip_dy}")
        # print(f"strip_heights: {strip_heights}")
        # print(f"strip_widths: {strip_widths}")
        # print(f"strip_island_width_offsets: {strip_island_width_offsets}")

        uvs_packed = np.zeros_like(uvs)
        for strip_indices, dy, island_width_offsets in zip(strips, strip_dy, strip_island_width_offsets):
            for i, dx in zip(strip_indices, island_width_offsets):
                # packed = original - aabb_min + [dx, dy]
                # dy = strip's height offset
                # dx = offset position within strip
                uvs_packed[i,:,:,:] = uvs[i,:,:,:] - uvs_aabb[i,0,:] + [dx, dy]
        
        # Apply paddings after layout:
        # (Note if there are large differences in number of islands per
        # strip, this will cause large differences in amount of applied
        # padding to each strip.)
        # 
        #     P      PP      P
        #     ____________________________
        #  P |  ____    ____              |
        #    | |    |  |    |             |
        #    | |____|  |____|             |
        #  P |                            |
        #  P |P ______ PP ____ PP ____ P  |
        #    | |      |  |    |  |    |   |
        #    | |______|  |____|  |____|   |
        #  P |                            |
        #  P |  ______    ______    ____  |
        #    | |      |  |      |  |    | |
        #    | |______|  |______|  |____| |
        #  P |____________________________|
        #     P        PP        PP      P
        #
        # Inject padding between each island...however we do padding in
        # terms of pixels. But UV space defined from [0,1], so first we need
        # to correspond uv space to pixel space based on the total uv bounds.
        #
        # Determine how much constant padding to apply such that when we finally
        # divide by uv scaling, the pad sizing is equal to padding in pixels desired.
        #
        # Define all scales such that: x * scale = tex_size
        # 
        # After padding, the vertical and horizontal total uv lengths are
        # (note we have to consider padding each horizontal strip separately
        # due to different number of islands per horizontal strip):
        #      Y = tex_size / scale_by_y  = 2 * pad * num_strips + sum(strip_heights)
        #     X0 = tex_size / scale_by_x0 = 2 * pad * num_islands_0 + strip_width_0
        #     X1 = tex_size / scale_by_x1 = 2 * pad * num_islands_1 + strip_width_1
        #        ...
        #     Xn = tex_size / scale_by_xn = 2 * pad * num_islands_n + strip_width_n
        #
        # foreach equation Y, X0, X1, ... XN
        #     pad_pixels = pad * scale_by_y
        #     pad_pixels = pad * scale_by_x0
        #     pad_pixels = pad * scale_by_x1
        #       ...
        #     pad_pixels = pad * scale_by_xn
        #
        # 2 equations, 2 unknowns (pad, scale_by_y), solve for pad, e.g. for Y:
        #     scale_by_y = pad_pixels / pad
        #     scale_by_y * tex_size  = tex_size * pad / pad_pixels = 2 * pad * num_strips + sum(strip_heights)
        #     pad = sum(strip_heights) / (tex_size / pad_pixels - 2 * num_strips)
        # 
        # Solve for each `scale_by_[ ]` value, then we will use the largest
        # scaling needed to scale entire uv map to fit into the (0, 1) square.
        # Final padding size in local space determined from scale:
        #     scale = max(scale_by_y, scale_by_x0, scale_by_x1, ..., scale_by_xn)
        #     pad = pad_pixels * scale

        # calculate y and scale_by_y after padding:
        if pad_pixels > 0:
            num_strips = len(strips)
            pad_candidates = np.zeros(1 + num_strips) # first for y vertical dir, rest for x strips
            scale_candidates = np.zeros_like(pad_candidates)
            
            pad_y = np.sum(strip_heights) / (float(tex_size) / float(pad_pixels) - 2.0 * num_strips)
            scale_by_y = float(pad_pixels) / pad_y

            pad_candidates[0] = pad_y
            scale_candidates[0] = scale_by_y

            # calculate each strip x and scale_by_x after padding:
            for i, (strip_indices, strip_width) in enumerate(zip(strips, strip_widths)):
                num_islands = len(strip_indices)
                pad_x = strip_width / (float(tex_size) / float(pad_pixels) - 2.0 * num_islands)
                scale_by_x = float(pad_pixels) / pad_x
                pad_candidates[i+1] = pad_x
                scale_candidates[i+1] = scale_by_x
            
            # print(f"pad_candidates: {pad_candidates}")
            # print(f"scale_candidates: {scale_candidates}")

            # find largest scale factor
            idx_scale = np.argmin(scale_candidates)
            global_uv_scale = scale_candidates[idx_scale]
        else:
            global_uv_max_x = np.max(uvs_packed[:,:,:,0])
            global_uv_max_y = np.max(uvs_packed[:,:,:,1])
            global_uv_scale = float(tex_size) / np.max([global_uv_max_x, global_uv_max_y])
            # print(f"global_uv_max_x: {global_uv_max_x}")
            # print(f"global_uv_max_y: {global_uv_max_y}")
            # print(f"global_uv_scale: {global_uv_scale}")

        # first scale UVs to target square tex_size: (tex_size, tex_size)
        uvs_packed *= global_uv_scale

        # round to nearest pixel
        # does not always do clean rounding: to do clean rounding
        # need to do this together with uv unwrap width/height calculations
        if round_to_pixels:
            if round_to_pixel_method == "closest":
                uvs_packed = uvs_packed.round(decimals=0)
            elif round_to_pixel_method == "width_height":
                for i in range(uvs_packed.shape[0]):
                    uvs_packed[i,:,:,:] = round_cuboid_uv_island_to_pixels(uvs_packed[i,:,:,:])
            else:
                raise ValueError(f"Unknown round_to_pixel_method: {round_to_pixel_method}")
        
        # apply padding in terms of pixels
        if pad_pixels > 0:
            dy = 0
            for strip_indices in strips:
                dy += pad_pixels
                dx = 0
                for i in strip_indices:
                    dx += pad_pixels
                    uvs_packed[i,:,:,:] = uvs_packed[i,:,:,:] + [dx, dy]
                    dx += pad_pixels
                dy += pad_pixels

        # finally scale down from `tex_size` to (0, 1) square
        if scale_to_unit:
            uvs_packed /= float(tex_size)

        # apply uvs to each object
        for i, obj in enumerate(cuboids):
            mesh = obj.data
            uv_layer = mesh.uv_layers.active.data

            # apply packed uvs to faces
            for f, face in enumerate(mesh.polygons):
                loop_start = face.loop_start
                uv_layer[loop_start].uv = uvs_packed[i,f,0,:]
                uv_layer[loop_start+1].uv = uvs_packed[i,f,1,:]
                uv_layer[loop_start+2].uv = uvs_packed[i,f,2,:]
                uv_layer[loop_start+3].uv = uvs_packed[i,f,3,:]

        if need_to_switch_mode_back:
            bpy.ops.object.mode_set(mode=user_mode)
        
        return {"FINISHED"}

DISABLE_MATERIAL_ENUM = (
    # (ID, Name, Description, Icon, Number)
    ("ENABLE", "Enable Material", "Enable material", "SNAP_VOLUME", 0),
    ("DISABLE", "Disable Material", "Disable material", "MESH_CUBE", 1),
)

class OpDisableMaterial(bpy.types.Operator):
    """Disable or enable selected face material in export"""
    bl_idname = "vintagestory.disable_material"
    bl_label = "Rename on Export"
    bl_options = {"REGISTER", "UNDO"}

    disable: bpy.props.EnumProperty(
        name="disable",
        description="Disable face material in export",
        default="ENABLE",
        items=DISABLE_MATERIAL_ENUM,
    )

    def execute(self, context):
        import bmesh
        args = self.as_keywords()

        # convert enum to boolean
        disable = args.get("disable", "ENABLE") == "DISABLE"

        # if not in edit mode, print error
        if context.mode != "EDIT_MESH":
            self.report({"ERROR"}, "Must select faces in Edit Mode")
            return {"FINISHED"}
        
        # get selected faces
        obj = context.object        
        bm = bmesh.from_edit_mesh(obj.data)
        
        # disable material for selected faces
        for f in bm.faces:
            if not f.select:
                continue
            if f.material_index < len(obj.material_slots):
                slot = obj.material_slots[f.material_index]
                material = slot.material
                material["disable"] = disable
                if disable:
                    self.report({"INFO"}, f"Disabled material: {material.name}")
                else:
                    self.report({"INFO"}, f"Enabled material: {material.name}")

        return {"FINISHED"}


class OpAssignGlow(bpy.types.Operator):
    """Assign glow custom property to object (0 to remove)"""
    bl_idname = "vintagestory.assign_glow"
    bl_label = "Glow"
    bl_options = {"REGISTER", "UNDO"}

    glow: bpy.props.IntProperty(
        name="Glow",
        description="Value of glow in 0 to 255",
        default=128,
        min=0,
        max=255,
        soft_min=0,
        soft_max=255,
    )

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        args = self.as_keywords()

        # unpack args
        glow = args.get("glow")
        
        if len(bpy.context.selected_objects) == 0:
            self.report({"ERROR"}, "No objects selected")
            return {"FINISHED"}

        if glow > 0:
            # add "glow" custom property to selected objects with value
            for obj in bpy.context.selected_objects:
                obj["glow"] = glow
        elif glow == 0:
            # remove glow custom property
            num_removed = 0
            for obj in bpy.context.selected_objects:
                if "glow" in obj:
                    num_removed += 1
                    del obj["glow"]
            self.report({"INFO"}, f"Removed glow property from {num_removed} objects")
        else:
            self.report({"ERROR"}, "Glow value must be > 0 or 0 to remove")
        
        # refresh n panel
        for region in context.area.regions:
            if region.type == "UI":
                region.tag_redraw()
        
        return {"FINISHED"}