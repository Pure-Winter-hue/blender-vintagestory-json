import bpy
from bpy import context
from mathutils import Vector, Euler, Quaternion, Matrix
from dataclasses import dataclass
import math
import numpy as np
import bmesh
import posixpath # need "/" separator
import os
import json
import re
from . import animation

import importlib
importlib.reload(animation)

# convert deg to rad
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# direction names for vs cube face UVs
DIRECTIONS = np.array([
    "north",
    "east",
    "west",
    "south",
    "up",
    "down",
])

# normals for vs directions in BLENDER world space
# e.g. blender (-1, 0, 0) is vs north (0, 0, -1)
# shape (f,n) = (6,3)
#   f = 6: number of cuboid faces to test
#   v = 3: vertex coordinates (x,y,z)
DIRECTION_NORMALS = np.array([
    [-1.,  0.,  0.],
    [ 0.,  1.,  0.],
    [ 0., -1.,  0.],
    [ 1.,  0.,  0.],
    [ 0.,  0.,  1.],
    [ 0.,  0., -1.],
])
# DIRECTION_NORMALS = np.tile(DIRECTION_NORMALS[np.newaxis,...], (6,1,1))

# blender counterclockwise uv -> vs uv rotation lookup table
# (these values experimentally determined)
# access using [uv_loop_start_index][vert_loop_start_index]
COUNTERCLOCKWISE_UV_ROTATION_LOOKUP = (
    (0, 270, 180, 90),
    (90, 0, 270, 180),
    (180, 90, 0, 270),
    (270, 180, 90, 0),
)

# blender clockwise uv -> vs uv rotation lookup table
# (these values experimentally determined)
# access using [uv_loop_start_index][vert_loop_start_index]
# Note: vs uv must also be x-flipped
CLOCKWISE_UV_ROTATION_LOOKUP = (
    (90, 0, 270, 180),
    (0, 270, 180, 90),
    (270, 180, 90, 0),
    (180, 90, 0, 270),
)

def filter_root_objects(objects):
    """Get root objects (objects without parents) in scene."""
    root_objects = []
    for obj in objects:
        if obj.parent is None:
            root_objects.append(obj)
    return root_objects


def matrix_roughly_equal(m1, m2, eps=1e-5):
    """Return if two matrices are roughly equal
    by comparing elements.
    """
    for i in range(0, 4):
        for j in range(0, 4):
            if abs(m1[i][j] - m2[i][j]) > eps:
                return False
    return True


def sign(x):
    """Return sign of value as 1 or -1"""
    if x >= 0:
        return 1
    else:
        return -1


def to_vintagestory_axis(ax):
    """Convert blender space to VS space:
    X -> Z
    Y -> X
    Z -> Y
    """
    if "X" in ax:
        return "Z"
    elif "Y" in ax:
        return "X"
    elif "Z" in ax:
        return "Y"


def to_y_up(arr):
    """Convert blender space to VS space:
    X -> Z
    Y -> X
    Z -> Y
    """
    return np.array([arr[1], arr[2], arr[0]])


def to_vintagestory_rotation(rot):
    """Convert blender space rotation to VS space:
    VS space XYZ euler is blender space XZY euler,
    so convert euler order, then rename axes
        X -> Z
        Y -> X
        Z -> Y
    Inputs:
    - euler: Blender rotation, either euler or quat
    """
    if isinstance(rot, Quaternion):
        r = rot.to_euler("XZY")
    else: # euler or rotation matrix
        r = rot.to_quaternion().to_euler("XZY")
    return np.array([
        r.y * RAD_TO_DEG,
        r.z * RAD_TO_DEG,
        r.x * RAD_TO_DEG,
    ])


def clamp_rotation(r):
    """Clamp a euler angle rotation in numpy array format
    [rx, ry, rz] to within bounds [-360, 360]
    """
    while r[0] > 360.0:
        r[0] -= 360.0
    while r[0] < -360.0:
        r[0] += 360.0
    
    while r[1] > 360.0:
        r[1] -= 360.0
    while r[1] < -360.0:
        r[1] += 360.0
    
    while r[2] > 360.0:
        r[2] -= 360.0
    while r[2] < -360.0:
        r[2] += 360.0
    
    return r


class TextureInfo():
    """Description of a texture, gives image source path and size
    """
    def __init__(self, path, size):
        self.path = path
        self.size = size


def get_material_color(mat):
    """Get material color as tuple (r, g, b, a). Return None if no
    material node has color property.
    Inputs:
    - mat: Material
    Returns:
    - color tuple (r, g, b,a ) if a basic color,
      "texture_path" string if a texture,
      or None if no color/path
    """
    # get first node with valid color
    if mat.node_tree is not None:
        for n in mat.node_tree.nodes:
            # principled BSDF
            if "Base Color" in n.inputs:
                node_color = n.inputs["Base Color"]
                # check if its a texture path
                for link in node_color.links:
                    from_node = link.from_node
                    if isinstance(from_node, bpy.types.ShaderNodeTexImage):
                        if from_node.image is not None and from_node.image.filepath != "":
                            img = from_node.image
                            img_size = [img.size[0], img.size[1]]
                            return TextureInfo(img.filepath, img_size)
                # else, export color tuple
                color = node_color.default_value
                color = (color[0], color[1], color[2], color[3])
                return color
            # most other materials with color
            elif "Color" in n.inputs:
                color = n.inputs["Color"].default_value
                color = (color[0], color[1], color[2], color[3])
                return color
    
    return None


@dataclass
class FaceMaterial:
    """Face material data for a single face of a cuboid.
    """
    COLOR = 0
    TEXTURE = 1
    DISABLE = 2

    # type enum, one of the integers above 
    type: int
    # name of material
    name: str
    # color
    color: tuple[int, int, int, int] = (0.0, 0.0, 0.0, 1.0),
    # texture path + size
    texture_path: str = ""
    # texture size
    texture_size: tuple[int, int] = (0, 0)
    # material glow, 0 to 255
    glow: int = 0


    def from_face(
        obj,
        material_index: int,
        default_color = (0.0, 0.0, 0.0, 1.0),
    ):
        """Get obj material color in index as either 
        - tuple (r, g, b, a) if using a default color input
        - texture file name string "path" if using a texture input
        """
        material = None
        if material_index < len(obj.material_slots):
            slot = obj.material_slots[material_index]
            material = slot.material
            if material is not None:
                # check if material is disabled, return disable material
                if "disable" in material and material["disable"] == True:
                    return FaceMaterial(
                        type=FaceMaterial.DISABLE,
                        name=material.name,
                    )
                
                # get glow from object or material
                if "glow" in obj:
                    glow = obj["glow"]
                elif "glow" in material:
                    glow = material["glow"]
                else:
                    glow = 0

                color = get_material_color(material)
                if color is not None:
                    if isinstance(color, tuple):
                        return FaceMaterial(
                            type=FaceMaterial.COLOR,
                            name=material.name,
                            color=color,
                            glow=glow,
                        )
                    # texture
                    elif isinstance(color, TextureInfo):
                        return FaceMaterial(
                            type=FaceMaterial.TEXTURE,
                            name=material.name,
                            color=default_color,
                            texture_path=color.path,
                            texture_size=color.size,
                            glow=glow,
                        )
                    
                # warn that material has no color or texture
                print(f"WARNING: {obj.name} material {material.name} has no color or texture")
            
        return FaceMaterial(
            type=FaceMaterial.COLOR,
            name=material.name if material else 'null',
            color=default_color,
        )


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


def create_color_texture(
    colors,
    min_size = 16,
):
    """Create a packed square texture from list of input colors. Each color
    is a distinct RGB tuple given a 3x3 pixel square in the texture. These
    must be 3x3 pixels so that there is no uv bleeding near the face edges.
    Also includes a tile for a default color for faces with no material.
    This is the next unfilled 3x3 tile.

    Inputs:
    - colors: Iterable of colors. Each color should be indexable like an rgb
              tuple c = (r, g, b), just so that r = c[0], b = c[1], g = c[2].
    - min_size: Minimum size of texture (must be power 2^n). By default
                16 because vs needs min sized 16 textures for 4 mipmap levels.'
    
    Returns:
    - tex_pixels: Flattened array of texture pixels.
    - tex_size: Size of image texture.
    - color_tex_uv_map: Dict map from rgb tuple color to vs format uv coords
                        (r, g, b) -> (xmin, ymin, xmax, ymax)
    - default_color_uv: Default uv coords for unmapped materials (xmin, ymin, xmax, ymax).
    """
    # blender interprets (r,g,b,a) in sRGB space
    def linear_to_sRGB(v):
        if v < 0.0031308:
            return v * 12.92
        else:
            return 1.055 * (v ** (1/2.4)) - 0.055
    
    # fit textures into closest (2^n,2^n) sized texture
    # each color takes a (3,3) pixel chunk to avoid color
    # bleeding at UV edges seams
    # -> get smallest n to fit all colors, add +1 for a default color tile
    color_grid_size = math.ceil(math.sqrt(len(colors) + 1)) # colors on each axis
    tex_size = max(min_size, 2 ** math.ceil(math.log2(3 * color_grid_size))) # fit to (2^n, 2^n) image
    
    # composite colors into white RGBA grid
    tex_colors = np.ones((color_grid_size, color_grid_size, 4))
    color_tex_uv_map = {}
    for i, c in enumerate(colors):
        # convert color to sRGB
        c_srgb = (linear_to_sRGB(c[0]), linear_to_sRGB(c[1]), linear_to_sRGB(c[2]), c[3])

        tex_colors[i // color_grid_size, i % color_grid_size, :] = c_srgb
        
        # uvs: [x1, y1, x2, y2], each value from [0, 16] as proportion of image
        # map each color to a uv
        x1 = ( 3*(i % color_grid_size) + 1 ) / tex_size * 16
        x2 = ( 3*(i % color_grid_size) + 2 ) / tex_size * 16
        y1 = ( 3*(i // color_grid_size) + 1 ) / tex_size * 16
        y2 = ( 3*(i // color_grid_size) + 2 ) / tex_size * 16
        color_tex_uv_map[c] = [x1, y1, x2, y2]
    
    # default color uv coord (last coord + 1)
    idx = len(colors)
    default_color_uv = [
        ( 3*(idx % color_grid_size) + 1 ) / tex_size * 16,
        ( 3*(idx // color_grid_size) + 1 ) / tex_size * 16,
        ( 3*(idx % color_grid_size) + 2 ) / tex_size * 16,
        ( 3*(idx // color_grid_size) + 2 ) / tex_size * 16
    ]

    # triple colors into 3x3 pixel chunks
    tex_colors = np.repeat(tex_colors, 3, axis=0)
    tex_colors = np.repeat(tex_colors, 3, axis=1)
    tex_colors = np.flip(tex_colors, axis=0)

    # pixels as flattened array (for blender Image api)
    tex_pixels = np.ones((tex_size, tex_size, 4))
    tex_pixels[-tex_colors.shape[0]:, 0:tex_colors.shape[1], :] = tex_colors
    tex_pixels = tex_pixels.flatten("C")

    return tex_pixels, tex_size, color_tex_uv_map, default_color_uv


def generate_mesh_element(
    obj,                           # current object
    skip_disabled_render=True,     # skip children with disabled render 
    parent=None,                   # parent Blender object
    armature=None,                 # Blender Armature object (NOT Armature data)
    bone_hierarchy=None,           # map of armature bones => children mesh objects
    is_bone_child=False,           # is a child to a dummy bone element
    groups=None,                   # running dict of collections
    model_colors=None,             # running dict of all model colors
    model_textures=None,           # running dict of all model textures
    parent_matrix_world=None,      # parent matrix world transform 
    parent_cube_origin=None,       # parent cube "from" origin (coords in VintageStory space)
    parent_rotation_origin=None,   # parent object rotation origin (coords in VintageStory space)
    # Some callers (export tree rebuild) pass this flag through generate_element.
    # Mesh elements handle children outside this function, so we ignore it here.
    skip_children_recurse=False,
    export_uvs=True,               # export uvs
    export_generated_texture=True,
    texture_size_x_override=None,  # override texture size x
    texture_size_y_override=None,  # override texture size y
    **_ignored_kwargs,
):
    """Recursive function to generate output element from
    Blender object

    See diagram from importer for location transformation.

    VintageStory => Blender space:
        child_blender_origin = parent_cube_origin - parent_rotation_origin + child_rotation_origin
        from_blender_local = from - child_rotation_origin
        to_blender_local = to - child_rotation_origin

    Blender space => VS Space:
        child_rotation_origin = child_blender_origin - parent_cube_origin + parent_rotation_origin
        from = from_blender_local + child_rotation_origin
        to = to_blender_local + child_rotation_origin
    """

    mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh):
        return None
    
    obj_name = obj.name # may be overwritten if part of armature
    
    # count number of vertices, ignore if not cuboid
    num_vertices = len(mesh.vertices)
    if num_vertices != 8:
        return None

    """
    object blender origin and rotation
    -> if this is part of an armature, must get relative
    to parent bone
    """
        # Use world-space transform instead of obj.location/rotation_euler.
    # Bone-parenting stores obj.location in bone space which breaks roundtrip centering.
    matrix_world = obj.matrix_world.copy()
    origin = matrix_world.translation.copy()
    bone_location = None
    bone_origin = None
    try:
        obj_rotation = matrix_world.to_quaternion().to_euler("XYZ")
    except Exception:
        obj_rotation = obj.rotation_euler.copy()
    origin_bone_offset = np.array([0., 0., 0.])
    is_main_bone_mesh = False

    if armature is not None and obj.parent is not None and obj.parent_bone != "":
        bone_name = obj.parent_bone
        if bone_name in armature.data.bones and bone_name in bone_hierarchy and bone_hierarchy[bone_name].main.name == obj.name:
            # origin_bone_offset = obj.location - origin
            bone = armature.data.bones[bone_name]
            # bone_location = relative location of bone from its parent bone
            if bone.parent is not None:
                bone_location = bone.head_local - bone.parent.head_local
            else:
                bone_location = bone.head_local
            
            bone_origin = bone.head_local
            is_main_bone_mesh = True
            origin_bone_offset = np.array([0., 0., 0.])
            matrix_world.translation = bone.head_local
    
    # use step parent bone if available
    if "StepParentName" in obj and len(obj["StepParentName"]) > 0:
        step_parent_name = obj["StepParentName"]
        # "b_[name]" is hard-coded bone prefix, convention for this
        # plugin. if step parent name starts with "b_", remove prefix
        if step_parent_name.startswith("b_"):
            bone_parent_name = step_parent_name[2:]
        else:
            bone_parent_name = step_parent_name
    else:
        step_parent_name = None
        bone_parent_name = None

    if armature is not None and bone_parent_name is not None and bone_parent_name in armature.data.bones and bone_parent_name in bone_hierarchy:
        parent_bone = armature.data.bones[bone_parent_name]
        parent_matrix_world = parent_bone.matrix_local.copy()
        parent_cube_origin = parent_bone.head
        parent_rotation_origin = parent_bone.head
    else:
        print(f"WARNING: cannot find {obj.name} step parent bone {bone_parent_name}")

    # more robust but higher performance cost, just get relative
    # location/rotation from world matrices, required for complex
    # parent hierarchies with armature bones + object-object parenting
    # TODO: global flag for mesh with an armature so this is used instead
    # of just obj.location and obj.rotation_euler
    if parent_matrix_world is not None:
        # print(obj.name, "parent_matrix_world", parent_matrix_world)
        mat_local = parent_matrix_world.inverted_safe() @ matrix_world
        origin, quat, _ = mat_local.decompose()
        obj_rotation = quat.to_euler("XYZ")

        # adjustment for vertices
        if bone_origin is not None:
            origin_bone_offset = obj_rotation.to_matrix().to_4x4().inverted_safe() @ origin_bone_offset
    # using bone origin instead of parent origin offset
    elif bone_location is not None:
        origin = bone_location
    
    # ================================
    # get local mesh coordinates
    # ================================
    v_local = np.zeros((3, 8))
    for i, v in enumerate(mesh.vertices):
        v_local[0:3,i] = v.co
    
    # apply scale matrix to local vertices
    if obj.scale[0] != 1.0 or obj.scale[1] != 1.0 or obj.scale[2] != 1.0:
        scale_matrix = np.array([
            [obj.scale[0], 0, 0],
            [0, obj.scale[1], 0],
            [0, 0, obj.scale[2]],
        ])
        v_local = scale_matrix @ v_local

    # create output coords, rotation
    # get min/max for to/from points
    v_min = np.amin(v_local, axis=1)
    v_max = np.amax(v_local, axis=1)

    # change axis to vintage story y-up axis
    v_min = to_y_up(v_min)
    v_max = to_y_up(v_max)
    origin = to_y_up(origin)
    origin_bone_offset = to_y_up(origin_bone_offset)
    rotation = to_vintagestory_rotation(obj_rotation)
    
    # translate to vintage story coord space
    rotation_origin = origin - parent_cube_origin + parent_rotation_origin
    v_from = v_min + rotation_origin + origin_bone_offset
    v_to = v_max + rotation_origin + origin_bone_offset
    cube_origin = v_from
    
    # ================================
    # texture/uv generation
    # 
    # NOTE: BLENDER VS vs/VINTAGE STORY UV AXIS
    # - blender: uvs origin is bottom-left (0,0) to top-right (1, 1)
    # - vs/vs: uvs origin is top-left (0,0) to bottom-right (16, 16)
    # vs uvs: [x1, y1, x2, y2], each value from [0, 16] as proportion of image
    # as well as 0, 90, 180, 270 degree uv rotation

    # uv loop to export depends on:
    # - clockwise/counterclockwise order
    # - uv starting coordinate (determines rotation) relative to face
    #   vertex loop starting coordinate
    # 
    # Assume "natural" index order of face vertices and uvs without
    # any rotations in local mesh space is counterclockwise loop:
    #   3___2      ^ +y
    #   |   |      |
    #   |___|      ---> +x
    #   0   1
    # 
    # uv, vertex starting coordinate is based on this loop.
    # Use the uv rotation lookup tables constants to determine rotation.
    # ================================

    # initialize faces
    faces = {
        "north": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
        "east": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
        "south": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
        "west": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
        "up": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
        "down": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
    }

    # Disabled face directions persisted on the object (importer/op).
    # Stored as a list of VS direction strings, e.g. ['down', 'up'].
    disabled_dirs_prop = set()
    try:
        v = obj.get("vs_disabled_faces")
        if isinstance(v, (list, tuple, set)):
            disabled_dirs_prop = set(str(x) for x in v)
        elif isinstance(v, str) and v:
            try:
                vv = json.loads(v)
                if isinstance(vv, (list, tuple, set)):
                    disabled_dirs_prop = set(str(x) for x in vv)
            except Exception:
                pass
    except Exception:
        pass

    # Robust UV access for Blender 4.x+.
    # In Blender 4.x/4.5, UVs may be backed by mesh.attributes (CORNER/FLOAT2)
    # and mesh.uv_layers.active.data can sometimes be empty (len==0).
    uv_layer = None               # classic UVLoop data (preferred)
    uv_attr = None                # mesh.attributes fallback
    bm = None                     # last-resort fallback
    bm_uv = None
    bm_faces_by_poly_index = None
    uv_name = None

    if export_uvs:
        # 1) Try classic uv_layers.active.data
        try:
            uvl = mesh.uv_layers.active
            if uvl is not None:
                uv_name = uvl.name
                if uvl.data is not None and len(uvl.data) == len(mesh.loops):
                    uv_layer = uvl.data
        except Exception:
            uv_layer = None

        # 2) Try attributes-based UVs (Blender 4.x)
        if uv_layer is None:
            try:
                if uv_name is None and getattr(mesh, "uv_layers", None) and mesh.uv_layers.active is not None:
                    uv_name = mesh.uv_layers.active.name
                attr = mesh.attributes.get(uv_name) if uv_name else None
                if attr is not None and getattr(attr, "domain", None) == 'CORNER' and len(attr.data) == len(mesh.loops):
                    uv_attr = attr
            except Exception:
                uv_attr = None

        # 3) Last resort: bmesh loop UV layer
        if uv_layer is None and uv_attr is None:
            try:
                bm = bmesh.new()
                bm.from_mesh(mesh)
                bm.faces.ensure_lookup_table()
                bm_uv = bm.loops.layers.uv.active
                if bm_uv is None:
                    export_uvs = False
                    bm.free()
                    bm = None
                    bm_uv = None
                else:
                    # Keep a direct index lookup only if face counts match.
                    if len(bm.faces) == len(mesh.polygons):
                        bm_faces_by_poly_index = bm.faces
            except Exception:
                export_uvs = False
                if bm is not None:
                    bm.free()
                bm = None
                bm_uv = None
                bm_faces_by_poly_index = None


    
    # Group polygons by direction so triangulated or subdivided cube faces
    # still export correctly. (Some Blender ops keep 8 verts but increase poly count.)
    dir_polys = {k: [] for k in faces.keys()}

    for poly in mesh.polygons:
        face_normal = np.array(poly.normal)
        face_normal_stacked = np.tile(face_normal, (6, 1))
        face_direction_index = np.argmax(np.sum(face_normal_stacked * DIRECTION_NORMALS, axis=1), axis=0)
        d = DIRECTIONS[face_direction_index]
        if d in dir_polys:
            dir_polys[d].append(poly.index)

    def _corner_index(x, y, minx, maxx, miny, maxy, eps=1e-6):
        # 0=bottom-left, 1=bottom-right, 2=top-right, 3=top-left
        # Use distance-to-extremes instead of strict comparisons to avoid float jitter.
        left = abs(x - minx) <= abs(x - maxx) + eps
        bottom = abs(y - miny) <= abs(y - maxy) + eps
        if bottom and left:
            return 0
        if bottom and not left:
            return 1
        if (not bottom) and (not left):
            return 2
        return 3

    def _project_vert_2d(v, face_normal):
        # v is a length-3 numpy-like (x,y,z). Returns (px,py) in the face plane.
        if face_normal[0] > 0.5:      # normal = (1, 0, 0)
            return (v[1], v[2])
        if face_normal[0] < -0.5:     # normal = (-1, 0, 0)
            return (-v[1], v[2])
        if face_normal[1] > 0.5:      # normal = (0, 1, 0)
            return (-v[0], v[2])
        if face_normal[1] < -0.5:     # normal = (0, -1, 0)
            return (v[0], v[2])
        if face_normal[2] > 0.5:      # normal = (0, 0, 1)
            return (v[1], -v[0])
        # face_normal[2] < -0.5       # normal = (0, 0, -1)
        return (v[1], v[0])

    for d, poly_indices in dir_polys.items():

        # If this face direction is explicitly marked disabled, export as disabled.
        if d in disabled_dirs_prop:
            faces[d]["texture"] = "#0"
            faces[d]["uv"] = [0, 0, 0, 0]
            faces[d]["autoUv"] = False
            faces[d]["enabled"] = False
            faces[d].pop("rotation", None)
            faces[d].pop("glow", None)
            continue
        if not poly_indices:
            # Face missing in Blender mesh: treat as a disabled face in VS JSON
            faces[d]["texture"] = "#0"
            faces[d]["uv"] = [0, 0, 0, 0]
            faces[d]["autoUv"] = False
            faces[d]["enabled"] = False
            continue

        # Representative polygon: largest area for stable winding/loop start.
        rep_poly = max((mesh.polygons[i] for i in poly_indices), key=lambda p: p.area)
        rep_normal = np.array(rep_poly.normal)

        face_material = FaceMaterial.from_face(
            obj,
            rep_poly.material_index,
        )

        # disabled face
        if face_material.type == FaceMaterial.DISABLE:
            faces[d]["texture"] = "#0"  # VSMC-style disabled face
            # VSMC-style disabled face: keep explicit marker and clear UVs.
            faces[d]["uv"] = [0, 0, 0, 0]
            faces[d]["autoUv"] = False
            faces[d]["enabled"] = False
            continue
        # solid color tuple
        if face_material.type == FaceMaterial.COLOR and export_generated_texture:
            faces[d] = face_material  # replace face with face material, will convert later
            if model_colors is not None:
                model_colors.add(face_material.color)
            continue
        # texture
        if face_material.type != FaceMaterial.TEXTURE:
            continue

        faces[d]["texture"] = "#0"  # VSMC-style disabled face
        model_textures[face_material.name] = face_material

        # face glow
        if face_material.glow > 0:
            faces[d]["glow"] = face_material.glow

        tex_width = face_material.texture_size[0] if texture_size_x_override is None else texture_size_x_override
        tex_height = face_material.texture_size[1] if texture_size_y_override is None else texture_size_y_override

        if not export_uvs:
            continue

        # Collect UV + projected-vert samples across all polys in this direction to get a robust bbox.
        uv_samples = []
        vert2d_samples = []

        # Also collect loop sequence for the representative poly (for rotation inference).
        rep_uvs = []
        rep_verts2d = []

        for pi in poly_indices:
            poly = mesh.polygons[pi]
            loop_start = poly.loop_start
            loop_total = poly.loop_total

            # bmesh face (only if we constructed bm and uv layer exists there)
            bm_face = None
            if bm_faces_by_poly_index is not None and bm_uv is not None:
                try:
                    bm_face = bm_faces_by_poly_index[pi]
                except Exception:
                    bm_face = None

            for li in range(loop_total):
                # UV (prefer uv_layers, then attributes, then bmesh)
                if uv_layer is not None:
                    luv = uv_layer[loop_start + li].uv
                    u, v = float(luv[0]), float(luv[1])
                elif uv_attr is not None:
                    a = uv_attr.data[loop_start + li]
                    # Blender attribute API varies: try vector/uv/value
                    if hasattr(a, "vector"):
                        luv = a.vector
                    elif hasattr(a, "uv"):
                        luv = a.uv
                    else:
                        luv = a.value
                    u, v = float(luv[0]), float(luv[1])
                elif bm_face is not None:
                    luv = bm_face.loops[li][bm_uv].uv
                    u, v = float(luv[0]), float(luv[1])
                else:
                    # no UVs available
                    continue

                uv_samples.append((u, v))

                # Vertex -> 2D in face plane
                vidx = mesh.loops[loop_start + li].vertex_index
                v3 = v_local[:, vidx]
                p2 = _project_vert_2d(v3, rep_normal)
                vert2d_samples.append((float(p2[0]), float(p2[1])))

                # Representative poly sequences
                if pi == rep_poly.index:
                    rep_uvs.append((u, v))
                    rep_verts2d.append((float(p2[0]), float(p2[1])))

        if not uv_samples or not rep_uvs:
            continue

        if not uv_samples:
            # No UV data for this face: treat as disabled.
            faces[d]["texture"] = "#0"
            faces[d]["uv"] = [0, 0, 0, 0]
            faces[d]["autoUv"] = False
            faces[d]["enabled"] = False
            faces[d].pop("rotation", None)
            faces[d].pop("glow", None)
            continue

        uv_min_x = min(u for u, _ in uv_samples)
        uv_max_x = max(u for u, _ in uv_samples)
        uv_min_y = min(v for _, v in uv_samples)
        uv_max_y = max(v for _, v in uv_samples)

        vert_min_x = min(x for x, _ in vert2d_samples)
        vert_max_x = max(x for x, _ in vert2d_samples)
        vert_min_y = min(y for _, y in vert2d_samples)
        vert_max_y = max(y for _, y in vert2d_samples)

        # If representative poly isn't a quad, bbox export is still fine but
        # rotation inference becomes unreliable. Fall back to no rotation.
        if rep_poly.loop_total != 4 or len(rep_uvs) < 4 or len(rep_verts2d) < 4:
            face_uvs = np.array([uv_min_x, uv_max_y, uv_max_x, uv_min_y], dtype=float)
            face_uv_rotation = 0
        else:
            uv_clockwise = loop_is_clockwise(rep_uvs)
            vert_clockwise = loop_is_clockwise(rep_verts2d)

            # Determine start-corner indices using bbox corners with tolerance.
            uv_loop_start_index = _corner_index(rep_uvs[0][0], rep_uvs[0][1], uv_min_x, uv_max_x, uv_min_y, uv_max_y)
            vert_loop_start_index = _corner_index(rep_verts2d[0][0], rep_verts2d[0][1], vert_min_x, vert_max_x, vert_min_y, vert_max_y)

            face_uvs = np.zeros((4,), dtype=float)

            if (not uv_clockwise) and (not vert_clockwise):
                # TL-BR rectangle in UV space (y handled later)
                face_uvs[0] = uv_min_x
                face_uvs[1] = uv_max_y
                face_uvs[2] = uv_max_x
                face_uvs[3] = uv_min_y
                face_uv_rotation = COUNTERCLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
            elif uv_clockwise and (not vert_clockwise):
                # mirrored in U, then rotate
                face_uvs[0] = uv_max_x
                face_uvs[1] = uv_max_y
                face_uvs[2] = uv_min_x
                face_uvs[3] = uv_min_y
                face_uv_rotation = CLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
            elif (not uv_clockwise) and vert_clockwise:
                # rare, but keep consistent with previous fallback behavior
                face_uvs[0] = uv_max_x
                face_uvs[1] = uv_max_y
                face_uvs[2] = uv_min_x
                face_uvs[3] = uv_min_y
                face_uv_rotation = CLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
            else:
                face_uvs[0] = uv_min_x
                face_uvs[1] = uv_max_y
                face_uvs[2] = uv_max_x
                face_uvs[3] = uv_min_y
                face_uv_rotation = COUNTERCLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]

        xmin = face_uvs[0] * tex_width
        ymin = (1.0 - face_uvs[1]) * tex_height
        xmax = face_uvs[2] * tex_width
        ymax = (1.0 - face_uvs[3]) * tex_height

        # "down" orientation quirk preserved from original exporter
        if d == "down":
            xmin, xmax = xmax, xmin
            ymin, ymax = ymax, ymin

        faces[d]["uv"] = [xmin, ymin, xmax, ymax]

        # If UVs are fully cleared (VSMC-style disable), export as disabled even if geometry exists.
        eps = 1e-6
        if (abs(xmin) <= eps and abs(ymin) <= eps and abs(xmax) <= eps and abs(ymax) <= eps):
            faces[d]["texture"] = "#0"
            faces[d]["uv"] = [0, 0, 0, 0]
            faces[d]["autoUv"] = False
            faces[d]["enabled"] = False
            faces[d].pop("rotation", None)
            faces[d].pop("glow", None)
            continue

        if face_uv_rotation != 0 and face_uv_rotation != 360:
            faces[d]["rotation"] = face_uv_rotation if face_uv_rotation >= 0 else 360 + face_uv_rotation


    # free bmesh if we used it for UV reads
    if bm is not None:
        try:
            bm.free()
        except Exception:
            pass


    # ================================
    # build children
    # ================================
    children = []
    attachpoints = []
    
    # parse direct children objects normally
    if not skip_children_recurse:
        for child in obj.children:
            if skip_disabled_render and child.hide_render:
                continue

            child_element = generate_element(
                child,
                skip_disabled_render=skip_disabled_render,
                parent=obj,
                armature=None,
                bone_hierarchy=None,
                groups=groups,
                model_colors=model_colors,
                model_textures=model_textures,
                parent_matrix_world=matrix_world,
                parent_cube_origin=cube_origin,
                parent_rotation_origin=rotation_origin,
                export_uvs=export_uvs,
                texture_size_x_override=texture_size_x_override,
                texture_size_y_override=texture_size_y_override,
            )
            if child_element is not None:
                if child.name.startswith("attach_"):
                    attachpoints.append(child_element)
                else:
                    children.append(child_element)

    # use parent bone children if this is part of an armature
    if not skip_children_recurse:
        if bone_hierarchy is not None and obj.parent is not None and obj.parent_type == "ARMATURE":
            bone_obj_children = []
            parent_bone_name = obj.parent_bone
            if parent_bone_name != "" and parent_bone_name in bone_hierarchy and parent_bone_name in armature.data.bones:
                # if this is main bone, parent other objects to this
                if bone_hierarchy[parent_bone_name].main.name == obj.name:
                    # Keep exported element names based on the current object name.
                    # Bone names are used internally for hierarchy and animation sampling;
                    # we map them to export names separately during export.

                    # parent other objects in same bone to this object
                    if len(bone_hierarchy[parent_bone_name].children) > 1:
                        bone_obj_children.extend(bone_hierarchy[parent_bone_name].children[1:])
                
                    # parent children main objects to this
                    parent_bone = armature.data.bones[parent_bone_name]
                    for child_bone in parent_bone.children:
                        child_bone_name = child_bone.name
                        if child_bone_name in bone_hierarchy:
                            bone_obj_children.append(bone_hierarchy[child_bone_name].main)
    
            for child in bone_obj_children:
                if skip_disabled_render and child.hide_render:
                    continue
            
                child_element = generate_element(
                    child,
                    skip_disabled_render=skip_disabled_render,
                    parent=obj,
                    armature=armature,
                    bone_hierarchy=bone_hierarchy,
                    groups=groups,
                    model_colors=model_colors,
                    model_textures=model_textures,
                    parent_matrix_world=matrix_world,
                    parent_cube_origin=cube_origin,
                    parent_rotation_origin=rotation_origin,
                    export_uvs=export_uvs,
                    texture_size_x_override=texture_size_x_override,
                    texture_size_y_override=texture_size_y_override,
                )
                if child_element is not None:
                    if child.name.startswith("attach_"):
                        attachpoints.append(child_element)
                    else:
                        children.append(child_element)

    # ================================
    # build element
    # ================================
    # ------------------------------------------------
    # Choose the name to write into VS JSON.
    #
    # Importer stores the original VS name in obj["vs_name"] to preserve whitespace,
    # but we must respect user renames in Blender.
    #
    # Priority:
    #   1) obj["rename"] override (explicit)
    #   2) if user did NOT rename since import: use original VS name (obj["vs_name"]) to preserve whitespace
    #   3) otherwise: use current Blender object name (obj_name)
    # ------------------------------------------------
    export_name = obj_name

    # Explicit override
    if "rename" in obj and isinstance(obj["rename"], str) and obj["rename"].strip():
        export_name = obj["rename"]
    else:
        vs_name = obj.get("vs_name") if hasattr(obj, "get") else None
        import_blender_name = obj.get("vs_import_blender_name") if hasattr(obj, "get") else None
        try:
            if isinstance(vs_name, str) and isinstance(import_blender_name, str) and obj.name == import_blender_name:
                export_name = vs_name
        except Exception:
            pass
    
    new_element = {
        "name": export_name,
        "from": v_from,
        "to": v_to,
        "rotationOrigin": rotation_origin,
    }

    # step parent name
    if step_parent_name is not None:
        new_element["stepParentName"] = step_parent_name
    
    # add rotations
    if rotation[0] != 0.0:
        new_element["rotationX"] = rotation[0]
    if rotation[1] != 0.0:
        new_element["rotationY"] = rotation[1]
    if rotation[2] != 0.0:
        new_element["rotationZ"] = rotation[2]

    # add collection link
    users_collection = obj.users_collection
    if len(users_collection) > 0:
        new_element["group"] = users_collection[0].name

    # add faces
    # Storage helper volumes: export faceless so VS does not render them.
    if (isinstance(export_name, str) and ("storage" in export_name.casefold())) or bool(obj.get("vs_is_storage", False) if hasattr(obj,"get") else False):
        new_element["faces"] = {}
    else:
        new_element["faces"] = faces

    # add children
    new_element["children"] = children

    # add attachpoints if they exist
    if len(attachpoints) > 0:
        new_element["attachmentpoints"] = attachpoints
    
    return new_element


def generate_attach_point(
    obj,                           # current object
    parent=None,                   # parent Blender object
    armature=None,                 # Blender Armature object (NOT Armature data)
    parent_matrix_world=None,      # parent matrix world transform
    parent_cube_origin=None,       # parent cube "from" origin (coords in VintageStory space)
    parent_rotation_origin=None,   # parent object rotation origin (coords in VintageStory space)
    skip_children_recurse=False,   # if True, do not recurse Blender children/bone children
    **kwargs,
):
    """Parse an attachment point
    """
    if not obj.name.startswith("attach_"):
        return None
    
    # get attachpoint name
    name = obj.name[7:]

    """
    object blender origin and rotation
    -> if this is part of an armature, must get relative
    to parent bone
    """
    origin = np.array(obj.location)
    obj_rotation = obj.rotation_euler

    if armature is not None and obj.parent is not None and obj.parent_bone != "":
        bone_name = obj.parent_bone
        if bone_name in armature.data.bones:
            bone_matrix = armature.data.bones[bone_name].matrix_local
            # print(obj.name, "BONE MATRIX:", bone_matrix)
        mat_loc = parent.matrix_world.inverted_safe() @ obj.matrix_world
        origin, quat, _ = mat_loc.decompose()
        obj_rotation = quat.to_euler("XYZ")
    
    # more robust but higher performance cost, just get relative
    # location/rotation from world matrices, required for complex
    # parent hierarchies with armature bones + object-object parenting
    if parent_matrix_world is not None:
        # print(obj.name, "parent_matrix_world", parent_matrix_world)
        mat_local = parent_matrix_world.inverted_safe() @ obj.matrix_world
        origin, quat, _ = mat_local.decompose()
        obj_rotation = quat.to_euler("XYZ")

    # change axis to vintage story y-up axis
    origin = to_y_up(origin)
    rotation = to_vintagestory_rotation(obj_rotation)
    
    # translate to vintage story coord space
    rotation_origin = origin - parent_cube_origin + parent_rotation_origin

    export_name = name
    if "rename" in obj and isinstance(obj["rename"], str):
        export_name = obj["rename"]
    
    return {
        "code": export_name,
        "posX": rotation_origin[0],
        "posY": rotation_origin[1],
        "posZ": rotation_origin[2],
        "rotationX": rotation[0],
        "rotationY": rotation[1],
        "rotationZ": rotation[2],
    }


def generate_dummy_element(
    obj,                           # current object
    parent=None,                   # parent Blender object
    armature=None,                 # Blender Armature object (NOT Armature data)
    bone_hierarchy=None,           # map of armature bones => children mesh objects
    parent_matrix_world=None,      # parent matrix world transform
    parent_cube_origin=None,       # parent cube "from" origin (coords in VintageStory space)
    parent_rotation_origin=None,   # parent object rotation origin (coords in VintageStory space)
    **kwargs,
):
    """Parse a "dummy" object. In Blender this is an object with
    "dummy_" prefix, which will be converted into a VS 0-sized cube
    with all faces disabled. This can be used for positioning
    "stepParentName" type shape attachments used in VS.
    """
    if not obj.name.startswith("dummy_"):
        return None
    
    # get dummy object name
    name = obj.name[6:]

    """
    object blender origin and rotation
    -> if this is part of an armature, must get relative
    to parent bone
    """
    origin = np.array(obj.location)
    obj_rotation = obj.rotation_euler

    if armature is not None and obj.parent is not None and obj.parent_bone != "":
        bone_name = obj.parent_bone
        if bone_name in armature.data.bones:
            bone_matrix = armature.data.bones[bone_name].matrix_local
            # print(obj.name, "BONE MATRIX:", bone_matrix)
        mat_loc = parent.matrix_world.inverted_safe() @ obj.matrix_world
        origin, quat, _ = mat_loc.decompose()
        obj_rotation = quat.to_euler("XYZ")
    
    # use step parent bone if available
    if "StepParentName" in obj and len(obj["StepParentName"]) > 0:
        step_parent_name = obj["StepParentName"]
        # "b_[name]" is hard-coded bone prefix, convention for this
        # plugin. if step parent name starts with "b_", remove prefix
        if step_parent_name.startswith("b_"):
            bone_parent_name = step_parent_name[2:]
        else:
            bone_parent_name = step_parent_name
    else:
        step_parent_name = None
        bone_parent_name = None

    if bone_parent_name is not None and bone_parent_name in armature.data.bones and bone_parent_name in bone_hierarchy:
        parent_bone = armature.data.bones[bone_parent_name]
        parent_matrix_world = parent_bone.matrix_local.copy()
        parent_cube_origin = parent_bone.head
        parent_rotation_origin = parent_bone.head
    else:
        if step_parent_name is not None:
            print(f"WARNING: cannot find {obj.name} step parent bone {bone_parent_name}")
    
    # more robust but higher performance cost, just get relative
    # location/rotation from world matrices, required for complex
    # parent hierarchies with armature bones + object-object parenting
    if parent_matrix_world is not None:
        # print(obj.name, "parent_matrix_world", parent_matrix_world)
        mat_local = parent_matrix_world.inverted_safe() @ obj.matrix_world
        origin, quat, _ = mat_local.decompose()
        obj_rotation = quat.to_euler("XYZ")

    # change axis to vintage story y-up axis
    origin = to_y_up(origin)
    rotation = to_vintagestory_rotation(obj_rotation)
    
    # translate to vintage story coord space
    loc = origin - parent_cube_origin + parent_rotation_origin

    # get scale
    scale = to_y_up(obj.scale)

    export_name = name
    if "rename" in obj and isinstance(obj["rename"], str):
        export_name = obj["rename"]
    
    element = {
        "name": export_name,
        "from": loc,
        "to": loc, 
        "rotationOrigin": loc,
        "rotationX": rotation[0],
        "rotationY": rotation[1],
        "rotationZ": rotation[2],
        "faces": {
            "north": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
            "east": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
            "south": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
            "west": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
            "up": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
            "down": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
        },
        "children": [],
    }

    # optional properties:

    # scale
    for (idx, axis) in enumerate(["scaleX", "scaleY", "scaleZ"]):
        if scale[idx] != 1.0:
            element[axis] = scale[idx]

    # step parent
    if "StepParentName" in obj and len(obj["StepParentName"]) > 0:
        element["stepParentName"] = obj["StepParentName"]

    return element


def create_dummy_bone_object(
    name,
    location, # in blender coordinates
    rotation, # in blender coordinates
):
    loc = to_y_up(location)
    rot = to_vintagestory_rotation(rotation)
    return {
        "name": "b_" + name, # append "bone" to animation name so bone does not conflict with main objects
        "from": loc,
        "to": loc, 
        "rotationOrigin": loc,
        "rotationX": rot[0],
        "rotationY": rot[1],
        "rotationZ": rot[2],
        "faces": {
            "north": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
            "east": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
            "south": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
            "west": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
            "up": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
            "down": { "texture": "#null", "uv": [ 0.0, 0.0, 0.0, 0.0 ], "enabled": False },
        },
        "children": [],
    }


def generate_element(
    obj,
    **kwargs,
):
    """Routes to correct element generation function based on object type.
    """
    if obj.type == "EMPTY":
        if obj.name.startswith("attach_"):
            return generate_attach_point(obj, **kwargs)
        elif obj.name.startswith("dummy_"):
            return generate_dummy_element(obj, **kwargs)
    else:
        # TODO: check for quad type
        return generate_mesh_element(obj, **kwargs)


class BoneNode():
    """Contain information on bone hierarchy: bones in armature and
    associated Blender object children of the bone. For exporting to
    Vintage Story, there are no bones, so one of these objects needs to
    act as the bone. The "main" object is the object that will be the
    bone after export.

    Keep main object in index 0 of children, so can easily get non-main
    children using children[1:]
    """
    def __init__(self, name = ""):
        self.name = name                     # string name, for debugging
        self.parent = None                   # bone parent
        self.main = None                     # main object for this bone
        self.position = None                 # position
        self.rotation_residual = None        # rotation after removing 90 deg components
        self.creating_dummy_object = False   # if bone will create a new dummy object in output tree
                                             # used to decide if output bone name in animation keyframes
                                             # should be "{bone.name}" or "b_{bone.name}" (using dummy object)
        
        self.children = [] # blender object children associated with this bone
                           # main object should always be index 0

    def __str__(self):
        return f"""BoneNode {{ name: {self.name},
        main: {self.main},
        parent: {self.parent},
        position: {self.position},
        rotation_residual: {self.rotation_residual},
        children: {self.children} }}"""


def get_bone_relative_matrix(bone, parent):
    """Get bone matrix relative to its parent
    bone: armature data bone
    parent: armature data bone
    """
    if parent is not None:
        return parent.matrix_local.inverted_safe() @ bone.matrix_local.copy()
    else:
        return bone.matrix_local.copy()


def get_bone_hierarchy(armature, root_bones):
    """Create map of armature bone name => BoneNode objects.
    armature: blender armature object
    root_bones: array of root bone objects (data bones)

    "Main" bone determined by following rule in order:
    1. if a child object has same name as the bone, set as main
    2. else, use first object
    """
    # insert empty bone nodes for all armature bones
    bone_hierarchy = {}
    for bone in armature.data.bones:
        bone_hierarchy[bone.name] = BoneNode(name=bone.name)

    for obj in armature.children:
        if obj.parent_bone != "":
            bone_hierarchy[obj.parent_bone].children.append(obj)
    
    # set the "main" object associated with each bone
    for bone_name, node in bone_hierarchy.items():
        for i, obj in enumerate(node.children):
            if obj.name == bone_name:
                node.main = obj
                node.children[0], node.children[i] = node.children[i], node.children[0] # swap so main index 0
                break
        # use first object
        if node.main is None and len(node.children) > 0:
            node.main = node.children[0]

    # go down bone tree and calculate rotations
    def get_bone_rotation(hierarchy, bone, parent):
        if bone.name not in hierarchy:
            return
        
        node = hierarchy[bone.name]

        if parent is not None:
            node.parent = parent.name

        mat_local = get_bone_relative_matrix(bone, bone.parent)
        bone_pos, bone_quat, _ = mat_local.decompose()
        
        node.position = bone_pos
        node.rotation_residual = bone_quat.to_matrix()

        for child in bone.children:
            get_bone_rotation(hierarchy, child, bone)
    
    for root_bone in root_bones:
        get_bone_rotation(bone_hierarchy, root_bone, None)

    return bone_hierarchy


def get_bone_hierarchy_from_armature(
    armature,
):
    """Helper function to get bone hierarchy from Blender objects."""

    # reset all pose bones in armature to bind pose
    for bone in armature.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update() # force update
    
    root_bones = filter_root_objects(armature.data.bones)
    bone_hierarchy = get_bone_hierarchy(armature, root_bones)
    
    return root_bones, bone_hierarchy


def get_armatures_from_objects(
    objects,
    skip_disabled_render=True,
):
    """Helper function to get first armature from Blender objects."""
    armatures = []
    for obj in objects:
        if skip_disabled_render and obj.hide_render:
            continue

        if isinstance(obj.data, bpy.types.Armature):
            armatures.append(obj)
    
    return armatures


# maps a PoseBone rotation mode name to the proper
# action Fcurve property type
ROTATION_MODE_TO_FCURVE_PROPERTY = {
    "QUATERNION": "rotation_quaternion",
    "XYZ": "rotation_euler",
    "XZY": "rotation_euler",
    "YXZ": "rotation_euler",
    "YZX": "rotation_euler",
    "ZXY": "rotation_euler",
    "ZYX": "rotation_euler",
    "AXIS_ANGLE": "rotation_euler", # ??? TODO: handle properly...
}

def save_all_animations(
    obj_armature,
    export_objects=None,
    rotate_shortest_distance=False,
    smart_bake_only=False,
    bake_animations=False,
    bake_step=1,
    tol_loc=1e-4,
    tol_rot_deg=1e-3,
):
    """Export Vintage Story animations from a Blender Armature.

    Key idea: Vintage Story animates *element groups* (the hierarchy nodes, analogous to bones),
    not every child cube. Exporting per-cube tracks causes double-transforms in VSMC (the
    "spiderweb hair" effect). Therefore we export tracks for Armature pose bones.

    - Sample only real keyed frames (plus start/end).
    - Convert pose-bone delta-from-rest into VS axis conventions.
    - Do NOT export scale/stretch channels.
    """
    animations = []
    if obj_armature is None or obj_armature.type != "ARMATURE":
        return animations

    scene = bpy.context.scene
    arm_data = obj_armature.data

    # Save/restore state
    orig_frame = scene.frame_current
    orig_pose_pos = getattr(arm_data, "pose_position", "POSE")
    orig_action = obj_armature.animation_data.action if obj_armature.animation_data else None

    if obj_armature.animation_data is None:
        obj_armature.animation_data_create()

    def near_zero(x, tol):
        try:
            return abs(float(x)) <= tol
        except Exception:
            return True

    def safe_f(x):
        try:
            x = float(x)
            if not math.isfinite(x):
                return 0.0
            return x
        except Exception:
            return 0.0

    def extract_bone_from_datapath(dp: str):
        """Extract the PoseBone name from an Action FCurve data_path.

        Blender versions differ in how they quote bone names in data paths:
        - pose.bones["Bone"]...
        - pose.bones['Bone']...

        If we fail to parse this, animation export silently becomes empty.
        """
        dp = dp or ""
        # Accept both single and double quotes, with optional whitespace.
        m = re.search(r"pose\.bones\[\s*['\"]([^'\"]+)['\"]\s*\]", dp)
        return m.group(1) if m else None

    def bone_export_name(bone):
        """Resolve the name to emit for animation tracks.

        Priority:
        1) bone["vs_export_name"] if present (set during model export to follow user renames)
        2) original VS name (bone["vs_name"]) only if the bone hasn't been renamed since import
        3) current Blender bone.name
        """
        try:
            v = bone.get("vs_export_name", "")
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass

        try:
            vs_name = bone.get("vs_name", "")
            imp = bone.get("vs_import_blender_name", "")
            if isinstance(vs_name, str) and vs_name.strip() and isinstance(imp, str) and imp and bone.name == imp:
                return vs_name
        except Exception:
            pass

        return bone.name

    try:
        arm_data.pose_position = "POSE"
    except Exception:
        pass

    # Mute NLA tracks during sampling so only the current action is evaluated
    ad = obj_armature.animation_data
    orig_nla_mute = []
    try:
        if ad and getattr(ad, "nla_tracks", None):
            for tr in ad.nla_tracks:
                orig_nla_mute.append((tr, tr.mute))
                tr.mute = True
    except Exception:
        orig_nla_mute = []

    def actions_touching_armature():
        out = []
        for a in bpy.data.actions:
            fcurves = getattr(a, "fcurves", None)
            if not fcurves:
                continue
            for fcu in fcurves:
                bname = extract_bone_from_datapath(getattr(fcu, "data_path", ""))
                if bname and arm_data.bones.get(bname) is not None:
                    out.append(a)
                    break
        out.sort(key=lambda x: x.name.lower())
        return out

    def gather_used_actions():
        acts = []
        if ad:
            if ad.action:
                acts.append(ad.action)
            for tr in getattr(ad, "nla_tracks", []) or []:
                for st in getattr(tr, "strips", []) or []:
                    if st.action:
                        acts.append(st.action)
        seen = set()
        out = []
        for a in acts:
            if a and a.name not in seen:
                out.append(a)
                seen.add(a.name)
        return out

    # Export any action that actually drives this armature.
    # "Used" actions (active + NLA strips) are included, but we also include
    # newly-created actions that are not yet on the NLA stack, as long as they
    # have fcurves targeting this armature's pose bones.
    _acts = []
    _acts.extend(gather_used_actions())
    _acts.extend(actions_touching_armature())
    # de-duplicate while preserving order
    actions_to_export = []
    _seen = set()
    for a in _acts:
        if a is None:
            continue
        if a.name in _seen:
            continue
        actions_to_export.append(a)
        _seen.add(a.name)

    for action in actions_to_export:
        # only consider actions that actually touch pose bones
        fcurves = getattr(action, "fcurves", None)
        if not fcurves:
            continue

        keyed_bones = set()
        keyed_frames = set()

        for fcu in fcurves:
            bname = extract_bone_from_datapath(getattr(fcu, "data_path", ""))
            if bname:
                keyed_bones.add(bname)
                for kp in getattr(fcu, "keyframe_points", []):
                    try:
                        keyed_frames.add(int(round(kp.co[0])))
                    except Exception:
                        pass

        if not keyed_bones:
            continue

        # Determine sampling bounds
        fr0, fr1 = action.frame_range
        start = int(round(fr0))
        end = int(round(fr1))
        if end < start:
            start, end = end, start

        # Decide which bones/channels need baking (cubic curves, modifiers, quaternion auth).
        # When smart_bake_only is enabled we will bake only those bones, keeping others sparse.
        step = 1
        try:
            step = int(bake_step)
        except Exception:
            step = 1
        if step <= 0:
            step = 1

        baked_frames = list(range(start, end + 1, step))
        if baked_frames and baked_frames[-1] != end:
            baked_frames.append(end)

        sparse_frames = {start, end}
        for f in keyed_frames:
            if start <= f <= end:
                sparse_frames.add(int(f))
        sparse_frames = sorted(sparse_frames)

        # Per-bone frame emission plan (used for "bake only what needs baking").
        # Keyframes in VS can omit elements, so we can keep non-problematic bones sparse
        # while baking only the ones that need parity with Blender's cubic/quaternion motion.
        frames_for_bone = {}
        if bake_animations:
            frames = baked_frames
            for b in keyed_bones:
                frames_for_bone[b] = set(frames)
        elif smart_bake_only:
            needs_bake = {b: False for b in keyed_bones}
            for fcu in fcurves:
                dp = getattr(fcu, "data_path", "") or ""
                bname = extract_bone_from_datapath(dp)
                if not bname or bname not in needs_bake:
                    continue

                # Any non-linear interpolation, any modifiers, or quaternion-authored curves
                # get baked to avoid engine interpolation mismatches.
                try:
                    if getattr(fcu, "modifiers", None) and len(fcu.modifiers) > 0:
                        needs_bake[bname] = True
                except Exception:
                    pass

                if ".rotation_quaternion" in dp:
                    needs_bake[bname] = True

                try:
                    for kp in getattr(fcu, "keyframe_points", []) or []:
                        interp = getattr(kp, "interpolation", "LINEAR")
                        if interp not in {"LINEAR", "CONSTANT"}:
                            needs_bake[bname] = True
                            break
                except Exception:
                    pass

            all_frames = set()
            for b in keyed_bones:
                use_baked = needs_bake.get(b, False)
                bf = baked_frames if use_baked else sparse_frames
                frames_for_bone[b] = set(bf)
                all_frames.update(frames_for_bone[b])
            frames = sorted(all_frames)
        else:
            frames = sparse_frames
            for b in keyed_bones:
                frames_for_bone[b] = set(frames)

        # Determine quantityframes (preserve on roundtrip if available)
        qf = action.get("vs_quantityframes", None)
        if qf is None:
            quantityframes = int(end - start + 1)
        else:
            try:
                quantityframes = int(qf)
            except Exception:
                quantityframes = int(end - start + 1)
        if quantityframes <= 0:
            quantityframes = int(end - start + 1)

        # Build bone objects and ensure parents are included for local-space conversion
        data_bones = arm_data.bones
        pose_bones = obj_armature.pose.bones

        bones_to_eval = set()
        animated_posebones = []

        for bname in keyed_bones:
            pb = pose_bones.get(bname)
            db = data_bones.get(bname)
            if pb is None or db is None:
                continue
            animated_posebones.append(pb)
            # include parent chain
            cur = db
            while cur is not None:
                bones_to_eval.add(cur.name)
                cur = cur.parent

        if not animated_posebones:
            continue

        # NOTE: Do not subtract a "root baseline" translation.
        # Vintage Story allows (and many vanilla models rely on) constant root
        # offsets/rotations authored in animations. Removing them makes the
        # animation appear static or incorrect in VSMC.

        # Determine which rotation channels the action actually animates for each bone.
        # If a PoseBone is left in QUATERNION mode while keys are authored on
        # rotation_euler (common after import), Blender will ignore those curves
        # when computing pb.matrix, and we export zero rotations.
        bone_rot_mode = {}
        for fcu in fcurves:
            dp = getattr(fcu, "data_path", "") or ""
            bname = extract_bone_from_datapath(dp)
            if not bname:
                continue
            if ".rotation_quaternion" in dp:
                bone_rot_mode[bname] = "QUATERNION"
            elif ".rotation_euler" in dp or ".rotation_axis_angle" in dp:
                # VS expects Euler; use Blender Euler order XZY to match VS axis mapping.
                bone_rot_mode.setdefault(bname, "XZY")

        obj_armature.animation_data.action = action

        # Force pose bones into a compatible rotation mode for evaluation.
        # (Save original modes so we can restore after export.)
        orig_rot_modes = {}
        try:
            for bn in bones_to_eval:
                pb = pose_bones.get(bn)
                if pb is None:
                    continue
                orig_rot_modes[bn] = pb.rotation_mode
                desired = bone_rot_mode.get(bn)
                if desired == "QUATERNION":
                    pb.rotation_mode = "QUATERNION"
                elif desired == "XZY":
                    pb.rotation_mode = "XZY"
                else:
                    # default to XZY so rotation_euler curves are always respected
                    pb.rotation_mode = "XZY"
        except Exception:
            orig_rot_modes = {}

        keyframes_out = []
        any_motion = False
        # Precompute rest matrices in world space
        rest_world = {}
        for bn in bones_to_eval:
            db = data_bones.get(bn)
            if db is None:
                continue
            rest_world[bn] = (obj_armature.matrix_world @ db.matrix_local).copy()

        for frame in frames:
            try:
                scene.frame_set(frame)
            except Exception:
                continue

            # pose matrices in world space for needed bones
            pose_world = {}
            for bn in bones_to_eval:
                pb = pose_bones.get(bn)
                if pb is None:
                    continue
                pose_world[bn] = (obj_armature.matrix_world @ pb.matrix).copy()

            frame_elements = {}

            for pb in animated_posebones:
                bn = pb.name
                # Smart bake: only emit this bone on frames we planned for it.
                try:
                    if bn in frames_for_bone and frame not in frames_for_bone[bn]:
                        continue
                except Exception:
                    pass
                db = data_bones.get(bn)
                if db is None:
                    continue
                parent = db.parent.name if db.parent else None

                # parent-relative local matrices (both rest and pose)
                if parent and parent in rest_world and parent in pose_world:
                    rest_loc = rest_world[parent].inverted_safe() @ rest_world[bn]
                    pose_loc = pose_world[parent].inverted_safe() @ pose_world[bn]
                else:
                    rest_loc = rest_world[bn]
                    pose_loc = pose_world[bn]

                delta = rest_loc.inverted_safe() @ pose_loc
                loc, rot_quat, _scale = delta.decompose()

                # VS axes: offsets X->Blender Y, Y->Blender Z, Z->Blender X
                off_x = safe_f(loc.y)
                off_y = safe_f(loc.z)
                off_z = safe_f(loc.x)

                # Rotation export:
                # - If the action keys rotation_euler, preserve the *exact* Euler values
                #   (including windings > 360 deg) for WYSIWYG in VSMC.
                # - If the action keys rotation_quaternion, fall back to quaternion->Euler.
                if bone_rot_mode.get(bn) == "QUATERNION":
                    r = to_vintagestory_rotation(rot_quat)
                else:
                    # pb.rotation_mode was forced to XZY above when Euler curves exist.
                    e = pb.rotation_euler
                    r = np.array([
                        safe_f(e.y) * RAD_TO_DEG,
                        safe_f(e.z) * RAD_TO_DEG,
                        safe_f(e.x) * RAD_TO_DEG,
                    ])

                # IMPORTANT: do NOT clamp/wrap angles. VS/VSMC should receive the authored
                # values so long rotations don't get "shortest-pathed".
                rot_x = safe_f(r[0])
                rot_y = safe_f(r[1])
                rot_z = safe_f(r[2])

                out_name = bone_export_name(db)
                # snap tiny values to 0
                if near_zero(off_x, tol_loc): off_x = 0.0
                if near_zero(off_y, tol_loc): off_y = 0.0
                if near_zero(off_z, tol_loc): off_z = 0.0
                if near_zero(rot_x, tol_rot_deg): rot_x = 0.0
                if near_zero(rot_y, tol_rot_deg): rot_y = 0.0
                if near_zero(rot_z, tol_rot_deg): rot_z = 0.0

                # Track if there is any non-zero motion anywhere (for skipping empty actions)
                if off_x != 0.0 or off_y != 0.0 or off_z != 0.0 or rot_x != 0.0 or rot_y != 0.0 or rot_z != 0.0:
                    any_motion = True

                # Output schema (no stretch). Always include offsets so downstream
                # tools don't have to guess when a channel is "missing" vs "0".
                el = {
                    "offsetX": off_x,
                    "offsetY": off_y,
                    "offsetZ": off_z,
                    "rotationX": rot_x,
                    "rotationY": rot_y,
                    "rotationZ": rot_z,
                }

                if rotate_shortest_distance:
                    el["rotShortestDistanceX"] = True
                    el["rotShortestDistanceY"] = True
                    el["rotShortestDistanceZ"] = True

                frame_elements[out_name] = el

            # Don't emit empty keyframes
            if frame_elements:
                keyframes_out.append({
                    "frame": int(frame - start),
                    "elements": frame_elements
                })

        if not any_motion or not keyframes_out:
            continue

        anim_name = action.get("vs_anim_name", action.name)
        anim_code = action.get("vs_anim_code", action.name)

        animations.append({
            "name": anim_name,
            "code": anim_code,
            "quantityframes": int(quantityframes),
            "onActivityStopped": action.get("on_activity_stopped", "PlayTillEnd"),
            "onAnimationEnd": action.get("on_animation_end", "EaseOut"),
            "keyframes": keyframes_out,
        })

        # Restore original pose-bone rotation modes for this action
        try:
            for bn, rm in orig_rot_modes.items():
                pb = pose_bones.get(bn)
                if pb is not None:
                    pb.rotation_mode = rm
        except Exception:
            pass

    # restore state
    try:
        scene.frame_set(orig_frame)
    except Exception:
        pass
    try:
        arm_data.pose_position = orig_pose_pos
    except Exception:
        pass

    if obj_armature.animation_data:
        obj_armature.animation_data.action = orig_action

    # restore NLA mute states
    try:
        for tr, m in orig_nla_mute:
            tr.mute = m
    except Exception:
        pass

    return animations


def save_objects_by_armature(
    bone,
    bone_hierarchy,
    skip_disabled_render=True,
    armature=None,
    groups=None,
    model_colors=None,
    model_textures=None,
    parent_matrix_world=None,
    parent_cube_origin=np.array([0., 0., 0.]),
    parent_rotation_origin=np.array([0., 0., 0.]),
    export_uvs=True,               # export uvs
    export_generated_texture=True, # export generated color texture
    texture_size_x_override=None,  # texture size overrides
    texture_size_y_override=None,  # texture size overrides
    use_main_object_as_bone=False,  # allow using main object as bone
):
    """Recursively save object children of a bone to a parent
    bone object
    """
    bone_element = None
    
    if bone.name in bone_hierarchy:
        # print(bone.name, bone, bone.children, bone_hierarchy[bone.name].main)

        bone_node = bone_hierarchy[bone.name]
        bone_object = bone_node.main
        bone_children = bone_node.children
        mat_world = bone.matrix_local.copy()

        # main bone object world transform == bone transform, can simply use 
        # as the bone
        if use_main_object_as_bone and bone_object is not None and matrix_roughly_equal(bone.matrix_local, bone_object.matrix_world):
            # print(bone.name)
            # print("bone.matrix_local:", bone.matrix_local)
            # print("object.world_matrix:", bone_object.matrix_world)
            # print("MATRIX EQUAL")
            bone_element = generate_mesh_element(
                bone_object,
                skip_disabled_render=skip_disabled_render,
                parent=None,
                armature=None,
                bone_hierarchy=None,
                groups=groups,
                model_colors=model_colors,
                model_textures=model_textures,
                parent_matrix_world=parent_matrix_world,
                parent_cube_origin=parent_cube_origin,
                parent_rotation_origin=parent_rotation_origin,
                export_uvs=export_uvs,
                export_generated_texture=export_generated_texture,
                texture_size_x_override=texture_size_x_override,
                texture_size_y_override=texture_size_y_override,
            )
            if len(bone_children) > 1:
                bone_children = bone_children[1:]
            else:
                bone_children = []

        # main object could not be used, insert a dummy object with bone transform
        if bone_element is None:
            bone_hierarchy[bone.name].creating_dummy_object = True

            if parent_matrix_world is not None:
                mat_local = parent_matrix_world.inverted_safe() @ mat_world
            else:
                mat_local = mat_world
            bone_loc, quat, _ = mat_local.decompose()
            
            bone_element = create_dummy_bone_object(bone.name, bone_loc, bone_node.rotation_residual)
        
            cube_origin = bone.head
            rotation_origin = bone.head
        else:
            cube_origin = bone_element["from"]
            rotation_origin = bone_element["rotationOrigin"]

        # attachment points (will only add entry to bone if exists in mode)
        attachpoints = []
        
        for obj in bone_children:
            if skip_disabled_render and obj.hide_render:
                continue
            
            child_element = generate_element(
                obj,
                skip_disabled_render=skip_disabled_render,
                parent=None,
                armature=None,
                bone_hierarchy=None,
                is_bone_child=True,
                groups=groups,
                model_colors=model_colors,
                model_textures=model_textures,
                parent_matrix_world=mat_world,
                parent_cube_origin=cube_origin,
                parent_rotation_origin=rotation_origin,
                export_uvs=export_uvs,
                export_generated_texture=export_generated_texture,
                texture_size_x_override=texture_size_x_override,
                texture_size_y_override=texture_size_y_override,
            )
            if child_element is not None:
                if obj.name.startswith("attach_"):
                    attachpoints.append(child_element)
                else:
                    bone_element["children"].append(child_element)
        
        if len(attachpoints) > 0:
            bone_element["attachmentpoints"] = attachpoints

        # recursively add child bones
        for child_bone in bone.children:
            child_element = save_objects_by_armature(
                child_bone,
                bone_hierarchy,
                skip_disabled_render=skip_disabled_render,
                armature=armature,
                groups=groups,
                model_colors=model_colors,
                model_textures=model_textures,
                parent_matrix_world=mat_world,
                parent_cube_origin=cube_origin,
                parent_rotation_origin=rotation_origin,
                export_uvs=export_uvs,
                export_generated_texture=export_generated_texture,
                texture_size_x_override=texture_size_x_override,
                texture_size_y_override=texture_size_y_override,
                use_main_object_as_bone=False,
            )
            if child_element is not None:
                bone_element["children"].append(child_element)
    
    return bone_element


def save_objects(
    filepath="",
    objects=[],
    skip_disabled_render=True,
    translate_origin=None,
    texture_folder="",
    generate_texture=True,
    use_only_exported_object_colors=False,
    color_texture_filename="",
    texture_size_x_override=None, # override texture image size x
    texture_size_y_override=None, # override texture image size y
    export_uvs=True,
    skip_texture_export=False,
    minify=False,
    decimal_precision=-1,
    export_armature=False,
    export_animations=True,
    generate_animations_file=False,
    bake_animations=False,
    bake_step=1,
    smart_bake_only=False,
    sanitize_keyframes=False,
    sanitize_epsilon=1e-4,
    sanitize_rot_clamp_deg=7200.0,
    use_main_object_as_bone=False,
    use_step_parent=True,
    repair_hierarchy=False,
    rotate_shortest_distance=False,
    animation_version_0=False,
    logger=None,
    **kwargs
) -> tuple[set[str], set[str], str]:
    """Main exporter function. Parses Blender objects into VintageStory
    cuboid format, uvs, and handles texture read and generation.
    Will save .json file to output path because this needs to save both
    json containing model structure and textures generated during export.
    
    TODO: If need arises, make this return (model, textures) and separate
    wrapper func that saves to files.

    Parameters
    ----------
    filepath : str
        Output file path name. Returns error if file is "" or None.
    objects : list of bpy.types.Object
        Iterable collection of Blender objects. Returns warning if empty.
    skip_disabled_render : bool
        Skip objects with disable render (hide_render) flag
    translate_origin : list[float]
        New origin to shift, None for no shift, [x, y, z] list in Blender
        coords to apply shift
    texture_folder : str
        Output texture subpath, for typical "item/texture_name" the texture
        folder would be "item".
    generate_texture : bool
        Generate texture from solid material colors. By default, creates
        a color texture from all materials in file (so all groups of objects
        can share the same texture file).
    use_only_exported_object_colors : bool
        Generate texture colors from only exported objects instead of default
        using all file materials.
    color_texture_filename : str
        Name of exported generated color texture file.
    export_uvs : bool
        Export object uvs (required for textures to render properly).
    skip_texture_export : bool
        Skip exporting texture paths and sizes.
    minify : bool
        Minimize output file size (write into single line, remove spaces, ...)
    decimal_precision : int
        Number of digits after decimal to keep in numbers. Set to -1 to
        disable.
    export_armature : bool
        Export by bones, makes custom hierarchy based on bone tree and
        attaches generated elements to their bone parent.
    export_animations : bool
        Export bones and animation actions.
    generate_animations_file : bool
        Generate separate animations file, e.g. "model.json" and
        "model_animations.json".
    use_main_object_as_bone : bool
        Detect if bone and same named parent object have same transform
        and collapse into a single object (to reduce element count).
    use_step_parent : bool
        Transform root elements relative to their step parent element, so
        elements are correctly attached in game.
        TODO: make this flag actually work, right now automatically enabled
    repair_hierarchy : bool
        Attempt to reconstruct a parent/child hierarchy for VS elements when the
        original VS metadata (obj['vs_parent']) is missing or broken.
        Useful for edge cases where objects were unparented or bone-parented
        and animation inheritance breaks after export.
    rotate_shortest_distance : bool
        Use shortest distance rotation interpolation for animations.
        This sets the "rotShortestDistance_" flags in the output keyframes.
    animation_version_0 : bool
        Use VintageStory animation version 0 format incompatible with Blender,
        which uses which uses R*T*v format and additive bone euler rotations.
    logger : Optional bpy.types.Operator
        Optional blender object that can use `.report(type, message)` to
        report warning or error messages to the user.
    
    Returns
    -------
    Tuple : (result, message_type, message)
        - result: set of result types, e.g. {"FINISHED"} or {"CANCELLED"}
        - message_type: set of message types, e.g. {"WARNING"} or {"INFO"}
        - message: string message, for use in `op.report(type, message)`.
    """
    # Sync datablock names to object names right before export (helps VS JSON / VSMC and avoids stale Cube.### names)
    try:
        import bpy
        for _o in bpy.data.objects:
            try:
                if getattr(_o, "data", None) is not None and getattr(_o.data, "users", 0) == 1:
                    _o.data.name = _o.name
            except Exception:
                pass
    except Exception:
        pass

    if filepath == "" or filepath is None:
        return {"CANCELLED"}, {"ERROR"}, "No output file path specified"

    # ------------------------------------------------------------
    # Pre-export cleanup/sync
    # - Ensure datablock names match object names (fixes Cube.### leftovers)
    # - Must run as late as possible, right before we read scene data
    # ------------------------------------------------------------
    try:
        import bpy
        for obj in bpy.data.objects:
            data = getattr(obj, "data", None)
            if data is not None and getattr(data, "users", 0) == 1:
                data.name = obj.name
    except Exception:
        pass

    # Vintage Story JSON does not support exporting Blender armatures/bones as model elements.
    # Bones are an authoring rig only; we always export the VS element hierarchy.
    export_armature = False
    use_main_object_as_bone = False
    animation_version_0 = False
    
    # export status, may be modified by function if errors occur
    status = {"INFO"}

    # Resolve texture size for UV export.
    # Prefer explicit overrides, otherwise use the size declared in the imported VS JSON (stored on the scene),
    # and finally fall back to 16.
    scn = bpy.context.scene
    declared_w = scn.get("vs_textureWidth", None)
    declared_h = scn.get("vs_textureHeight", None)

    # Only force overrides if we have declared sizes and the user didn't explicitly override.
    if texture_size_x_override is None and declared_w is not None:
        try:
            texture_size_x_override = float(declared_w)
        except Exception:
            pass
    if texture_size_y_override is None and declared_h is not None:
        try:
            texture_size_y_override = float(declared_h)
        except Exception:
            pass

    tex_w_header = 16 if texture_size_x_override is None else texture_size_x_override
    tex_h_header = 16 if texture_size_y_override is None else texture_size_y_override

    # output json model stub
    model_json = {
        "textureWidth": int(tex_w_header) if abs(tex_w_header - int(tex_w_header)) < 1e-6 else tex_w_header,
        "textureHeight": int(tex_h_header) if abs(tex_h_header - int(tex_h_header)) < 1e-6 else tex_h_header,
        "textures": {},
        "textureSizes": {},
    }

    # elements at top of hierarchy
    root_elements = []

    # object collections to be exported
    groups = {}
    
    # all material colors tuples from all object faces
    if use_only_exported_object_colors:
        model_colors = set()
    else:
        model_colors = None
    
    # all object face material texture or color info
    # material.name => FaceMaterial
    model_textures: dict[str, FaceMaterial] = {}
    
    # first pass: check if parsing an armature
    # NOTE: VS JSON does not support exporting Blender armatures/bones as elements.
    # Always export element objects (cuboids) only.
    export_armature = False

    armature = None
    root_bones = []
    bone_hierarchy = None
    export_objects = [o for o in objects if getattr(o, "type", None) != "ARMATURE"]
    # Keep reference to armature for animation baking (not exported as VS elements)
    _all_objs_for_armature = list(objects)


    # check if any objects have step parent, this means root objects
    # need armature bone offsets
    has_step_parent = False
    for obj in export_objects:
        if "StepParentName" in obj and len(obj["StepParentName"]) > 0:
            has_step_parent = True
            break
    
    # check if any objects in scene are armatures
    scene_armatures = get_armatures_from_objects(bpy.data.objects, skip_disabled_render=skip_disabled_render)

    # ---------------------------------------------------------------------
    # Animation export needs an armature even when we are not exporting the
    # model hierarchy "by armature" (we always export VS elements/objects).
    #
    # Prior bug: if no StepParentName was used and export_armature was forced
    # off, we never resolved an armature, so animation export silently
    # produced an empty "animations" list.
    # ---------------------------------------------------------------------
    if export_animations and armature is None:
        candidates = {}

        def _bump(a, w=1):
            if a is None or getattr(a, "type", None) != "ARMATURE":
                return
            candidates[a] = candidates.get(a, 0) + w

        # Prefer an explicitly included armature, else armatures that actually
        # drive the exported objects (via parenting/modifiers), else fall back
        # to the single armature in the scene.
        for o in _all_objs_for_armature:
            if getattr(o, "type", None) == "ARMATURE":
                _bump(o, 100)
                continue

            # If the object is influenced by an armature (armature modifier or
            # parent chain), this will generally find it.
            try:
                _bump(o.find_armature(), 10)
            except Exception:
                pass

            # Bone-parented objects have parent == armature
            try:
                if getattr(o, "parent", None) is not None and getattr(o.parent, "type", None) == "ARMATURE":
                    _bump(o.parent, 10)
            except Exception:
                pass

        if candidates:
            armature = max(candidates.items(), key=lambda kv: kv[1])[0]
            try:
                root_bones, bone_hierarchy = get_bone_hierarchy_from_armature(armature)
            except Exception:
                root_bones, bone_hierarchy = [], None
        elif len(scene_armatures) == 1:
            armature = scene_armatures[0]
            try:
                root_bones, bone_hierarchy = get_bone_hierarchy_from_armature(armature)
            except Exception:
                root_bones, bone_hierarchy = [], None

    # case 1. exporting objects with step parent property, need armature
    # for performing transform offset calculations
    
    # try to use first armature in scene
    if has_step_parent:
        if len(scene_armatures) > 0:
            # use first armature in scene if did not find armature from before
            armature = scene_armatures[0]
            root_bones, bone_hierarchy = get_bone_hierarchy_from_armature(armature)
        else:
            status = {"WARNING"}
            logger.report({"WARNING"}, "No armature found for step parent")
    
    # case 2. exporting selection of objects, get armature within objects
    # selection (if any)
    elif export_armature and len(scene_armatures) > 0:
        # use armature only from export objects
        exported_armatures = get_armatures_from_objects(objects, skip_disabled_render=skip_disabled_render)
        if len(exported_armatures) > 0:
            armature = exported_armatures[0]
            root_bones, bone_hierarchy = get_bone_hierarchy_from_armature(armature)
        else:
            return {"CANCELLED"}, {"WARNING"}, "No armature found for export by armature"

        if export_armature and armature is not None:
            # do export starting from root bone children
            export_objects = []
            for bone in root_bones:
                export_objects.append(bone_hierarchy[bone.name].main)
    
    # ---------------------------------------------------------------------
    # Bone -> export-name mapping
    #
    # Imported VS models use bones for animation authoring, but VS JSON tracks
    # animate named nodes. If the user renames cubes/objects in the Outliner,
    # we want both the exported *elements* and exported *animations* to use the
    # new names.
    #
    # We do this by writing a transient custom property on data bones:
    #     bone["vs_export_name"]
    # Animation export will prefer this value.
    # ---------------------------------------------------------------------
    if armature is not None and bone_hierarchy is not None:
        def _vs_object_export_name(o):
            try:
                if "rename" in o and isinstance(o["rename"], str) and o["rename"].strip():
                    return o["rename"]
            except Exception:
                pass
            try:
                vs_name = o.get("vs_name") if hasattr(o, "get") else None
                imp = o.get("vs_import_blender_name") if hasattr(o, "get") else None
                if isinstance(vs_name, str) and isinstance(imp, str) and o.name == imp:
                    return vs_name
            except Exception:
                pass
            return getattr(o, "name", "") or ""

        for bname, bnode in getattr(bone_hierarchy, "items", lambda: [])():
            try:
                bone = armature.data.bones.get(bname)
                if bone is None:
                    continue
                main_obj = getattr(bnode, "main", None)
                if main_obj is None:
                    continue
                bone["vs_export_name"] = _vs_object_export_name(main_obj)
            except Exception:
                pass

    # =========================================================================
    # export by armature
    # =========================================================================
    if export_armature and armature is not None:
        for root_bone in root_bones:
            element = save_objects_by_armature(
                bone=root_bone,
                bone_hierarchy=bone_hierarchy,
                skip_disabled_render=skip_disabled_render,
                armature=armature,
                groups=groups,
                model_colors=model_colors,
                model_textures=model_textures,
                export_uvs=export_uvs,
                export_generated_texture=generate_texture,
                texture_size_x_override=texture_size_x_override,
                texture_size_y_override=texture_size_y_override,
                use_main_object_as_bone=False,
            )
            if element is not None:
                root_elements.append(element)

    else:

        # ---------------------------------------------------------------------
        # Geometry export must be in REST pose; to do that reliably we need the
        # driving armature even if the user disabled animation export. Imported
        # VS models are typically bone-parented, so without an armature the
        # current pose/frame can accidentally bake into the exported geometry.
        # ---------------------------------------------------------------------
        if armature is None:
            exported_armatures = []
            try:
                exported_armatures = get_armatures_from_objects(_all_objs_for_armature, skip_disabled_render=skip_disabled_render)
            except Exception:
                exported_armatures = []
            if len(exported_armatures) > 0:
                armature = exported_armatures[0]
            else:
                # Fallback: look for parent armature
                for _o in _all_objs_for_armature:
                    if getattr(_o, "parent", None) is not None and getattr(_o.parent, "type", None) == "ARMATURE":
                        armature = _o.parent
                        break
                if armature is None:
                    scene_armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
                    if len(scene_armatures) > 0:
                        armature = scene_armatures[0]

        # ---------------------------------------------------------------------
        # Geometry export must be in REST pose with no action, otherwise the
        # current pose/frame bakes into the model (wrong).
        # ---------------------------------------------------------------------
        _arm_restore = None
        if armature is not None:
            try:
                armature.animation_data_create()
                _arm_restore = (armature.data.pose_position, armature.animation_data.action, bpy.context.scene.frame_current)
                armature.data.pose_position = "REST"
                armature.animation_data.action = None
                bpy.context.view_layer.update()
            except Exception:
                _arm_restore = None
        # =====================================================================
        # normal export geometry tree (VS hierarchy)
        # =====================================================================
        # Imported rigs are typically bone-parented, which destroys Blender object
        # parenting. Preserve the original VS "parented to" chain using obj['vs_parent'].
        def _export_name(o):
            if o is None:
                return ""
            try:
                if "rename" in o and isinstance(o["rename"], str) and o["rename"].strip():
                    return o["rename"]
            except Exception:
                pass
            try:
                vs_name = o.get("vs_name") if hasattr(o, "get") else None
                imp = o.get("vs_import_blender_name") if hasattr(o, "get") else None
                if isinstance(vs_name, str) and vs_name.strip() and isinstance(imp, str) and imp and o.name == imp:
                    return vs_name
            except Exception:
                pass
            return o.name

        def _is_elem(o):
            if o is None:
                return False
            if o.type == "MESH":
                return True
            if o.type == "EMPTY" and (o.name.startswith("dummy_")):
                return True
            return False

        def _is_attach(o):
            return o is not None and o.type == "EMPTY" and o.name.startswith("attach_")

        # elements and attachpoints in export set
        elem_objs = [o for o in export_objects if _is_elem(o)]
        attach_objs = [o for o in export_objects if _is_attach(o)]

        name_to_obj = {}
        for o in elem_objs:
            # Map multiple aliases so VS hierarchy links keep working even after renames.
            # - current Blender name
            # - original VS name (vs_name)
            # - Blender name at import time (vs_import_blender_name)
            # - explicit override (rename)
            keys = []

            try:
                keys.append(o.name)
            except Exception:
                pass

            try:
                if hasattr(o, "get"):
                    vsn = o.get("vs_name")
                    if isinstance(vsn, str) and vsn:
                        keys.append(vsn)
                    impn = o.get("vs_import_blender_name")
                    if isinstance(impn, str) and impn:
                        keys.append(impn)
                    rn = o.get("rename")
                    if isinstance(rn, str) and rn:
                        keys.append(rn)
            except Exception:
                pass

            # export name (handles whitespace preservation when unchanged)
            try:
                keys.append(_export_name(o))
            except Exception:
                pass

            for k in keys:
                if isinstance(k, str) and k:
                    name_to_obj[k] = o
                    name_to_obj[k.strip()] = o

        # parent map (VS hierarchy)
        # Primary source: obj['vs_parent'] (set on import) so we can preserve VS hierarchy
        # even when objects were bone-parented (which destroys Blender object parenting).
        #
        # Optional repair: when the user has cleared custom props / parenting, we can
        # reconstruct a reasonable hierarchy using:
        #   1) Blender object parenting (if present)
        #   2) Armature bone parent chain (if bone-parented)
        #   3) Spatial inference (smallest enclosing/closest element by AABB)
        if repair_hierarchy and armature is not None and bone_hierarchy is None:
            try:
                _, bone_hierarchy = get_bone_hierarchy_from_armature(armature)
            except Exception:
                bone_hierarchy = None

        def _infer_parent_from_bone(obj):
            """If obj is bone-parented, infer its VS parent from the bone parent chain."""
            try:
                if armature is None or bone_hierarchy is None:
                    return None
                if getattr(obj, "parent", None) is None or getattr(obj.parent, "type", None) != "ARMATURE":
                    return None
                bone_name = getattr(obj, "parent_bone", "") or ""
                if not bone_name:
                    return None
                bone = armature.data.bones.get(bone_name)
                if bone is None or bone.parent is None:
                    return None
                pb = bone.parent
                if pb.name not in bone_hierarchy:
                    return None
                p_obj = getattr(bone_hierarchy[pb.name], "main", None)
                return p_obj if p_obj in elem_objs else None
            except Exception:
                return None

        def _spatial_parent_map(objs):
            """Infer a hierarchy when all metadata is missing.

            Heuristic: choose the parent whose world-space AABB contains (or is closest to)
            the child's origin, preferring the smallest such parent.
            """
            from mathutils import Vector
            # Precompute world AABBs + origins
            aabb = {}
            origin = {}
            vol = {}
            for o in objs:
                try:
                    pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
                    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
                    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
                    aabb[o] = (mn, mx)
                    origin[o] = o.matrix_world.translation.copy()
                    d = mx - mn
                    vol[o] = abs(d.x * d.y * d.z)
                except Exception:
                    aabb[o] = (Vector((0, 0, 0)), Vector((0, 0, 0)))
                    origin[o] = Vector((0, 0, 0))
                    vol[o] = 0.0

            def _dist_point_aabb(p, mn, mx):
                dx = max(mn.x - p.x, 0.0, p.x - mx.x)
                dy = max(mn.y - p.y, 0.0, p.y - mx.y)
                dz = max(mn.z - p.z, 0.0, p.z - mx.z)
                return math.sqrt(dx*dx + dy*dy + dz*dz)

            mapping = {}
            for child in objs:
                best_parent = None
                best_key = None
                child_v = vol.get(child, 0.0)
                p0 = origin.get(child)
                if p0 is None:
                    mapping[child] = None
                    continue
                for cand in objs:
                    if cand is child:
                        continue
                    # Prevent tiny mutual loops: parent should be meaningfully larger.
                    if vol.get(cand, 0.0) < child_v * 1.01:
                        continue
                    mn, mx = aabb[cand]
                    d = _dist_point_aabb(p0, mn, mx)
                    key = (d, vol.get(cand, 0.0))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_parent = cand
                # If everything fails, root.
                mapping[child] = best_parent
            return mapping

        parent_map = {}
        missing_vs_parent = 0
        for o in elem_objs:
            p = None
            try:
                vs_parent = o.get("vs_parent")
            except Exception:
                vs_parent = None

            if isinstance(vs_parent, str) and vs_parent:
                p = name_to_obj.get(vs_parent) or name_to_obj.get(vs_parent.strip())
            else:
                missing_vs_parent += 1

            # Optional repairs for missing/broken vs_parent
            if repair_hierarchy and p is None:
                # 1) Blender object parenting (ignore Armature parent here)
                try:
                    if getattr(o, "parent", None) is not None and getattr(o.parent, "type", None) != "ARMATURE" and o.parent in elem_objs:
                        p = o.parent
                except Exception:
                    pass

            if repair_hierarchy and p is None:
                # 2) Bone parent chain
                p = _infer_parent_from_bone(o)

            parent_map[o] = p

        # 3) Spatial inference as a last resort when the hierarchy is completely missing.
        if repair_hierarchy and len(elem_objs) > 1 and missing_vs_parent == len(elem_objs):
            try:
                spatial = _spatial_parent_map(elem_objs)
                # only fill where we still have no parent
                for o in elem_objs:
                    if parent_map.get(o) is None:
                        parent_map[o] = spatial.get(o)
            except Exception:
                pass

        # attachpoints parent map: prefer stored vs_parent, else use Blender parent if it is a VS element
        attach_parent = {}
        for a in attach_objs:
            p = None
            vs_parent = a.get("vs_parent")
            if isinstance(vs_parent, str) and vs_parent:
                p = name_to_obj.get(vs_parent) or name_to_obj.get(vs_parent.strip())
            if p is None and a.parent in elem_objs:
                p = a.parent
            attach_parent[a] = p

        # children maps
        children_map = {o: [] for o in elem_objs}
        for o, p in parent_map.items():
            if p is not None and p in children_map:
                children_map[p].append(o)

        attach_map = {o: [] for o in elem_objs}
        for a, p in attach_parent.items():
            if p is not None and p in attach_map:
                attach_map[p].append(a)

        roots = [o for o in elem_objs if parent_map.get(o) is None]

        def export_tree(o, parent_obj, parent_matrix_world, parent_cube_origin, parent_rotation_origin):
            if skip_disabled_render and o.hide_render:
                return None

            element = generate_element(
                o,
                skip_disabled_render=skip_disabled_render,
                parent=parent_obj,
                armature=None,
                bone_hierarchy=None,
                groups=groups,
                model_colors=model_colors,
                model_textures=model_textures,
                parent_matrix_world=parent_matrix_world,
                parent_cube_origin=parent_cube_origin,
                parent_rotation_origin=parent_rotation_origin,
                export_uvs=export_uvs,
                export_generated_texture=generate_texture,
                texture_size_x_override=texture_size_x_override,
                texture_size_y_override=texture_size_y_override,
                skip_children_recurse=True,
            )
            if element is None:
                return None

            # rebuild children explicitly from VS hierarchy
            element_children = []
            for ch in children_map.get(o, []):
                ce = export_tree(
                    ch,
                    o,
                    o.matrix_world.copy(),
                    element.get("from", np.array([0., 0., 0.])),
                    element.get("rotationOrigin", np.array([0., 0., 0.])),
                )
                if ce is not None:
                    element_children.append(ce)
            element["children"] = element_children

            # attachpoints (kept separate in VS)
            ap_elems = []
            for ap in attach_map.get(o, []):
                if skip_disabled_render and ap.hide_render:
                    continue
                ae = generate_element(
                    ap,
                    skip_disabled_render=skip_disabled_render,
                    parent=o,
                    armature=None,
                    bone_hierarchy=None,
                    groups=groups,
                    model_colors=model_colors,
                    model_textures=model_textures,
                    parent_matrix_world=o.matrix_world.copy(),
                    parent_cube_origin=element.get("from", np.array([0., 0., 0.])),
                    parent_rotation_origin=element.get("rotationOrigin", np.array([0., 0., 0.])),
                    export_uvs=export_uvs,
                    export_generated_texture=generate_texture,
                    texture_size_x_override=texture_size_x_override,
                    texture_size_y_override=texture_size_y_override,
                    skip_children_recurse=True,
                )
                if ae is not None:
                    ap_elems.append(ae)
            if len(ap_elems) > 0:
                element["attachmentpoints"] = ap_elems

            return element

        for obj in roots:
            element = export_tree(
                obj,
                None,
                None,
                np.array([0., 0., 0.]),
                np.array([0., 0., 0.]),
            )
            if element is not None:
                # skip root attach points
                if isinstance(obj.name, str) and obj.name.startswith("attach_"):
                    continue
                root_elements.append(element)
    # =========================================================================
    # Color texture image generation
    # =========================================================================
    if generate_texture:
        # default, get colors from all materials in file
        if model_colors is None:
            model_colors = set()
            for mat in bpy.data.materials:
                color = get_material_color(mat)
                if isinstance(color, tuple):
                    model_colors.add(color)
        
        tex_pixels, tex_size, color_tex_uv_map, default_color_uv = create_color_texture(model_colors)

        # texture output filepaths
        if color_texture_filename == "":
            current_dir = os.path.dirname(filepath)
            filepath_name = os.path.splitext(os.path.basename(filepath))[0]
            texture_save_path = os.path.join(current_dir, filepath_name + ".png")
            texture_model_path = posixpath.join(texture_folder, filepath_name)
        else:
            current_dir = os.path.dirname(filepath)
            texture_save_path = os.path.join(current_dir, color_texture_filename + ".png")
            texture_model_path = posixpath.join(texture_folder, color_texture_filename)
        
        # create + save texture
        tex = bpy.data.images.new("tex_colors", alpha=True, width=tex_size, height=tex_size)
        tex.file_format = "PNG"
        tex.pixels = tex_pixels
        tex.filepath_raw = texture_save_path
        tex.save()

        # write texture info to output model
        model_json["textureSizes"]["0"] = [tex_size, tex_size]
        model_json["textures"]["0"] = texture_model_path
    else:
        color_tex_uv_map = None
        default_color_uv = None
        
        # if not generating texture, just write texture path to json file
        # TODO: scan materials for textures, then update output size
        if color_texture_filename != "":
            model_json["textureSizes"]["0"] = [16, 16]
            model_json["textures"]["0"] = posixpath.join(texture_folder, color_texture_filename)
    
    # =========================================================================
    # Texture paths and sizes
    # convert blender path names "//folder\tex.png" -> "{texture_folder}/tex"
    # add textures indices for textures, and create face mappings like "#1"
    # NOTE: #0 id reserved for generated color texture
    # =========================================================================
    if not skip_texture_export:
        texture_refs = {} # maps blender path name -> #n identifiers
        for material in model_textures.values():
            texture_filename = material.texture_path
            if texture_filename[0:2] == "//":
                texture_filename = texture_filename[2:]
            texture_filename = texture_filename.replace("\\", "/")
            texture_filename = os.path.split(texture_filename)[1]
            texture_filename = os.path.splitext(texture_filename)[0]
            
            texture_refs[material.name] = "#" + material.name
            model_json["textures"][material.name] = posixpath.join(texture_folder, texture_filename)
            
            tex_size_x = material.texture_size[0] if texture_size_x_override is None else texture_size_x_override
            tex_size_y = material.texture_size[1] if texture_size_y_override is None else texture_size_y_override

            model_json["textureSizes"][material.name] = [tex_size_x, tex_size_y]

    # =========================================================================
    # Origin shift post-processing
    # =========================================================================
    if translate_origin is not None:
        translate_origin = to_y_up(translate_origin)
        for element in root_elements:        
            # re-centering
            element["to"] = translate_origin + element["to"]
            element["from"] = translate_origin + element["from"]
            element["rotationOrigin"] = translate_origin + element["rotationOrigin"]  
    
    # =========================================================================
    # All object post processing
    # 1. convert numpy to python list
    # 2. map solid color face uv -> location in generated texture
    # 3. disable faces with user specified disable texture
    # =========================================================================
    def final_element_processing(element):
        # convert numpy to python list
        element["to"] = element["to"].tolist()
        element["from"] = element["from"].tolist()
        element["rotationOrigin"] = element["rotationOrigin"].tolist()

        faces = element["faces"]
        for d, f in faces.items():
            if isinstance(f, FaceMaterial): # face is mapped to a solid color
                if color_tex_uv_map is not None:
                    color_uv = color_tex_uv_map[f.color] if f.color in color_tex_uv_map else default_color_uv
                else:
                    color_uv = [0, 0, 16, 16]
                faces[d] = {
                    "uv": color_uv,
                    "texture": "#0",
                }
                # glow
                if f.glow > 0:
                    faces[d]["glow"] = f.glow
            
        for child in element["children"]:
            final_element_processing(child)
        
    for element in root_elements:
        final_element_processing(element)
    
    # =========================================================================
    # convert groups
    # =========================================================================
    groups_export = []
    for g in groups:
        groups_export.append({
            "name": g,
            "origin": [0, 0, 0],
            "children": groups[g],
        })

    # save
    model_json["elements"] = root_elements
    model_json["groups"] = groups_export

    # =========================================================================
    # export animations
    # =========================================================================
    if export_animations:
        animations = save_all_animations(
            armature,
            export_objects=objects,
            rotate_shortest_distance=rotate_shortest_distance,
            smart_bake_only=smart_bake_only,
            bake_animations=bake_animations,
            bake_step=bake_step,
        )
        if len(animations) > 0:
            model_json["animations"] = animations

    # =========================================================================
    # Keyframe sanitizer (opt-in)
    # - unwrap Euler angles per channel
    # - snap near-zero noise to 0
    # - clamp extreme values and report
    # =========================================================================
    def _isfinite_num(v):
        try:
            fv = float(v)
            return math.isfinite(fv)
        except Exception:
            return False

    def _sanitize_animations(anims, eps=1e-4, rot_clamp=7200.0):
        issues = []
        if not anims:
            return issues
        rot_clamp = abs(float(rot_clamp)) if rot_clamp is not None else 0.0
        for anim in anims:
            prev = {}  # element -> (rx, ry, rz)
            # Iterate keyframes in order; for each element present, unwrap against last seen.
            kfs = anim.get("keyframes", []) or []
            kfs.sort(key=lambda k: int(k.get("frame", 0)))
            for kf in kfs:
                elems = kf.get("elements", {}) or {}
                for ename, tr in elems.items():
                    if not isinstance(tr, dict):
                        continue

                    # Snap small noise
                    for key in ("offsetX", "offsetY", "offsetZ", "rotationX", "rotationY", "rotationZ"):
                        if key in tr:
                            try:
                                val = float(tr[key])
                                if not math.isfinite(val):
                                    issues.append(f"Non-finite {key} on '{ename}' in anim '{anim.get('code', anim.get('name',''))}'. Set to 0.")
                                    val = 0.0
                                if abs(val) < eps:
                                    val = 0.0
                                tr[key] = val
                            except Exception:
                                issues.append(f"Non-numeric {key} on '{ename}' in anim '{anim.get('code', anim.get('name',''))}'. Set to 0.")
                                tr[key] = 0.0

                    # Unwrap rotations per-channel
                    rx = float(tr.get("rotationX", 0.0))
                    ry = float(tr.get("rotationY", 0.0))
                    rz = float(tr.get("rotationZ", 0.0))

                    if ename in prev:
                        prx, pry, prz = prev[ename]
                        def _unwrap(p, c):
                            try:
                                k = int(round((p - c) / 360.0))
                                return c + 360.0 * k
                            except Exception:
                                return c
                        rx = _unwrap(prx, rx)
                        ry = _unwrap(pry, ry)
                        rz = _unwrap(prz, rz)

                    # Clamp extreme rotations (helps broken conversions)
                    if rot_clamp > 0.0:
                        def _clamp(v):
                            if v > rot_clamp:
                                return rot_clamp
                            if v < -rot_clamp:
                                return -rot_clamp
                            return v
                        crx, cry, crz = _clamp(rx), _clamp(ry), _clamp(rz)
                        if (crx, cry, crz) != (rx, ry, rz):
                            issues.append(f"Clamped rotation on '{ename}' in anim '{anim.get('code', anim.get('name',''))}'.")
                        rx, ry, rz = crx, cry, crz

                    tr["rotationX"], tr["rotationY"], tr["rotationZ"] = rx, ry, rz
                    prev[ename] = (rx, ry, rz)
        return issues

    if sanitize_keyframes and "animations" in model_json:
        san_issues = _sanitize_animations(model_json.get("animations", []), eps=sanitize_epsilon, rot_clamp=sanitize_rot_clamp_deg)
        if san_issues:
            print("\n[VS Export] Keyframe Sanitizer Report")
            for s in san_issues[:200]:
                print("  -", s)
            if len(san_issues) > 200:
                print(f"  ... {len(san_issues)-200} more")
            if logger is not None:
                try:
                    logger.report({"WARNING"}, f"Keyframe sanitizer: {len(san_issues)} notes (see console)")
                except Exception:
                    pass

    # =========================================================================
    # Export validation report (ALWAYS runs; hard-fails on critical issues)
    # =========================================================================
    def _collect_elements(el_list):
        out = []
        stack = list(el_list or [])
        while stack:
            e = stack.pop()
            if not isinstance(e, dict):
                continue
            out.append(e)
            for c in e.get("children", []) or []:
                stack.append(c)
        return out

    def _validate_export(model):
        errors = []
        warnings = []

        roots = model.get("elements", []) or []
        if len(roots) == 0:
            errors.append("No exported elements found.")
        elif len(roots) > 1:
            warnings.append(f"Model has {len(roots)} root elements (valid, but often indicates broken parenting).")

        elems = _collect_elements(roots)
        names = []
        for e in elems:
            n = e.get("name", None)
            if isinstance(n, str):
                names.append(n)
        name_set = set(names)
        if len(name_set) != len(names):
            # duplicates are almost always fatal for animation resolution
            dup = sorted({n for n in names if names.count(n) > 1})
            errors.append(f"Duplicate element names detected: {', '.join(dup[:20])}" + ("..." if len(dup) > 20 else ""))

        # Validate element numeric sanity
        def _num_ok(v):
            try:
                fv = float(v)
                return math.isfinite(fv)
            except Exception:
                return False

        for e in elems:
            fn = e.get("name", "<unnamed>")
            fr = e.get("from", None)
            to = e.get("to", None)
            ro = e.get("rotationOrigin", None)
            for label, arr in (("from", fr), ("to", to), ("rotationOrigin", ro)):
                if arr is None:
                    errors.append(f"Element '{fn}' missing '{label}'.")
                    continue
                if not isinstance(arr, (list, tuple)) or len(arr) != 3:
                    errors.append(f"Element '{fn}' has invalid '{label}' (expected 3 numbers).")
                    continue
                if not all(_num_ok(x) for x in arr):
                    errors.append(f"Element '{fn}' has non-finite values in '{label}'.")

            # size sanity
            if isinstance(fr, (list, tuple)) and isinstance(to, (list, tuple)) and len(fr) == 3 and len(to) == 3:
                try:
                    sx, sy, sz = float(to[0]) - float(fr[0]), float(to[1]) - float(fr[1]), float(to[2]) - float(fr[2])
                    if sx < 0 or sy < 0 or sz < 0:
                        errors.append(f"Element '{fn}' has negative size (to < from).")
                    elif sx == 0 or sy == 0 or sz == 0:
                        warnings.append(f"Element '{fn}' has zero size on at least one axis.")
                except Exception:
                    pass

        # Validate animations
        anims = model.get("animations", []) or []
        if anims:
            for anim in anims:
                acode = anim.get("code", anim.get("name", "<anim>"))
                kfs = anim.get("keyframes", []) or []
                if not kfs:
                    errors.append(f"Animation '{acode}' has no keyframes.")
                    continue
                # Track last values for extreme-delta detection per element
                last = {}
                for kf in kfs:
                    if "frame" not in kf:
                        errors.append(f"Animation '{acode}' has a keyframe missing 'frame'.")
                        continue
                    elems_k = kf.get("elements", None)
                    if elems_k is None or not isinstance(elems_k, dict):
                        errors.append(f"Animation '{acode}' keyframe {kf.get('frame','?')} missing 'elements' dict.")
                        continue
                    for ename, tr in elems_k.items():
                        if ename not in name_set:
                            errors.append(f"Animation '{acode}' targets missing element '{ename}'.")
                            continue
                        if not isinstance(tr, dict):
                            errors.append(f"Animation '{acode}' keyframe {kf.get('frame','?')} element '{ename}' has invalid transform.")
                            continue
                        for key in ("offsetX", "offsetY", "offsetZ", "rotationX", "rotationY", "rotationZ"):
                            if key not in tr:
                                errors.append(f"Animation '{acode}' element '{ename}' missing '{key}'.")
                                continue
                            if not _num_ok(tr[key]):
                                errors.append(f"Animation '{acode}' element '{ename}' has non-finite '{key}'.")

                        # extreme delta check (warn; doesn't fail unless absurd)
                        try:
                            cur = (
                                float(tr.get("offsetX", 0.0)), float(tr.get("offsetY", 0.0)), float(tr.get("offsetZ", 0.0)),
                                float(tr.get("rotationX", 0.0)), float(tr.get("rotationY", 0.0)), float(tr.get("rotationZ", 0.0)),
                            )
                            if ename in last:
                                prevv = last[ename]
                                drot = max(abs(cur[3]-prevv[3]), abs(cur[4]-prevv[4]), abs(cur[5]-prevv[5]))
                                dpos = max(abs(cur[0]-prevv[0]), abs(cur[1]-prevv[1]), abs(cur[2]-prevv[2]))
                                if drot > 1440:
                                    warnings.append(f"Animation '{acode}' element '{ename}' has large rotation jump ({drot:.1f} deg).")
                                if drot > 20000:
                                    errors.append(f"Animation '{acode}' element '{ename}' has absurd rotation jump ({drot:.1f} deg).")
                                if dpos > 2000:
                                    warnings.append(f"Animation '{acode}' element '{ename}' has large offset jump ({dpos:.3f}).")
                                if dpos > 1e6:
                                    errors.append(f"Animation '{acode}' element '{ename}' has absurd offset jump ({dpos:.3f}).")
                            last[ename] = cur
                        except Exception:
                            pass

        return errors, warnings

    v_errors, v_warnings = _validate_export(model_json)
    if v_errors or v_warnings:
        print("\n[VS Export] Validation Report")
        if v_errors:
            print("Errors:")
            for e in v_errors:
                print("  -", e)
        if v_warnings:
            print("Warnings:")
            for w in v_warnings[:200]:
                print("  -", w)
            if len(v_warnings) > 200:
                print(f"  ... {len(v_warnings)-200} more")

    if v_errors:
        if logger is not None:
            try:
                logger.report({"ERROR"}, f"Export validation failed: {len(v_errors)} error(s) (see console)")
            except Exception:
                pass
        return {"CANCELLED"}, {"ERROR"}, "Export cancelled due to validation errors (see console)"
    elif v_warnings and logger is not None:
        try:
            logger.report({"WARNING"}, f"Export validation: {len(v_warnings)} warning(s) (see console)")
        except Exception:
            pass

    # =========================================================================
    # minification options to reduce .json file size
    # =========================================================================
    indent = 2
    if minify == True:
        # remove json indent + newline
        indent = None

    # go through json dict and replace all float with rounded strings
    if decimal_precision >= 0:
        parameters_to_round = ("offsetX", "offsetY", "offsetZ", "rotationX", "rotationY", "rotationZ")

        def normalize(d):
            if minify == True: # remove trailing .0 as extreme minify
                if isinstance(d, int):
                    return d
                if isinstance(d, float) and d.is_integer():
                    return int(d)
            return d


        def round_float(x):
            value = round(x, decimal_precision)
            return normalize(value)
        

        def minify_element(elem):
            elem["from"] = [round_float(x) for x in elem["from"]]
            elem["to"] = [round_float(x) for x in elem["to"]]
            elem["rotationOrigin"] = [round_float(x) for x in elem["rotationOrigin"]]
            
            for param in parameters_to_round:
                if param in elem:
                    elem[param] = round_float(elem[param])

            for face in elem["faces"].values():
                _uv = [round_float(x) for x in face["uv"]]
                # VSMC is happiest with integer pixel UVs when possible.
                face["uv"] = [int(round(v)) if abs(v - round(v)) < 1e-5 else v for v in _uv]
            
            for child in elem["children"]:
                minify_element(child)


        def minify_frame(frame_elem):
            for param in parameters_to_round:
                if param in frame_elem:
                    frame_elem[param] = round_float(frame_elem[param])


        def minify_animations(anim):
            for keyframe in anim["keyframes"]:
                elements = keyframe["elements"]
                for key in elements:
                    minify_frame(elements[key])


        for elem in model_json["elements"]:
            minify_element(elem)

        if "animations" in model_json:
            for anim in model_json["animations"]:
                minify_animations(anim)
    
    # save main shape json
    with open(filepath, "w") as f:
        json.dump(model_json, f, separators=(",", ":"), indent=indent)
    
    if generate_animations_file and "animations" in model_json:
        # additionally, save animations to separate .json file
        animations_dict = {
            "animations": model_json["animations"],
        }
        animations_filepath = os.path.splitext(filepath)[0] + "_animations.json"
        with open(animations_filepath, "w") as f:
            json.dump(animations_dict, f, separators=(",", ":"), indent=indent)
    
    
    # -------------------------------------------------------------------------
    # Restore armature state (pose/action/frame) so exporting doesn't mutate scene
    # -------------------------------------------------------------------------
    if '_arm_restore' in locals() and _arm_restore is not None and armature is not None:
        try:
            armature.animation_data_create()
            armature.data.pose_position = _arm_restore[0]
            armature.animation_data.action = _arm_restore[1]
            bpy.context.scene.frame_set(_arm_restore[2])
            bpy.context.view_layer.update()
        except Exception:
            pass

    if len(root_elements) == 0:
        status = {"WARNING"}
        return {"FINISHED"}, status, f"Exported to {filepath} (Warn: No objects exported)"
    else:
        return {"FINISHED"}, status, f"Exported to {filepath}"