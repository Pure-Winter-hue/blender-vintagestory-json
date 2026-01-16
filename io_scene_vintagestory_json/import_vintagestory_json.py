import os
import numpy as np
import math
import time
import copy
from math import inf
import bpy
from mathutils import Vector, Euler, Quaternion, Matrix
from . import animation

try:
    import pyjson5 as json
except ImportError:
    import json

import re

# -----------------------------------------------------------------------------
# Relaxed JSON loader
# Vintage Story models in the wild are sometimes "JSON-ish" (JSON5-like):
# - trailing commas in arrays/objects
# - // and /* */ comments
# - unquoted keys (name: value)
#
# We keep the optional pyjson5 dependency, but also provide a safe fallback that
# DOES NOT touch colons inside quoted strings (e.g. Windows drive paths "C:/...").
# -----------------------------------------------------------------------------

def _vsjson_strip_comments(s: str) -> str:
    """Remove // and /* */ comments, without touching content inside strings."""
    out = []
    i = 0
    in_str = False
    esc = False

    while i < len(s):
        ch = s[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == '/' and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == '/':
                # line comment: skip until newline, keep newline in output
                i += 2
                while i < len(s) and s[i] not in '\r\n':
                    i += 1
                continue
            if nxt == '*':
                # block comment: skip until */
                i += 2
                while i + 1 < len(s) and not (s[i] == '*' and s[i + 1] == '/'):
                    # preserve newlines so error line numbers stay roughly meaningful
                    if s[i] in '\r\n':
                        out.append(s[i])
                    i += 1
                i += 2 if i + 1 < len(s) else 0
                continue

        out.append(ch)
        i += 1

    return ''.join(out)


def _vsjson_strip_trailing_commas(s: str) -> str:
    """Remove trailing commas before ] or } (JSON5-style), ignoring strings."""
    out = []
    i = 0
    in_str = False
    esc = False

    while i < len(s):
        ch = s[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == ',':
            j = i + 1
            while j < len(s) and s[j] in ' \t\r\n':
                j += 1
            if j < len(s) and s[j] in '}]':
                # skip this comma
                i += 1
                continue

        out.append(ch)
        i += 1

    return ''.join(out)


def _vsjson_quote_unquoted_keys(s: str) -> str:
    """Quote bare word keys in objects: { key: value } -> { "key": value }.
    Safe against colons inside strings. Only targets keys matching \w+.
    """
    out = []
    i = 0
    in_str = False
    esc = False
    stack = []  # '{' or '['
    expect_key = False  # only true when current container is an object and we're expecting a key

    while i < len(s):
        ch = s[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        # whitespace
        if ch in ' \t\r\n':
            out.append(ch)
            i += 1
            continue

        # structure tokens
        if ch == '{':
            stack.append('{')
            expect_key = True
            out.append(ch)
            i += 1
            continue

        if ch == '[':
            stack.append('[')
            expect_key = False
            out.append(ch)
            i += 1
            continue

        if ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
            expect_key = (stack and stack[-1] == '{')
            out.append(ch)
            i += 1
            continue

        if ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
            expect_key = (stack and stack[-1] == '{')
            out.append(ch)
            i += 1
            continue

        if ch == ':':
            # after a key in an object
            expect_key = False
            out.append(ch)
            i += 1
            continue

        if ch == ',':
            # after a value (or after a key-value pair), decide what's next based on container
            expect_key = (stack and stack[-1] == '{')
            out.append(ch)
            i += 1
            continue

        # Potential bare key?
        if expect_key and stack and stack[-1] == '{' and re.match(r'[A-Za-z_]', ch):
            j = i + 1
            while j < len(s) and re.match(r'[A-Za-z0-9_]', s[j]):
                j += 1

            key = s[i:j]

            k = j
            while k < len(s) and s[k] in ' \t\r\n':
                k += 1

            if k < len(s) and s[k] == ':':
                out.append('"')
                out.append(key)
                out.append('"')
                i = j
                # keep expect_key True until we hit the ':'
                continue

        # default passthrough
        out.append(ch)
        i += 1

    return ''.join(out)


def _vsjson_loads(s: str):
    """Try strict JSON, then a safe relaxed pre-processing pipeline."""
    try:
        return json.loads(s)
    except Exception:
        s2 = _vsjson_strip_trailing_commas(_vsjson_strip_comments(s))
        try:
            return json.loads(s2)
        except Exception:
            s3 = _vsjson_quote_unquoted_keys(s2)
            return json.loads(s3)

import importlib
importlib.reload(animation)

# convert deg to rad
DEG_TO_RAD = math.pi / 180.0

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
# shape (n,v) = (6,3)
DIRECTION_NORMALS = np.array([
    [-1.,  0.,  0.],
    [ 0.,  1.,  0.],
    [ 0., -1.,  0.],
    [ 1.,  0.,  0.],
    [ 0.,  0.,  1.],
    [ 0.,  0., -1.],
])


def index_of(val, in_list):
    """Return index of value in in_list"""
    try:
        return in_list.index(val)
    except ValueError:
        return -1 


def merge_dict_properties(dict_original, d):
    """Merge inner dict properties"""
    for k in d:
        if k in dict_original and isinstance(dict_original[k], dict):
            dict_original[k].update(d[k])
        else:
            dict_original[k] = d[k]
    
    return dict_original


def get_base_path(filepath_parts, curr_branch=None, new_branch=None):
    """"Typical path formats for texture is like:
            "textures": {
                "skin": "entity/wolf/wolf9"
            },
    Matches the base path before branch, then get base path of other branch
        curr_branch = "shapes"
        new_branch = "textures"
        filepath_parts = ["C:", "vs", "resources", "shapes", "entity", "land", "wolf-male.json"]
                                                       |
                                             Matched branch point
        
        new base path = ["C:", "vs", "resources"] + ["textures"]
    """
    # match base path
    idx_base_path = index_of(curr_branch, filepath_parts)

    if idx_base_path != -1:
        # system agnostic path join
        joined_path = os.path.join(os.sep, filepath_parts[0] + os.sep, *filepath_parts[1:idx_base_path], new_branch)
        return joined_path
    else:
        return "" # failed


def create_textured_principled_bsdf(mat_name, tex_path, resolution, tex_width, tex_height):
    """Create new material with `mat_name` and texture path `tex_path`
    """
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    bsdf = nodes.get("Principled BSDF") 

    # add texture node
    if bsdf is not None:
        if "Base Color" in bsdf.inputs:
            tex_input = nodes.new(type="ShaderNodeTexImage")
            tex_input.interpolation = "Closest"

            # load image, if fail make a new image with filepath set to tex path
            try:
                img = bpy.data.images.load(tex_path, check_existing=True)
            except:
                print("FAILED TO LOAD IMAGE:", tex_path)
                img = bpy.data.images.new(os.path.split(tex_path)[-1], width=resolution * tex_width, height=resolution * tex_height)
                img.filepath = tex_path
        
            tex_input.image = img
            node_tree.links.new(tex_input.outputs[0], bsdf.inputs["Base Color"])
            node_tree.links.new(tex_input.outputs[1], bsdf.inputs["Alpha"]) #  We also want the alpha to be bound to not confuse the end user.
        
        # disable shininess
        if "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.0
    
    return mat


def parse_element(
    e,
    parent_cube_origin,
    parent_rotation_origin,  # = blender origin
    textures,
    tex_width=16.0,
    tex_height=16.0,
    import_uvs=True,
):
    """Load a single element into a Blender object.
    Note vintage story cube origins are relative to the parent's
    "from" corner, the origin input is the parent cube's from vertex.
                     from     to    (relative to parent_cube_origin)
                       |       |
                       v       v
                       |   x   |   child
                       |_______| 
                |
                |  xp  <------------parent rotation origin
                |.____  parent         (blender origin)
            parent_cube_origin

              .  
            (0,0,0)
    
    New locations in blender space:
        child_blender_origin = parent_cube_origin - parent_rotation_origin + child_rotation_origin
        from_blender_local = from - child_rotation_origin
        to_blender_local = to - child_rotation_origin

    Return tuple of:
        obj,                   # new Blender object
        local_cube_origin,     # local space "to" cube corner origin
        new_cube_origin,       # global space cube corner origin
        new_rotation_origin    # local space rotation (Blender) origin
    """
    # name-based helpers
    vs_name = e.get("name") or "cube"
    is_storage = isinstance(vs_name, str) and ("storage" in vs_name.casefold())

    # get cube min/max
    v_min = np.array([e["from"][2], e["from"][0], e["from"][1]])
    v_max = np.array([e["to"][2], e["to"][0], e["to"][1]])

    # get rotation origin
    location = np.array([
        parent_cube_origin[0] - parent_rotation_origin[0],
        parent_cube_origin[1] - parent_rotation_origin[1], 
        parent_cube_origin[2] - parent_rotation_origin[2],
    ])
    if "rotationOrigin" in e: # add rotation origin
        child_rotation_origin = np.array([
            e["rotationOrigin"][2],
            e["rotationOrigin"][0],
            e["rotationOrigin"][1],
        ])
        location = location + child_rotation_origin
    else:
        child_rotation_origin = np.array([0.0, 0.0, 0.0])
    
    # this cube corner origin
    new_cube_origin = parent_cube_origin + v_min
    new_rotation_origin = parent_rotation_origin + location

    # get euler rotation
    rot_euler = np.array([0.0, 0.0, 0.0])
    if "rotationX" in e:
        rot_euler[1] = e["rotationX"] * DEG_TO_RAD
    if "rotationY" in e:
        rot_euler[2] = e["rotationY"] * DEG_TO_RAD
    if "rotationZ" in e:
        rot_euler[0] = e["rotationZ"] * DEG_TO_RAD

    # create cube
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.rotation_mode = 'XZY'
    mesh = obj.data
    mesh_materials = {} # tex_name => material_index

    # NOTE: We intentionally do NOT strip faces here.
    # The importer below still needs a valid cube mesh (verts + polys) to assign coordinates and UVs.
    # We'll convert storage volumes to a faceless wire helper at the end of this function.


    # center local mesh coordiantes
    v_min = v_min - child_rotation_origin
    v_max = v_max - child_rotation_origin
    
    # set vertices
    mesh.vertices[0].co[:] = v_min[0], v_min[1], v_min[2]
    mesh.vertices[1].co[:] = v_min[0], v_min[1], v_max[2]
    mesh.vertices[2].co[:] = v_min[0], v_max[1], v_min[2]
    mesh.vertices[3].co[:] = v_min[0], v_max[1], v_max[2]
    mesh.vertices[4].co[:] = v_max[0], v_min[1], v_min[2]
    mesh.vertices[5].co[:] = v_max[0], v_min[1], v_max[2]
    mesh.vertices[6].co[:] = v_max[0], v_max[1], v_min[2]
    mesh.vertices[7].co[:] = v_max[0], v_max[1], v_max[2]
    # set face uvs
    uv_faces = e.get("faces") or {}
    element_uv = e.get("uv")
    element_uv0 = None
    if isinstance(element_uv, (list, tuple)) and len(element_uv) == 2:
        try:
            element_uv0 = (float(element_uv[0]), float(element_uv[1]))
        except Exception:
            element_uv0 = None

    # -------------------------------------------------------------------------
    # Disabled faces (VSMC style)
    # A face may be disabled by setting face["enabled"]=false.
    # (We DO NOT treat 0-UV as disabled, because artists sometimes intentionally place UVs at 0,0
    # e.g. hair cards, and we don't want that to get nuked or shown as "red".)
    # -------------------------------------------------------------------------
    def _is_disabled_face(face_dict):
        if not isinstance(face_dict, dict):
            return False
        if face_dict.get("enabled", True) is False:
            return True
        return False
    
    disabled_dirs = {d for d, fdict in uv_faces.items() if _is_disabled_face(fdict)}
    if disabled_dirs:

        # Persist disabled face directions on the object for lossless round-tripping.
        # Export will honor this even if geometry is later edited.
        try:
            obj["vs_disabled_faces"] = sorted(list(disabled_dirs))
        except Exception:
            pass

        try:
            import bmesh
            # IMPORTANT:
            # Blender's Mesh polygon indexing and bmesh face lookup ordering are not guaranteed
            # to be identical. The previous implementation deleted by numeric index, which could
            # result in only one (or the wrong) face being removed when multiple faces are disabled.
            #
            # Fix: classify and delete faces directly in bmesh by their normals.
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.faces.ensure_lookup_table()
            # normal_update may not exist on very old Blender/bmesh versions
            try:
                bm.normal_update()
            except Exception:
                pass

            faces_to_delete = []
            for f in bm.faces:
                n = np.array(f.normal[:], dtype=float)
                # Dot with the 6 canonical axis normals -> (6,)
                dots = n @ DIRECTION_NORMALS.T
                di = int(np.argmax(dots))
                dname = str(DIRECTIONS[di])
                if dname in disabled_dirs:
                    faces_to_delete.append(f)

            if faces_to_delete:
                bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")

            bm.to_mesh(mesh)
            bm.free()
            # Ensure viewport updates immediately
            try:
                mesh.update()
            except Exception:
                pass
        except Exception as ex:
            print(f"[VS Import] Failed to delete disabled faces: {ex}")
    
    # Precompute "autoUv" rectangles from element-level uv offset (VSMC style).
    # Many VSMC/Vintage Story shapes use element["uv"] with face["autoUv"]=true and omit explicit per-face uv rectangles.
    auto_uv_rects = None
    if element_uv0 is not None:
        try:
            # JSON coords: x, y, z
            x1, y1, z1 = float(e["from"][0]), float(e["from"][1]), float(e["from"][2])
            x2, y2, z2 = float(e["to"][0]), float(e["to"][1]), float(e["to"][2])
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            dz = abs(z2 - z1)
            u0, v0 = element_uv0

            # Standard box net (matches common model creators derived from MrCrayfish's Model Creator):
            #   [  up  ][ down ]
            # [west][north][east][south]
            # (Coordinates are in pixel units in VS JSON space: origin top-left, +v downward.)
            auto_uv_rects = {
                "west":  (u0,            v0 + dz,       u0 + dz,            v0 + dz + dy),
                "north": (u0 + dz,       v0 + dz,       u0 + dz + dx,       v0 + dz + dy),
                "east":  (u0 + dz + dx,  v0 + dz,       u0 + dz + dx + dz,  v0 + dz + dy),
                "south": (u0 + dz + dx + dz, v0 + dz,   u0 + dz + dx + dz + dx, v0 + dz + dy),
                "up":    (u0 + dz,       v0,            u0 + dz + dx,       v0 + dz),
                "down":  (u0 + dz + dx,  v0,            u0 + dz + dx + dx,  v0 + dz),
            }
        except Exception:
            auto_uv_rects = None

    def _interpret_vs_uv_rect(uv4, tex_w, tex_h):
        """Interpret a VS/VSMC uv array as either [x1,y1,x2,y2] or [x,y,w,h].

        Returns (xmin, ymin, xmax, ymax) in VS pixel space (top-left origin).
        """
        try:
            u0, v0, u2, v2 = [float(v) for v in uv4]
        except Exception:
            return None

        # Candidate A: [x1,y1,x2,y2]
        a = (u0, v0, u2, v2)

        # Candidate B: [x,y,w,h] -> [x1,y1,x1+w,y1+h]
        b = (u0, v0, u0 + u2, v0 + v2)

        def ok(rect):
            x1_, y1_, x2_, y2_ = rect
            # allow swapped coords (flips), but require non-zero area
            if abs(x2_ - x1_) < 1e-6 or abs(y2_ - y1_) < 1e-6:
                return False
            # bounds check (loose)
            xmn, xmx = (min(x1_, x2_), max(x1_, x2_))
            ymn, ymx = (min(y1_, y2_), max(y1_, y2_))
            eps = 1e-3
            return (xmx <= tex_w + eps) and (ymx <= tex_h + eps) and (xmn >= -eps) and (ymn >= -eps)

        a_ok = ok(a)
        b_ok = ok(b)

        if a_ok and not b_ok:
            return a
        if b_ok and not a_ok:
            return b
        if a_ok and b_ok:
            # Prefer the VS engine/our exporter convention when both are plausible.
            return a

        # neither passes strict checks, fall back to A
        return a

    # Storage volumes are non-rendering helpers in Vintage Story.
    # We still import them as selectable objects, but we don't bother importing UVs/materials.
    if import_uvs and (not is_storage) and (uv_faces or auto_uv_rects is not None):
        # get uvs per face in blender loop order
        uv_layer = mesh.uv_layers.active.data

        # Detect mesh face directions from normals.
        # We compute the best matching direction by dot-product against the
        # canonical face normals (same approach as the original importer).
        # Map face normal -> direction index, then to direction name.
        # Works even if the cuboid has < 6 faces (e.g. VSMC-disabled faces).
        face_normals = np.array([f.normal[:] for f in mesh.polygons], dtype=float).reshape((-1, 3))  # (F,3) even when F==0
        # Dot each face normal with the 6 canonical axis normals -> (F,6)
        dots = face_normals @ DIRECTION_NORMALS.T
        dir_idx = np.argmax(dots, axis=1)
        face_directions = DIRECTIONS[dir_idx]

        for uv_direction, face in zip(face_directions, mesh.polygons):
            face_uv = uv_faces.get(uv_direction)

            # Determine whether we should use explicit UVs or auto UVs
            use_autouv = False
            rotation = 0

            if isinstance(face_uv, dict):
                rotation = int(face_uv.get("rotation") or 0)
                # In many VSMC exports, autoUv defaults to true when explicit uv isn't present.
                if (face_uv.get("autoUv", True) is True) and ("uv" not in face_uv):
                    use_autouv = True
            else:
                face_uv = None

            # choose UV rectangle in VS pixel space
            if face_uv is not None and "uv" in face_uv:
                uv_rect = _interpret_vs_uv_rect(face_uv.get("uv"), tex_width, tex_height)
                if uv_rect is None:
                    uv_rect = (0.0, 0.0, tex_width, tex_height)
                xmin_px, ymin_px, xmax_px, ymax_px = uv_rect
            elif use_autouv and auto_uv_rects is not None and uv_direction in auto_uv_rects:
                xmin_px, ymin_px, xmax_px, ymax_px = auto_uv_rects[uv_direction]
            elif auto_uv_rects is not None and uv_direction in auto_uv_rects and face_uv is None:
                # faces missing entirely, but element has UV offset: assume auto unwrap
                xmin_px, ymin_px, xmax_px, ymax_px = auto_uv_rects[uv_direction]
            else:
                # default full map
                xmin_px, ymin_px, xmax_px, ymax_px = (0.0, 0.0, tex_width, tex_height)

            # convert VS coords (top-left origin) to Blender normalized UVs (bottom-left origin)
            xmin = xmin_px / tex_width
            ymin = 1.0 - (ymax_px / tex_height)
            xmax = xmax_px / tex_width
            ymax = 1.0 - (ymin_px / tex_height)

            if uv_direction == "down":
                rotation = (rotation + 180) % 360

            # apply rotation
            k = face.loop_start
            if rotation == 0:
                uv_layer[k].uv[0:2] = xmax, ymin
                uv_layer[k+1].uv[0:2] = xmax, ymax
                uv_layer[k+2].uv[0:2] = xmin, ymax
                uv_layer[k+3].uv[0:2] = xmin, ymin

            elif rotation == 90:
                uv_layer[k].uv[0:2] = xmax, ymax
                uv_layer[k+1].uv[0:2] = xmin, ymax
                uv_layer[k+2].uv[0:2] = xmin, ymin
                uv_layer[k+3].uv[0:2] = xmax, ymin

            elif rotation == 180:
                uv_layer[k].uv[0:2] = xmin, ymax
                uv_layer[k+1].uv[0:2] = xmin, ymin
                uv_layer[k+2].uv[0:2] = xmax, ymin
                uv_layer[k+3].uv[0:2] = xmax, ymax

            elif rotation == 270:
                uv_layer[k].uv[0:2] = xmin, ymin
                uv_layer[k+1].uv[0:2] = xmax, ymin
                uv_layer[k+2].uv[0:2] = xmax, ymax
                uv_layer[k+3].uv[0:2] = xmin, ymax

            else:  # invalid rotation, should never occur... do default
                uv_layer[k].uv[0:2] = xmax, ymin
                uv_layer[k+1].uv[0:2] = xmax, ymax
                uv_layer[k+2].uv[0:2] = xmin, ymax
                uv_layer[k+3].uv[0:2] = xmin, ymin

            # assign material (kept for compatibility; post-import we collapse to a single 'skin' material)
            if face_uv is not None and "texture" in face_uv:
                tex_name = face_uv["texture"][1:]  # remove the "#" in start
                if tex_name in mesh_materials:
                    face.material_index = mesh_materials[tex_name]
                else:
                    if not tex_name in textures:
                        textures[tex_name] = create_textured_principled_bsdf(tex_name, tex_name, 2, tex_width, tex_height)
                    idx = len(obj.data.materials)
                    obj.data.materials.append(textures[tex_name])
                    mesh_materials[tex_name] = idx
                    face.material_index = idx


    # set name (store VS name for animation mapping; Blender names may normalize whitespace)
    obj["vs_name"] = vs_name
    obj.name = vs_name.strip()
    # Remember the Blender-visible name at import time so we can detect user renames later
    obj["vs_import_blender_name"] = obj.name

    # assign step parent name
    if "stepParentName" in e:
        obj["StepParentName"] = e.get("stepParentName")

    # ---------------------------------------------------------------------
    # Storage helper volumes
    # In VS these are faceless/invisible selection/attachment volumes.
    # In Blender we represent them as a blue wire box: keep verts/edges, delete ALL faces.
    # This must happen after vertex assignment and any optional UV work above.
    # ---------------------------------------------------------------------
    if is_storage:
        try:
            import bmesh
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.faces.ensure_lookup_table()
            if len(bm.faces) > 0:
                # Prefer FACES_ONLY (keeps verts/edges so the object stays a wireframe box).
                # Fall back to FACES if the Blender build doesn't support that context.
                try:
                    bmesh.ops.delete(bm, geom=list(bm.faces), context="FACES_ONLY")
                except Exception:
                    bmesh.ops.delete(bm, geom=list(bm.faces), context="FACES")
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()
        except Exception as ex:
            print(f"[VS Import] Failed to strip faces for storage element '{vs_name}': {ex}")

        # Make it obvious (and nice) in the viewport
        try:
            obj.display_type = 'WIRE'
            obj.show_in_front = True
            obj.color = (0.25, 0.55, 1.0, 1.0)  # blue-ish
        except Exception:
            pass

        # Tag for exporter
        try:
            obj["vs_is_storage"] = True
        except Exception:
            pass

    return obj, v_min, new_cube_origin, new_rotation_origin, location, rot_euler


def parse_attachpoint(
    e,                      # json element
    parent_cube_origin,     # cube corner origin of parent
):
    """Load attachment point associated with a cube, convert
    into a Blender empty object with special name:
        "attach_AttachPointName"
    where the suffix is the "code": "AttachPointName" in the element.
    This format is used for exporting attachpoints from Blender.

    Location in json is relative to cube origin not rotation origin.
    For some reason json number is a string...wtf?
    """
    px = float(e.get("posX") or 0.0)
    py = float(e.get("posY") or 0.0)
    pz = float(e.get("posZ") or 0.0)

    rx = float(e.get("rotationX") or 0.0)
    ry = float(e.get("rotationY") or 0.0)
    rz = float(e.get("rotationZ") or 0.0)

    # get location, rotation converted to Blender space
    location = np.array([
        pz + parent_cube_origin[0],
        px + parent_cube_origin[1],
        py + parent_cube_origin[2],
    ])
    
    rotation = DEG_TO_RAD * np.array([rz, rx, ry])

    # create object
    bpy.ops.object.empty_add(type="ARROWS", radius=1.0, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.show_in_front = True
    obj['vs_desired_loc'] = [float(location[0]), float(location[1]), float(location[2])]
    obj['vs_desired_rot'] = [float(rotation[0]), float(rotation[1]), float(rotation[2])]
    obj.name = "attach_" + (e.get("code") or "attachpoint")

    return obj


def rebuild_hierarchy_with_bones(root_objects):
    """Create an armature that mirrors the imported object hierarchy and
    bone-parent all mesh objects to it WITHOUT changing their world transforms.

    This is critical for Vintage Story animations: the JSON keyframes animate
    named nodes, so we need stable names and stable rest pose.

    Notes:
    - We only create bones for mesh objects (cubes). Attach points and other
      helpers remain regular objects.
    - Object names in VS JSON sometimes contain trailing spaces. Blender tends to
      normalize names, so we store the original name in obj['vs_name'] and
      store the same on the corresponding bone for reliable mapping.
    """
    bpy.ops.object.mode_set(mode="OBJECT")

    # create armature at origin, in edit mode
    bpy.ops.object.add(type="ARMATURE", enter_editmode=True)
    armature_obj = bpy.context.active_object
    armature_obj.show_in_front = True
    arm_data = armature_obj.data

    # collect mesh objects in a stable order (preorder)
    mesh_objs = []

    def collect_meshes(o):
        if isinstance(getattr(o, "data", None), bpy.types.Mesh):
            mesh_objs.append(o)
        for ch in list(o.children):
            collect_meshes(ch)

    for ro in root_objects:
        collect_meshes(ro)

    # create bones in edit mode, using the object's world transform (armature space)
    edit_bones = arm_data.edit_bones

    def add_bone_for_object(o, parent_bone=None):
        bname = o.name
        eb = edit_bones.new(bname)

        vs_name = o.get("vs_name", bname)
        eb["vs_name"] = vs_name  # store original VS name for later mapping
        eb["vs_import_blender_name"] = bname

        # object world matrix -> armature local matrix
        mat = armature_obj.matrix_world.inverted_safe() @ o.matrix_world
        loc, rot, _scale = mat.decompose()
        mat_noscale = rot.to_matrix().to_4x4()
        mat_noscale.translation = loc

        # set orientation/position
        eb.matrix = mat_noscale
        eb.head = loc

        # give it a small length along the bone's local +Y axis
        y_axis = rot.to_matrix() @ Vector((0.0, 0.05, 0.0))
        eb.tail = loc + y_axis

        if parent_bone is not None:
            eb.parent = parent_bone
            eb.use_connect = False

        # recurse
        for ch in list(o.children):
            if isinstance(getattr(ch, "data", None), bpy.types.Mesh):
                add_bone_for_object(ch, eb)
            else:
                # still recurse through non-mesh nodes so grandchildren meshes are not skipped
                def recurse_nonmesh(n, parent_b):
                    for g in list(n.children):
                        if isinstance(getattr(g, "data", None), bpy.types.Mesh):
                            add_bone_for_object(g, parent_b)
                        else:
                            recurse_nonmesh(g, parent_b)
                recurse_nonmesh(ch, eb)

    # build bone tree from current object tree
    for ro in root_objects:
        if isinstance(getattr(ro, "data", None), bpy.types.Mesh):
            add_bone_for_object(ro, None)
        else:
            # root can be non-mesh; recurse until we find mesh descendants
            for ch in list(ro.children):
                if isinstance(getattr(ch, "data", None), bpy.types.Mesh):
                    add_bone_for_object(ch, None)

    # exit edit mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # IMPORTANT: preserve world transforms while re-parenting.
    # First, snapshot world matrices for every mesh object, then unparent them,
    # then bone-parent and restore world matrices.
    world_mats = {o: o.matrix_world.copy() for o in mesh_objs}

    for o in mesh_objs:
        o.parent = None

    for o in mesh_objs:
        o.parent = armature_obj
        o.parent_type = "BONE"
        o.parent_bone = o.name
        o.matrix_world = world_mats[o]

    # import should show the clean rig/rest pose by default
    armature_obj.data.pose_position = "POSE"  # default to Pose so imported Actions preview

    return armature_obj


def resolve_bone_name(armature_obj, vs_name):
    """Resolve a VS animation element name to an existing Blender bone name."""
    bones = armature_obj.data.bones
    if vs_name in bones:
        return vs_name
    stripped = vs_name.strip()
    if stripped in bones:
        return stripped
    # fall back: search by stored vs_name on bones
    for b in bones:
        if b.get("vs_name", "").strip() == vs_name.strip():
            return b.name
    return None


def parse_animation(e, armature_obj, stats):
    """Import a Vintage Story animation (current format only) as a Blender Action.

    We intentionally do NOT support VS animation version 0 here per user request.
    Missing 'version' is treated as current (v1).
    """
    action_name = e.get("code") or e.get("name") or "vs_anim"
    action = bpy.data.actions.new(name=action_name)
    action.use_fake_user = True

    # preserve original VS name/code for roundtrip
    if "name" in e:
        action["vs_anim_name"] = e.get("name")
    if "code" in e:
        action["vs_anim_code"] = e.get("code")

    # metadata passthrough
    if "onAnimationEnd" in e:
        action["on_animation_end"] = e["onAnimationEnd"]
    if "onActivityStopped" in e:
        action["on_activity_stopped"] = e["onActivityStopped"]

    # preserve quantityframes for roundtrip export
    if "quantityframes" in e:
        try:
            action["vs_quantityframes"] = int(e["quantityframes"])
        except Exception:
            pass

    # Ensure armature has animation data and bind the Action to the armature.
    #
    # Blender 4.x uses "Action Slots" under the hood. If you author fcurves on an
    # unbound Action, Blender may later create a fresh empty slot when you assign
    # the Action to an object, making the Action *look* populated (fcurves exist)
    # but evaluate as blank.
    #
    # Keying via PoseBone.keyframe_insert while the Action is assigned guarantees
    # the keys go into the correct slot and will evaluate when you switch Actions.
    armature_obj.animation_data_create()
    armature_obj.animation_data.action = action

    scene = bpy.context.scene
    prev_frame = scene.frame_current

    # cache original pose for touched bones so import leaves the model in rest pose
    touched = {}

    def cache_pose(pb):
        if pb.name in touched:
            return
        touched[pb.name] = (pb.location.copy(), pb.rotation_euler.copy(), pb.rotation_mode)


    # Import keyframes
    keyframes = e.get("keyframes", []) or []
    
    # VS exports are usually integer frame indices (0..quantityframes-1).
    # Some pipelines export normalized time (0..1) or fractional frames.
    # Older versions of this importer rounded to int, which can collapse all keys to frame 0/1.
    fps = float(scene.render.fps) if scene.render.fps else 30.0
    
    qf = None
    try:
        qf_val = e.get("quantityframes")
        qf = int(qf_val) if qf_val is not None else None
    except Exception:
        qf = None
    
    # Pre-scan frames to decide whether we need to scale normalized time.
    raw_frames = []
    for kf in keyframes:
        fr = kf.get("frame", 0)
        try:
            raw_frames.append(float(fr))
        except Exception:
            raw_frames.append(0.0)
    
    max_raw = max(raw_frames) if raw_frames else 0.0
    min_raw = min(raw_frames) if raw_frames else 0.0
    
    # Detect normalized 0..1 time when quantityframes is present.
    normalized = (qf is not None and qf > 1 and max_raw <= 1.000001 and min_raw >= -0.000001)
    
    max_frame_written = 0.0
    
    for keyframe, raw in zip(keyframes, raw_frames):
        frame = raw
        if normalized:
            frame = raw * float(qf - 1)
    
        # Track last written key for setting scene range.
        if frame > max_frame_written:
            max_frame_written = frame
    
        # Keep Blender's internal time in sync (some builds/operators rely on it),
        # but preserve subframe accuracy.
        fi = int(math.floor(frame))
        scene.frame_set(fi, subframe=float(frame - fi))
    
        elements = keyframe.get("elements", {}) or {}
        for vs_bone_name, data in elements.items():
            bone_name = resolve_bone_name(armature_obj, vs_bone_name)
            if bone_name is None:
                continue
    
            pb = armature_obj.pose.bones.get(bone_name)
            if pb is None:
                continue
            cache_pose(pb)
    
            pb.rotation_mode = "XZY"
    
            # translation: VS (X,Y,Z) -> Blender (Z,X,Y) -> (X,Y,Z) = (offsetZ, offsetX, offsetY)
            ox = float(data.get("offsetX", 0.0) or 0.0)
            oy = float(data.get("offsetY", 0.0) or 0.0)
            oz = float(data.get("offsetZ", 0.0) or 0.0)
            pb.location = Vector((oz, ox, oy))
    
            # rotation: VS (X,Y,Z) -> Blender Euler XZY with components (Z,X,Y)
            rx = float(data.get("rotationX", 0.0) or 0.0) * DEG_TO_RAD
            ry = float(data.get("rotationY", 0.0) or 0.0) * DEG_TO_RAD
            rz = float(data.get("rotationZ", 0.0) or 0.0) * DEG_TO_RAD
            pb.rotation_euler = (rz, rx, ry)
    
            # insert keys (use full channels to keep curves stable when VS omits components)
            if any(k in data for k in ("offsetX", "offsetY", "offsetZ")):
                pb.keyframe_insert(data_path="location", index=0, frame=frame)
                pb.keyframe_insert(data_path="location", index=1, frame=frame)
                pb.keyframe_insert(data_path="location", index=2, frame=frame)
    
            # VS commonly relies on default 0 rotations, so always key rotations for authored bones
            pb.keyframe_insert(data_path="rotation_euler", index=0, frame=frame)
            pb.keyframe_insert(data_path="rotation_euler", index=1, frame=frame)
            pb.keyframe_insert(data_path="rotation_euler", index=2, frame=frame)
    
    # Expand scene range to cover imported keys (nice for immediate playback).
    try:
        end_frame = int(math.ceil(max_frame_written))
        if end_frame > scene.frame_end:
            scene.frame_end = end_frame
        # VS commonly starts at frame 0.
        if scene.frame_start > 0:
            scene.frame_start = 0
    except Exception:
        pass
    
    # make all keyframe interpolation linear (VS style)
        for fcu in action.fcurves:
            for kp in fcu.keyframe_points:
                kp.interpolation = "LINEAR"
    
        # restore scene frame and bone poses
        scene.frame_set(prev_frame)
        for bname, (loc, rot, mode) in touched.items():
            pb = armature_obj.pose.bones.get(bname)
            if pb is None:
                continue
            pb.location = loc
            pb.rotation_mode = mode
            pb.rotation_euler = rot
    
        # update stats
        if stats:
            stats.animations += 1
    
        return action
    

def load_element(
    element,
    parent,
    cube_origin,
    rotation_origin,
    all_objects,
    textures,
    tex_width=16.0,
    tex_height=16.0,
    import_uvs=True,
    stats=None,
):
    """Recursively load a geometry cuboid"""

    obj, local_cube_origin, new_cube_origin, new_rotation_origin, desired_loc, desired_rot = parse_element(
        element,
        cube_origin,
        rotation_origin,
        textures,
        tex_width=tex_width,
        tex_height=tex_height,
        import_uvs=import_uvs,
    )
    all_objects.append(obj)
    
    # set parent
    if parent is not None:
        obj.parent = parent
        # Preserve the original VS hierarchy for exporters/animation baking.
        # After rig creation we bone-parent meshes, which would otherwise erase the
        # object parenting information.
        try:
            obj["vs_parent"] = parent.get("vs_name", parent.name)
        except Exception:
            pass
        # We want VS transforms to be interpreted in parent space (not world space).
        # Force parent inverse to identity so obj.matrix_basis is direct local transform.
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.rotation_mode = 'XZY'
        obj.matrix_basis = Matrix.Translation(Vector(desired_loc)) @ Euler(desired_rot, 'XZY').to_matrix().to_4x4()
    else:
        obj.rotation_mode = 'XZY'
        obj.matrix_world = Matrix.Translation(Vector(desired_loc)) @ Euler(desired_rot, 'XZY').to_matrix().to_4x4()
    
    # increment stats (debugging)
    if stats:
        stats.cubes += 1

    # parse attach points
    if "attachmentpoints" in element:
        for attachpoint in element["attachmentpoints"]:
            p = parse_attachpoint(
                attachpoint,
                local_cube_origin,
            )
            p.parent = obj
            try:
                p["vs_parent"] = obj.get("vs_name", obj.name)
            except Exception:
                pass
            try:
                if 'vs_desired_loc' in p and 'vs_desired_rot' in p:
                    p.matrix_parent_inverse = Matrix.Identity(4)
                    p.rotation_mode = 'XZY'
                    p.matrix_basis = Matrix.Translation(Vector(p['vs_desired_loc'])) @ Euler(p['vs_desired_rot'], 'XZY').to_matrix().to_4x4()
            except Exception:
                pass
            all_objects.append(p)
            
            # increment stats (debugging)
            if stats:
                stats.attachpoints += 1

    # recursively load children
    if "children" in element:
        for child in element["children"]:
            load_element(
                child,
                obj,
                local_cube_origin,
                np.array([0.0, 0.0, 0.0]),
                all_objects,
                textures,
                tex_width,
                tex_height,
                import_uvs,
                stats=stats,
            )

    return obj


class ImportStats():
    """Track statistics on imported data"""
    def __init__(self):
        self.cubes = 0
        self.attachpoints = 0
        self.animations = 0
        self.textures = 0


def load(context,
         filepath,
         import_uvs=True,               # import face uvs
         import_textures=True,          # import textures into materials
         import_animations=True,        # load animations
         translate_origin=None,         # origin translate either [x, y, z] or None
         recenter_to_origin=False,      # recenter model to origin, overrides translate origin
         debug_stats=True,              # print statistics on imported models
         **kwargs
):
    """Main import function"""

    # debug
    t_start = time.process_time()
    stats = ImportStats() if debug_stats else None

    with open(filepath, "r", encoding="utf-8-sig") as f:
        s = f.read()
        data = _vsjson_loads(s)

    # chunks of import file path, to get base directory
    filepath_parts = filepath.split(os.path.sep)

    # check if groups in .json, not a spec, used by this exporter as additional data to group models together
    if "groups" in data:
        groups = data["groups"]
    else:
        groups = {}
    
    # objects created
    root_objects = []  # root level objects
    all_objects = []   # all objects added
    armature = None  # created if animations are imported


    # vintage story coordinate system origin
    if translate_origin is not None:
        translate_origin = Vector(translate_origin)

    # set scene collection as active
    scene_collection = bpy.context.view_layer.layer_collection
    bpy.context.view_layer.active_layer_collection = scene_collection

    # =============================================
    # import textures, create map of material name => material
    # =============================================
    """Assume two types of texture formats:
        "textures:" {
            "down": "#bottom",               # texture alias to another texture
            "bottom": "block/stone",   # actual texture image
        }

    Loading textures is two pass:
        1. load all actual texture images
        2. map aliases to loaded texture images
    """
    tex_width = data["textureWidth"] if "textureWidth" in data else 16.0
    tex_height = data["textureHeight"] if "textureHeight" in data else 16.0

    # Store declared texture size on the scene so exports can scale UVs correctly (e.g. for VSMC).
    # This avoids relying on image node sizes, which may be scaled for authoring convenience.
    try:
        scn = bpy.context.scene
        scn["vs_textureWidth"] = float(tex_width)
        scn["vs_textureHeight"] = float(tex_height)
    except Exception:
        pass

    textures = {}
    if import_textures and "textures" in data:
        # get textures base path for models
        tex_base_path = get_base_path(filepath_parts, curr_branch="shapes", new_branch="textures")

        # load texture images
        for tex_name, tex_path in data["textures"].items():
            # skip aliases
            if tex_path[0] == "#":
                continue
            
            filepath_tex = os.path.join(tex_base_path, *tex_path.split("/")) + ".png"
            textures[tex_name] = create_textured_principled_bsdf(tex_name, filepath_tex, 2, tex_width, tex_height)

            # update stats
            if stats:
                stats.textures += 1

        # map texture aliases
        for tex_name, tex_path in data["textures"].items():
            if tex_path[0] == "#":
                tex_path = tex_path[1:]
                if tex_path in textures:
                    textures[tex_name] = textures[tex_path]
    
    # =============================================
    # recursively import geometry, uvs
    # =============================================
    root_origin = np.array([0.0, 0.0, 0.0])

    root_elements = data["elements"]
    for e in root_elements:
        obj = load_element(
            e,
            None,
            root_origin,
            root_origin,
            all_objects,
            textures,
            tex_width=tex_width,
            tex_height=tex_height,
            import_uvs=True,
            stats=stats,
        )
        root_objects.append(obj)

    # =============================================
    # model post-processing
    # =============================================
    if recenter_to_origin:
        # model bounding box vector
        model_v_min = np.array([inf, inf, inf])
        model_v_max = np.array([-inf, -inf, -inf])
        
        # re-used buffer
        v_world = np.zeros((3, 8))
        
        # get bounding box
        for obj in root_objects:
            mesh = obj.data
            mat_world = obj.matrix_world
            for i, v in enumerate(mesh.vertices):
                v_world[0:3,i] = mat_world @ v.co
            
            model_v_min = np.amin(np.append(v_world, model_v_min[...,np.newaxis], axis=1), axis=1)
            model_v_max = np.amax(np.append(v_world, model_v_max[...,np.newaxis], axis=1), axis=1)

        mean = 0.5 * (model_v_min + model_v_max)
        mean = Vector((mean[0], mean[1], mean[2]))

        for obj in root_objects:
            obj.location = obj.location - mean
    
    # do raw origin translation
    elif translate_origin is not None:
        for obj in root_objects:
            obj.location = obj.location + translate_origin
    
    # generate step parent constraints for root objects:
    # if Object has stepParentName definition, try to find and bind the
    # object without using direct blender parent hierarchy.
    for obj in root_objects:
        if "StepParentName" not in obj:
            continue

        step_parent_name = obj["StepParentName"]
        
        # first search for step parent bone inside armatures
        found = False
        for armature in bpy.data.objects:
            if not isinstance(armature.data, bpy.types.Armature):
                continue

            # search for bone with name
            if step_parent_name.startswith("b_"):
                bone = armature.data.bones.get(step_parent_name[2:])
            else:
                bone = armature.data.bones.get(step_parent_name)
            if bone is None:
                continue

            found = True

            # transform object back to bone
            obj.matrix_world = bone.matrix_local @ obj.matrix_world

            # add constraint to make the object a virtual child of the object
            constraint = obj.constraints.new("CHILD_OF")
            constraint.target = armature
            constraint.subtarget = bone.name
            # set inverse matrix to bone
            # https://blenderartists.org/t/set-inverse-child-of-constraints-via-python/1133914/4
            constraint.inverse_matrix = armature.matrix_world @ bone.matrix_local.inverted()
            break
        if found:
            continue

        # else, search for step parent object in scene
        if step_parent_name in bpy.data.objects:
            target = bpy.data.objects[step_parent_name]
            constraint = obj.constraints.new("CHILD_OF")
            constraint.target = target
            # clear inverse so constraint puts obj on target
            constraint.inverse_matrix = Matrix.Identity(4)
            # TODO: idk right way to offset object to parent and constraint here

    # import groups as collections
    for g in groups:
        name = g["name"]
        if name == "Master Collection":
            continue
        
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
        for index in g["children"]:
            col.objects.link(all_objects[index])
            bpy.context.scene.collection.objects.unlink(all_objects[index])
    
    # import animations
    if import_animations and "animations" in data and len(data["animations"]) > 0:
        # go through objects, rebuild hierarchy using bones instead of direct parenting
        # to support bone based animation
        armature = rebuild_hierarchy_with_bones(root_objects)

        # load animations
        first_action = None
        for anim in data["animations"]:
            anim = copy.deepcopy(anim)
            act = parse_animation(anim, armature, stats)
            if first_action is None and act is not None:
                first_action = act

        # Assign a default action so switching Armature to Pose Position immediately previews.
        # In Rest Position the model stays in rig pose.
        if first_action is not None:
            armature.animation_data.action = first_action
    

    # =============================================
    # Post-import cleanup
    # =============================================

    def _vs_sync_object_data_names(objs):
        """Make datablock names match object names for unique users.
        Helps keep Mesh/Armature datablocks stable for export and editing."""
        for o in objs:
            data = getattr(o, "data", None)
            if data is None:
                continue
            try:
                if getattr(data, "users", 0) == 1:
                    data.name = o.name
            except Exception:
                pass

    def _vs_postprocess_single_skin(mesh_objs):
        """Remove all materials on imported meshes and assign a single shared 'skin' material."""
        # Create or get the shared material
        skin = bpy.data.materials.get("skin")
        if skin is None:
            skin = bpy.data.materials.new("skin")
            try:
                skin.use_nodes = True
            except Exception:
                pass

        # Track materials that were previously used by imported meshes so we can purge unused ones.
        prev_mats = set()
        for o in mesh_objs:
            try:
                for m in getattr(o.data, "materials", []) or []:
                    if m is not None:
                        prev_mats.add(m)
            except Exception:
                pass

        # Clear + assign only skin
        for o in mesh_objs:
            me = getattr(o, "data", None)
            if not isinstance(me, bpy.types.Mesh):
                continue

            # clear material slots
            try:
                me.materials.clear()
            except Exception:
                # fallback for older APIs
                while len(me.materials) > 0:
                    me.materials.pop(index=len(me.materials) - 1)

            me.materials.append(skin)

            # force all faces to material slot 0
            try:
                polys = getattr(me, "polygons", None)
                if polys is not None:
                    for p in polys:
                        p.material_index = 0
            except Exception:
                pass

        # Remove now-unused materials that were only introduced by this import
        for m in list(prev_mats):
            if m is None:
                continue
            if m.name == "skin":
                continue
            try:
                if m.users == 0:
                    bpy.data.materials.remove(m)
            except Exception:
                pass

    # Set armature display to rest pose so the model imports in rest pose.
    # This does not delete or modify any actions/keyframes.
    if armature is not None:
        try:
            armature.data.pose_position = "REST"
        except Exception:
            pass

    # Apply single-material policy ONLY when textures/materials were not imported.
    # When importing textures, keep per-face materials so UVs + texture references round-trip.
    if not import_textures:
        _vs_postprocess_single_skin([o for o in all_objects if isinstance(getattr(o, "data", None), bpy.types.Mesh)])

    # Sync datablock names after the import has finished creating/linking objects
    _vs_sync_object_data_names(all_objects + ([armature] if armature is not None else []))

    # select newly imported objects
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in all_objects:
        obj.select_set(True)
    
    # print stats
    if debug_stats:
        t_end = time.process_time()
        dt = t_end - t_start
        print("Imported .json in {}s".format(dt))
        print("- Cubes: {}".format(stats.cubes))
        print("- Attach Points: {}".format(stats.attachpoints))
        print("- Textures: {}".format(stats.textures))
        print("- Animations: {}".format(stats.animations))

    return {"FINISHED"}