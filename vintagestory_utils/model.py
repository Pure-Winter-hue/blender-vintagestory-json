import bpy
import numpy as np
import bmesh
import re
from mathutils import Matrix, Vector


def _swap_lr_name(name: str) -> str:
    """Swap common *left* markers to *right* markers.

    - Left/left/LEFT -> Right/right/RIGHT (case-aware)
    - common side markers: .L/_L/-L/ L -> .R/_R/-R/ R
    - prefix marker: LEar -> REar

    Intentionally only maps left -> right (not the reverse) to avoid
    double-swaps.
    """
    if not name:
        return name

    def repl_left(m: re.Match) -> str:
        s = m.group(0)
        if s.isupper():
            return "RIGHT"
        if s.islower():
            return "right"
        if s.istitle():
            return "Right"
        return "Right"

    out = re.sub(r"(?i)left", repl_left, name)

    # Prefix like "LEar" (L followed by uppercase letter)
    out = re.sub(r"^L(?=[A-Z])", "R", out)
    out = re.sub(r"^l(?=[A-Z])", "r", out)

    # After separators: ".L", "_L", "-L", " L"
    out = re.sub(r"(?:(?<=\.)|(?<=_)|(?<=-)|(?<=\s))L\b", "R", out)
    out = re.sub(r"(?:(?<=\.)|(?<=_)|(?<=-)|(?<=\s))l\b", "r", out)

    # Token like "L_" / "L-" / "L." / "L "
    out = re.sub(r"\bL(?=(?:[_\-.\s]))", "R", out)
    out = re.sub(r"\bl(?=(?:[_\-.\s]))", "r", out)

    return out



def _unique_name_no_dot(desired: str, exists, suffix: str = "_mir") -> str:
    """Return a unique name without Blender's default '.001' suffix style.

    If `desired` already exists, tries:
      - desired + suffix
      - desired + suffix + <n>  (n = 1..)
    """
    if not desired:
        return desired

    # Fast path
    if not exists(desired):
        return desired

    cand = f"{desired}{suffix}"
    if not exists(cand):
        return cand

    i = 1
    while True:
        cand = f"{desired}{suffix}{i}"
        if not exists(cand):
            return cand
        i += 1

def _link_object_to_same_collections(src: bpy.types.Object, dup: bpy.types.Object) -> None:
    """Link dup to every collection src is in (and ensure it's in the scene)."""
    cols = list(getattr(src, "users_collection", []))
    if not cols:
        bpy.context.scene.collection.objects.link(dup)
        return
    for c in cols:
        if dup.name not in c.objects:
            c.objects.link(dup)


def _duplicate_objects_in_place(objs: list[bpy.types.Object]) -> dict[bpy.types.Object, bpy.types.Object]:
    """Duplicate objects (and their datablocks) without bpy.ops.

    Preserves hierarchy among duplicates.

    Armature objects are intentionally NOT duplicated here; we duplicate bones
    instead so the mirrored copy doesn't share the same bone names/weights.
    """
    mapping: dict[bpy.types.Object, bpy.types.Object] = {}

    # First pass: object copies + datablocks
    for o in objs:
        if o.type == "ARMATURE":
            continue
        dup = o.copy()
        if getattr(o, "data", None) is not None and o.type in {"MESH", "CURVE", "FONT", "SURFACE", "META"}:
            dup.data = o.data.copy()
        mapping[o] = dup
        _link_object_to_same_collections(o, dup)

    # Second pass: parenting (keep duplicates in the same world space)
    for o, dup in mapping.items():
        if o.parent and o.parent in mapping:
            dup.parent = mapping[o.parent]
        else:
            dup.parent = o.parent
        dup.parent_type = o.parent_type
        dup.parent_bone = o.parent_bone
        dup.matrix_world = o.matrix_world.copy()

    return mapping


def _armature_of_object(obj: bpy.types.Object) -> bpy.types.Object | None:
    """Best-effort: find the armature influencing obj."""
    if obj.parent and obj.parent.type == "ARMATURE":
        return obj.parent
    for mod in getattr(obj, "modifiers", []):
        if mod.type == "ARMATURE" and getattr(mod, "object", None) and mod.object.type == "ARMATURE":
            return mod.object
    return None


def _ensure_mirrored_bones(
    arm_obj: bpy.types.Object,
    bone_name_map: dict[str, str],
    axis: str,
    pivot_world: Vector,
) -> int:
    """Create mirrored copies of bones inside an existing armature.

    - Does NOT move existing bones.
    - Creates new bones (if missing) with names from bone_name_map values.
    - Mirrors head/tail in WORLD space around the same plane as object mirror.
    """
    if not arm_obj or arm_obj.type != "ARMATURE":
        return 0

    axis = axis.upper()
    M_world = _mirror_about_world_plane(axis, pivot_world)

    created = 0

    prev_active = bpy.context.view_layer.objects.active
    prev_mode = bpy.context.mode

    try:
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")

        ebones = arm_obj.data.edit_bones
        arm_inv = arm_obj.matrix_world.inverted()

        # Ensure parents are created first (depth sort)
        def bone_depth(bname: str) -> int:
            b = ebones.get(bname)
            d = 0
            while b and b.parent:
                d += 1
                b = b.parent
            return d

        items = sorted(bone_name_map.items(), key=lambda it: bone_depth(it[0]))

        for src_name, dst_name in items:
            if not dst_name or dst_name == src_name:
                continue
            if ebones.get(dst_name) is not None:
                continue
            src = ebones.get(src_name)
            if src is None:
                continue

            # Mirror head/tail around the world plane
            h_w = arm_obj.matrix_world @ src.head
            t_w = arm_obj.matrix_world @ src.tail
            h_w2 = (M_world @ h_w.to_4d()).to_3d()
            t_w2 = (M_world @ t_w.to_4d()).to_3d()
            h_l = arm_inv @ h_w2
            t_l = arm_inv @ t_w2

            nb = ebones.new(dst_name)
            nb.head = h_l
            nb.tail = t_l
            nb.roll = (-src.roll) if axis == "X" else src.roll
            nb.use_connect = src.use_connect

            # Parent to mirrored parent if available, else keep same parent
            if src.parent:
                p_src = src.parent.name
                p_dst = bone_name_map.get(p_src, p_src)
                nb.parent = ebones.get(p_dst) or ebones.get(p_src)
            created += 1

    finally:
        # Restore mode/active
        bpy.ops.object.mode_set(mode="OBJECT")
        if prev_active:
            bpy.ops.object.select_all(action="DESELECT")
            prev_active.select_set(True)
            bpy.context.view_layer.objects.active = prev_active
        if prev_mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode=prev_mode)
            except Exception:
                pass

    return created


def _axis_reflection_matrix_4x4(axis: str) -> Matrix:
    """Return a 4x4 reflection matrix in the given axis (local/object space)."""
    axis = axis.upper()
    if axis == 'X':
        return Matrix(((-1, 0, 0, 0),
                       ( 0, 1, 0, 0),
                       ( 0, 0, 1, 0),
                       ( 0, 0, 0, 1)))
    if axis == 'Y':
        return Matrix((( 1, 0, 0, 0),
                       ( 0,-1, 0, 0),
                       ( 0, 0, 1, 0),
                       ( 0, 0, 0, 1)))
    if axis == 'Z':
        return Matrix((( 1, 0, 0, 0),
                       ( 0, 1, 0, 0),
                       ( 0, 0,-1, 0),
                       ( 0, 0, 0, 1)))
    raise ValueError(f"Invalid axis '{axis}'")


def _mirror_about_world_plane(axis: str, pivot: Vector) -> Matrix:
    """World-space reflection matrix about the plane through pivot with normal on axis.

    For axis X: plane is X=pivot.x, and reflection flips X.
    """
    axis = axis.upper()
    s_world = _axis_reflection_matrix_4x4(axis)
    t_to = Matrix.Translation(pivot)
    t_from = Matrix.Translation(-pivot)
    return t_to @ s_world @ t_from


def _ensure_single_user_mesh(obj: bpy.types.Object) -> None:
    if not isinstance(obj.data, bpy.types.Mesh):
        return
    # If multiple objects share the mesh datablock, we don't want to mirror all of them.
    if obj.data.users > 1:
        obj.data = obj.data.copy()


def _mirror_mesh_local(obj: bpy.types.Object, axis: str) -> None:
    """Mirror mesh coordinates in local space without leaving negative object scale.

    Also fixes face winding and recalculates normals so exporters don't get confused.
    """
    if not isinstance(obj.data, bpy.types.Mesh):
        return

    _ensure_single_user_mesh(obj)
    me: bpy.types.Mesh = obj.data

    bm = bmesh.new()
    bm.from_mesh(me)

    ax = axis.upper()
    ax_i = {'X': 0, 'Y': 1, 'Z': 2}[ax]
    for v in bm.verts:
        v.co[ax_i] *= -1.0

    # Reflection flips handedness; reverse faces to keep normals consistent.
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    bm.to_mesh(me)
    bm.free()
    me.update()


def _bake_negative_scale_into_mesh(obj: bpy.types.Object) -> None:
    """If obj has negative scale components, bake them into mesh and make scale positive."""
    if not isinstance(obj.data, bpy.types.Mesh):
        return

    for ax, i in (('X', 0), ('Y', 1), ('Z', 2)):
        if obj.scale[i] < 0:
            _mirror_mesh_local(obj, ax)
            obj.scale[i] *= -1.0


def _selection_with_children(objs: list[bpy.types.Object]) -> list[bpy.types.Object]:
    seen = set()
    out: list[bpy.types.Object] = []
    def add(o: bpy.types.Object):
        if o.name in seen:
            return
        seen.add(o.name)
        out.append(o)
        for ch in o.children:
            add(ch)
    for o in objs:
        add(o)
    return out


def _depth(o: bpy.types.Object) -> int:
    d = 0
    p = o.parent
    while p is not None:
        d += 1
        p = p.parent
    return d

class OpDuplicateCollection(bpy.types.Operator):
    """Duplicate collection of currently selected object."""
    bl_idname = "vintagestory.duplicate_collection"
    bl_label = "Duplicate Skin Part"
    bl_options = {"REGISTER", "UNDO"}

    name: bpy.props.StringProperty(
        name="name",
        description="New collection name",
    )

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        args = self.as_keywords()

        # unpack args
        new_name = args.get("name")

        if new_name is None or new_name == "":
            self.report({"ERROR"}, "No name provided")
            return {"FINISHED"}
        
        # if name already exists, error and exit
        if new_name in bpy.context.scene.collection.children.keys():
            self.report({"ERROR"}, f"Collection {new_name} already exists")
            return {"FINISHED"}
        
        if len(bpy.context.selected_objects) == 0:
            self.report({"ERROR"}, "No objects selected")
            return {"FINISHED"}

        # create new collection
        obj = bpy.context.selected_objects[0] # take first selected object
        collection = obj.users_collection[0]
        collection_name = collection.name

        if collection == bpy.context.scene.collection:
            self.report({"WARNING"}, "Cannot export root collection")
            return {"CANCELLED"}

        outer_collections = { collection.name: collection for collection in bpy.context.scene.collection.children }

        # first check if obj is a child of a collection in root collections
        # which is the most common case
        collection_to_duplicate = outer_collections.get(collection_name, None)

        if collection_to_duplicate is None:
            # collection is not directly in an outer collection, must search
            # recursively for which outer collection contains the obj's
            # direct collection
            for outer_collection in outer_collections.values():
                if collection in outer_collection.children_recursive:
                    collection_to_duplicate = outer_collection
                    break
        
        if collection_to_duplicate is None:
            self.report({"ERROR"}, "Could not find collection to export, is it in the scene?")
            return {"CANCELLED"}
        
        # create new collection
        new_part_collection = bpy.data.collections.new(new_name)

        # re-cursively duplicate new objects into new collection
        def duplicate_collection(old_collection, new_collection):
            for obj in old_collection.objects:
                new_obj = obj.copy()
                new_obj.data = obj.data.copy()
                new_obj.name = obj.name
                new_collection.objects.link(new_obj)
            for child_collection in old_collection.children:
                new_child_collection = bpy.data.collections.new(child_collection.name)
                new_collection.children.link(new_child_collection)
                duplicate_collection(child_collection, new_child_collection)
        
        duplicate_collection(collection_to_duplicate, new_part_collection)
        
        # link new collection to scene
        bpy.context.scene.collection.children.link(new_part_collection)

        # de-select original object, select new object
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in new_part_collection.all_objects:
            obj.select_set(True)
        
        return {"FINISHED"}


class OpMirrorSelectedSafe(bpy.types.Operator):
    """Mirror selected objects in a way that stays friendly to Vintage Story JSON export.

    This avoids negative object scale (common source of distorted cuboids/UVs in JSON)
    by pushing the reflection into the mesh data and keeping transforms with a
    positive determinant.
    """
    bl_idname = "vintagestory.mirror_selected_safe"
    bl_label = "Mirror (Safe for VS JSON)"
    bl_options = {"REGISTER", "UNDO"}

    axis: bpy.props.EnumProperty(
        name="Axis",
        description="Mirror across a world-space plane perpendicular to this axis",
        items=[('X', 'X', 'Mirror across X plane'),
               ('Y', 'Y', 'Mirror across Y plane'),
               ('Z', 'Z', 'Mirror across Z plane')],
        default='X',
    )

    pivot: bpy.props.EnumProperty(
        name="Pivot",
        description="Where the mirror plane passes through",
        items=[('WORLD_ORIGIN', 'World Origin', 'Plane passes through (0,0,0)'),
               ('CURSOR', '3D Cursor', 'Plane passes through the 3D Cursor'),
               ('ACTIVE', 'Active Object', 'Plane passes through the active object origin'),
               ('MEDIAN', 'Selection Median', 'Plane passes through the median of selection')],
        default='CURSOR',
    )

    include_children: bpy.props.BoolProperty(
        name="Include Children",
        description="Also mirror children of selected objects (useful for part hierarchies)",
        default=True,
    )

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        selection = list(context.selected_objects)
        if not selection:
            self.report({"ERROR"}, "No objects selected")
            return {"CANCELLED"}

        # Expand selection to include children if requested
        base_objs = _selection_with_children(selection) if self.include_children else selection

        # Ensure stable uniqueness (preserve order)
        seen = set()
        base_unique: list[bpy.types.Object] = []
        for o in base_objs:
            if o.name in seen:
                continue
            seen.add(o.name)
            base_unique.append(o)

        # Duplicate first (leave originals untouched)
        dup_map = _duplicate_objects_in_place(base_unique)
        dups = list(dup_map.values())
        if not dups:
            self.report({"ERROR"}, "Nothing to duplicate")
            return {"CANCELLED"}

        # Rename duplicates (objects + datablocks) using source names and avoiding '.001' suffixes
        for src, dup in dup_map.items():
            # Object name
            desired_obj_name = _swap_lr_name(src.name)

            def _obj_exists(n: str, _dup=dup) -> bool:
                other = bpy.data.objects.get(n)
                return (other is not None) and (other != _dup)

            desired_obj_name = _unique_name_no_dot(desired_obj_name, _obj_exists, suffix="_mir")
            if desired_obj_name and dup.name != desired_obj_name:
                dup.name = desired_obj_name

            # Datablock name (meshes most important here)
            if dup.type == 'MESH' and getattr(dup, 'data', None) is not None and getattr(src, 'data', None) is not None:
                desired_data_name = _swap_lr_name(src.data.name)

                def _mesh_exists(n: str, _data=dup.data) -> bool:
                    other = bpy.data.meshes.get(n)
                    return (other is not None) and (other != _data)

                desired_data_name = _unique_name_no_dot(desired_data_name, _mesh_exists, suffix="_mir")
                if desired_data_name and dup.data.name != desired_data_name:
                    dup.data.name = desired_data_name

        
        # Determine pivot point in world space (based on original selection)
        if self.pivot == 'WORLD_ORIGIN':
            pivot = Vector((0.0, 0.0, 0.0))
        elif self.pivot == 'CURSOR':
            pivot = context.scene.cursor.location.copy()
        elif self.pivot == 'ACTIVE' and context.active_object is not None:
            pivot = context.active_object.matrix_world.translation.copy()
        else:
            pts = [o.matrix_world.translation for o in selection]
            pivot = sum(pts, Vector((0.0, 0.0, 0.0))) / max(1, len(pts))

        # If duplicates are skinned, create mirrored bones inside the *existing* armature,
        # then rename vertex groups on the duplicate so it uses those bones.
        arm_to_bonemap: dict[bpy.types.Object, dict[str, str]] = {}
        arm_to_used_names: dict[bpy.types.Object, set[str]] = {}

        dup_meshes = [o for o in dups if o.type == 'MESH']
        for mobj in dup_meshes:
            arm = _armature_of_object(mobj)
            if not arm:
                continue

            # Build a mapping from existing group names -> swapped (and made-unique) names
            # We also ensure the mirrored copy gets *new* bones (no sharing) by always choosing
            # destination names that do not already exist on the armature.
            local_map: dict[str, str] = {}
            used_names = arm_to_used_names.setdefault(arm, set())

            for vg in list(getattr(mobj, 'vertex_groups', [])):
                old = vg.name
                desired = _swap_lr_name(old)
                if desired == old:
                    continue

                def _bone_exists(n: str, _arm=arm, _used=used_names) -> bool:
                    return (n in _used) or (_arm.data.bones.get(n) is not None)

                # Avoid Blender auto '.001' naming by generating our own unique names.
                # Also guarantees we won't collide with pre-existing bones (so we create new ones).
                new = _unique_name_no_dot(desired, _bone_exists, suffix="_mir")

                # Avoid collisions inside this mesh's vertex groups too
                def _vg_exists(n: str, _m=mobj, _old=old) -> bool:
                    return (n in _m.vertex_groups) and (n != _old)

                new = _unique_name_no_dot(new, _vg_exists, suffix="_mir")

                local_map[old] = new
                used_names.add(new)

            if not local_map:
                continue

            # Accumulate for bone creation
            arm_to_bonemap.setdefault(arm, {}).update(local_map)

            # Rename vertex groups on the duplicate (two-pass to avoid transient collisions)
            tmp_names: dict[str, str] = {}
            for old, new in local_map.items():
                vg = mobj.vertex_groups.get(old)
                if vg is None:
                    continue
                tmp = _unique_name_no_dot(f"__tmp__{new}", lambda n, _m=mobj: n in _m.vertex_groups, suffix="_t")
                tmp_names[old] = tmp
                vg.name = tmp

            for old, tmp in tmp_names.items():
                vg = mobj.vertex_groups.get(tmp)
                if vg is None:
                    continue
                vg.name = local_map[old]

        bones_created = 0
        for arm, bonemap in arm_to_bonemap.items():
            bones_created += _ensure_mirrored_bones(arm, bonemap, self.axis, pivot)

        # Mirror duplicated objects (meshes get "safe" mesh reflection; others just transform)
        M_world = _mirror_about_world_plane(self.axis, pivot)
        S_local = _axis_reflection_matrix_4x4(self.axis)

        orig_world = {o: o.matrix_world.copy() for o in dups}
        target_world = {o: (M_world @ orig_world[o] @ S_local) for o in dups}

        dups_sorted = sorted(dups, key=_depth)
        for obj in dups_sorted:
            if obj.type == 'MESH':
                _mirror_mesh_local(obj, self.axis)
                obj.matrix_world = target_world[obj]
                _bake_negative_scale_into_mesh(obj)
            else:
                obj.matrix_world = target_world[obj]

        # Select duplicates for convenience
        bpy.ops.object.select_all(action='DESELECT')
        for o in dups:
            o.select_set(True)
        if context.active_object in dup_map:
            context.view_layer.objects.active = dup_map[context.active_object]
        else:
            context.view_layer.objects.active = dups[0]

        self.report(
            {"INFO"},
            f"Duplicated + mirrored {len(dups_sorted)} object(s); created {bones_created} bone(s)"
        )
        return {"FINISHED"}


class OpCleanupRotation(bpy.types.Operator):
    """Cleanup cuboid edit mode rotation, convert to object mode rotation."""
    bl_idname = "vintagestory.cleanup_rotation"
    bl_label = "Cleanup Rotation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        num_obj_realigned = 0

        for obj in bpy.context.selected_objects:
            if not isinstance(obj.data, bpy.types.Mesh):
                continue
            if len(obj.data.polygons) != 6:
                self.report({"WARN"}, f"{obj.name} is not a cuboid for re-alignment (skipping)")
                continue
            
            # determine if this needs realignment:
            # check if face normals are not aligned with world axis
            obj_is_aligned = True
            for f in obj.data.polygons:
                # sum together abs value of face normal components
                sum_normal_components = sum(abs(c) for c in f.normal)
                # if sum not close to 1, then face normal is not aligned with world axis
                if not np.isclose(sum_normal_components, 1.0, atol=1e-6):
                    obj_is_aligned = False
                    break
            
            if obj_is_aligned:
                continue

            # assumption: object is still a cuboid and face normals
            # are negated with each other
            
            # determine 3 orthogonal vectors from face normals
            # to determine rotation matrix.
            # first find pairs of opposite face normals
            face_normals = [f.normal for f in obj.data.polygons]
            opposite_pairs = set() # set of tuples of opposite face normals
            for i, normal1 in enumerate(face_normals):
                for j, normal2 in enumerate(face_normals):
                    if i == j:
                        continue
                    if np.allclose(normal1, -normal2, atol=1e-3):
                        idx_pair = (i, j) if i < j else (j, i)
                        opposite_pairs.add(idx_pair)
                        break
            
            # if not 3 pairs of opposite face normals, warn and skip
            if len(opposite_pairs) != 3:
                self.report({"ERROR"}, f"{obj.name} does not have 3 pairs of opposite face normals (try re-calculating normals)")
                continue

            # use the first normal of each pair to form the rotation matrix
            rotation_matrix = np.array([face_normals[i] for i, j in opposite_pairs])
            
            # apply rotation to each raw vertex
            v_local = np.array([v.co for v in obj.data.vertices]).T
            v_new = rotation_matrix @ v_local
            for i, v in enumerate(obj.data.vertices):
                v.co = v_new[:,i]
            
            # apply inverse rotation to object transform
            obj.matrix_world = obj.matrix_world @ Matrix(rotation_matrix.T).to_4x4()

            num_obj_realigned += 1

        if num_obj_realigned > 0:
            self.report({"INFO"}, f"Re-aligned {num_obj_realigned} objects")
        else:
            self.report({"INFO"}, "No objects needed realignment")

        return {"FINISHED"}


