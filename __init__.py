bl_info = {
    "name": "Vintage Story JSON Import/Export",
    "description": "Tool for modeling/uv/anim/exporting VS JSON from Blender.",
    "author": "phonon, Pure Winter",
    "version": (0, 9, 2),
    "blender": (4, 5, 0),
    "location": "File > Import-Export",
    "warning": "",
    "tracker_url": "https://github.com/Pure-Winter-hue/blender-vintagestory-json",
    "category": "Vintage Story",
}

from . import io_scene_vintagestory_json
from . import vintagestory_utils

# reload imported modules
import importlib
importlib.reload(io_scene_vintagestory_json)
importlib.reload(vintagestory_utils)

def register():
    io_scene_vintagestory_json.register()
    vintagestory_utils.register()

def unregister():
    vintagestory_utils.unregister()
    io_scene_vintagestory_json.unregister()
