bl_info = {
    "name": "Vintage Story JSON Tool",
    "description": "Import/export/modeling/anim/uv/tools for VS Json Models.",
    "author": "phonon, Pure Winter",
    "version": (0, 8, 8),
    "blender": (4, 5, 5),
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