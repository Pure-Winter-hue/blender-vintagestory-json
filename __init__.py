bl_info = {
    "name": "Vintage Story JSON Import/Export (Validation + Smart Bake)",
    "description": "Vintage Story JSON import/export",
    "author": "phonon, Pure Winter",
    "version": (0, 9, 0),
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