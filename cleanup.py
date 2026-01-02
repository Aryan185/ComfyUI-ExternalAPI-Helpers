import os
import shutil
from comfy.comfy_types.node_typing import IO

class ClearDirectoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "any_input": (IO.ANY,),
            }
        }
    
    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("any_output",)
    FUNCTION = "clear_and_return"
    CATEGORY = "Utility/File"
    
    def clear_and_return(self, directory, any_input=None):
        dirpath = os.path.expanduser(directory)
        if not os.path.exists(dirpath):
            return (any_input,)
        
        if not os.path.isdir(dirpath):
            return (any_input,)
        
        try:
            for name in os.listdir(dirpath):
                path = os.path.join(dirpath, name)
                try:
                    if os.path.islink(path) or os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception:
                    pass
            return (any_input,)
        except Exception:
            return (any_input,)


NODE_CLASS_MAPPINGS = {"ClearDirectoryNode": ClearDirectoryNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ClearDirectoryNode": "Clear Directory"}