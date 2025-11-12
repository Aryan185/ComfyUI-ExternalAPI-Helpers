import os
import time
import shutil
from comfy.comfy_types.node_typing import IO

class ClearDirectoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "sleep_seconds": ("FLOAT", {"default": 1.0, "min": 0.0}),
            },
            "optional": {
                "any_input": (IO.ANY,),
            }
        }
    
    RETURN_TYPES = (IO.ANY, "STRING")
    RETURN_NAMES = ("any_output", "status")
    FUNCTION = "clear_and_return"
    CATEGORY = "Utility/File"
    
    def clear_and_return(self, directory, sleep_seconds, any_input=None):

        if not directory:
            return (any_input, "no_directory_provided")
        
        dirpath = os.path.expanduser(directory)
        if not os.path.exists(dirpath):
            return (any_input, f"directory_not_found:{dirpath}")
        
        if not os.path.isdir(dirpath):
            return (any_input, f"not_a_directory:{dirpath}")
        
        try:
            time.sleep(float(sleep_seconds))
        except Exception:
            pass
        
        try:
            for name in os.listdir(dirpath):
                path = os.path.join(dirpath, name)
                try:
                    if os.path.islink(path) or os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception as e:
                    return (any_input, f"error_removing:{path}:{e}")
            return (any_input, "cleared")
        except Exception as e:
            return (any_input, f"error:{e}")


NODE_CLASS_MAPPINGS = {"ClearDirectoryNode": ClearDirectoryNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ClearDirectoryNode": "Clear Directory After Sleep"}