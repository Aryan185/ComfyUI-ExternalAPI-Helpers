import os
import time
import requests
import io
import numpy as np
from PIL import Image
import folder_paths #type: ignore

class TripoITM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_version": (["v3.0-20250812", "v2.5-20250123", "v2.0-20240919", "v1.4-20240625"], {"default": "v2.5-20250123"}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Tripo API key or .env variable name (TRIPO_API_KEY)"}),
                "save_name": ("STRING", {"default": "tripo_img_model"}),
            },
            "optional": {
                "seed": ("INT", {"default": 69, "min": 0, "max": 2147483646, "step": 1}),
                "texture_seed": ("INT", {"default": 420, "min": 0, "max": 2147483646, "step": 1, "tooltip": "Controls randomness of colors/textures."}),
                "face_limit": ("INT", {"default": 20000, "min": 500, "max": 100000, "tooltip": "Max triangles in mesh."}),
                "texture_size": ([512, 1024, 2048], {"default": 1024}),
                "pbr": ("BOOLEAN", {"default": True, "tooltip": "Include realistic lighting maps."}),
            }
        }
    
    RETURN_TYPES, RETURN_NAMES = ("STRING",), ("glb",)
    FUNCTION, CATEGORY = "gen", "3D/generation"

    def gen(self, image, model_version, texture_quality, api_key, save_name, **kw):
        # 1. Auth & Session
        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("TRIPO_API_KEY")
        if not key: raise ValueError("No API Key")
        s = requests.Session()
        s.headers.update({"Authorization": f"Bearer {key}"})
        
        # 2. Image Prep (Tensor to PNG Bytes)
        img = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        
        # 3. Upload & Task Submission
        try:
            up = s.post("https://api.tripo3d.ai/v2/openapi/upload", files={'file': ('i.png', buf, 'image/png')}, timeout=60).json()
            token = up["data"]["image_token"]
            
            payload = {
                "type": "image_to_model", "model_version": model_version, "texture_quality": texture_quality,
                "file": {"type": "png", "file_token": token},
                "face_limit": kw.get("face_limit", 20000), "texture_size": kw.get("texture_size", 1024), "pbr": kw.get("pbr", True),
                "model_seed": kw.get("seed", 0), "texture_seed": kw.get("texture_seed", 0)
            }
            
            task = s.post("https://api.tripo3d.ai/v2/openapi/task", json=payload, timeout=30).json()
            return self._poll(s, task["data"]["task_id"], save_name)
        except Exception as e: raise Exception(f"Tripo Error: {e}")

    def _poll(self, s, tid, name):
        url, start, m_url = f"https://api.tripo3d.ai/v2/openapi/task/{tid}", time.time(), None
        while (time.time() - start) < 600:
            try:
                res = s.get(url, timeout=15).json().get("data", {})
                status = res.get("status")
                print(f"Tripo [{tid[:8]}]: {status} ({res.get('progress', 0)}%)")
                if status == "success":
                    m_url = res.get("output", {}).get("model") or res.get("output", {}).get("pbr_model")
                    if m_url: break
                elif status in ["failed", "cancelled", "expired"]: raise Exception(f"Task {status}: {res.get('error')}")
            except Exception as e: 
                if "Task failed" in str(e): raise e
            time.sleep(5)
        
        if not m_url: raise Exception("Tripo error: Success but no URL provided.")
        
        path = os.path.join(folder_paths.get_output_directory(), f"{name}_{int(time.time())}.glb")
        with open(path, "wb") as f: f.write(s.get(m_url, timeout=120).content)
        return (path,)

    @classmethod
    def IS_CHANGED(cls, seed, texture_seed, **kw): 
        return f"{seed}-{texture_seed}"

NODE_CLASS_MAPPINGS = {"TripoITM": TripoITM}
NODE_DISPLAY_NAME_MAPPINGS = {"TripoITM": "Tripo Image-to-3D (API)"}