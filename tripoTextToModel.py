import os
import time
import requests
import folder_paths #type: ignore


class TripoTTM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cute 3D robot toy", "tooltip": "Describe the object you want to create in 3D."}),
                "model_version": (["v3.0-20250812", "v2.5-20250123", "v2.0-20240919", "v1.4-20240625", "Turbo-v1.0-20250506"], {"default": "v2.5-20250123"}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Tripo API key or .env variable name (TRIPO_API_KEY)"}),
                "save_name": ("STRING", {"default": "tripo_model"}),
            },
            "optional": {
                "seed": ("INT", {"default": 69, "min": 0, "max": 2147483646, "step": 1}),
                "face_limit": ("INT", {"default": 20000, "min": 500, "max": 100000, "tooltip": "Polygon count. 20,000 is good; 50,000+ is high detail."}),
                "texture_size": ([512, 1024, 2048], {"default": 1024}),
                "pbr": ("BOOLEAN", {"default": True, "tooltip": "Adds realistic lighting data (shine, roughness, metalness)."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Things for the AI to avoid."})
            }
        }
    
    RETURN_TYPES, RETURN_NAMES = ("STRING",), ("glb",)
    FUNCTION, CATEGORY = "gen", "3D/generation"

    def gen(self, prompt, model_version, texture_quality, api_key, save_name, **kw):
        # 1. Setup Auth & Session
        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("TRIPO_API_KEY")
        if not key: raise ValueError("No Tripo API key provided.")
        
        s = requests.Session()
        s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        
        # 2. Prepare Payload
        payload = {
            "type": "text_to_model",
            "model_version": model_version,
            "prompt": prompt,
            "texture_quality": texture_quality,
            "face_limit": kw.get("face_limit", 20000),
            "texture_size": kw.get("texture_size", 1024),
            "pbr": kw.get("pbr", True),
            "model_seed": kw.get("seed", 0) # Maps 'seed' widget to 'model_seed' API key
        }
        if kw.get("negative_prompt"): payload["negative_prompt"] = kw["negative_prompt"]
        
        # 3. Submit Task
        print(f"Submitting Tripo Task: {prompt[:30]}...")
        try:
            res = s.post("https://api.tripo3d.ai/v2/openapi/task", json=payload, timeout=30)
            res.raise_for_status()
            task_id = res.json()["data"]["task_id"]
        except Exception as e: raise Exception(f"Tripo Submission Failed: {e}")

        # 4. Poll & Download
        return self._poll(s, task_id, save_name)

    def _poll(self, s, tid, name):
        url, start, m_url = f"https://api.tripo3d.ai/v2/openapi/task/{tid}", time.time(), None
        
        while (time.time() - start) < 600: # 10 min timeout
            try:
                res = s.get(url, timeout=15).json().get("data", {})
                status = res.get("status")
                print(f"Tripo [{tid[:8]}]: {status} ({res.get('progress', 0)}%)")

                if status == "success":
                    out = res.get("output", {})
                    m_url = out.get("model") or out.get("pbr_model") or out.get("base_model")
                    if m_url: break
                elif status in ["failed", "cancelled", "expired", "banned"]:
                    raise Exception(f"Tripo Task {status}: {res.get('error')}")
            except Exception as e: 
                if "Task" in str(e): raise e
            time.sleep(5)
        
        if not m_url: raise Exception("Tripo error: Task finished but no URL provided.")

        path = os.path.join(folder_paths.get_output_directory(), f"{name}_{int(time.time())}.glb")
        with open(path, "wb") as f:
            f.write(s.get(m_url, timeout=120).content)
        return (path,)

    @classmethod
    def IS_CHANGED(cls, prompt, seed, **kw):
        return f"{prompt}-{seed}"

NODE_CLASS_MAPPINGS = {"TripoTTM": TripoTTM}
NODE_DISPLAY_NAME_MAPPINGS = {"TripoTTM": "Tripo Text-to-3D (API)"}