import time
import os
import io
import torch
import numpy as np
import av
from PIL import Image
from google import genai
from google.genai import types

class VeoGeminiVideoGenerator:    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a cat reading a book"}),
                "model": ([
                    "veo-3.1-generate-preview",
                    "veo-3.1-fast-generate-preview",
                    "veo-3.1-lite",
                    "veo-2.0-generate-001",
                ], {"default": "veo-3.1-generate-preview"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "duration_seconds": ("INT", {"default": 8, "min": 4, "max": 8, "step": 1}),
                "api_key": ("STRING", {"default": "", "multiline": False, "tooltip": "Directly put Gemini API key or .env variable name (GEMINI_API_KEY)"}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"
    OUTPUT_IS_LIST = (True,)
    
    def generate_video(self, seed, prompt, model, aspect_ratio, resolution, duration_seconds, api_key,
                      negative_prompt=None, image=None):
        
        # 1. Setup Client
        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("GEMINI_API_KEY")
        if not key: raise ValueError("API Key required")
        client = genai.Client(http_options={"api_version": "v1beta"}, api_key=key)

        # 2. Duration compatibility clamp (Veo 2.0 requires minimum 5s)
        dur = duration_seconds
        if "veo-2.0" in model and dur < 5:
            dur = 5

        # 3. Prepare Config
        config_params = {
            "aspect_ratio": aspect_ratio,
            "duration_seconds": dur,
            "negative_prompt": negative_prompt.strip() if negative_prompt else None,
        }

        # Resolution is supported on Veo 3.x
        if "veo-3" in model:
            config_params["resolution"] = resolution

        gen_kwargs = {
            "model": model, 
            "prompt": prompt, 
            "config": types.GenerateVideosConfig(**config_params)
        }

        # 4. Handle Image Input (Image-to-Video)
        if image is not None:
            buf = io.BytesIO()
            Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8)).save(buf, format="PNG")
            gen_kwargs["image"] = types.Image(image_bytes=buf.getvalue(), mime_type="image/png")

        # 5. Generate & Poll
        op = client.models.generate_videos(**gen_kwargs)
        print(f"Gemini Veo Operation: {op.name}")
        
        while not op.done:
            time.sleep(4)
            op = client.operations.get(op)
        
        if not op.response or not op.response.generated_videos:
            raise Exception("No videos generated")

        # 6. Download & Decode Video Frames
        video_bytes = io.BytesIO(client.files.download(file=op.response.generated_videos[0].video))
        
        container = av.open(video_bytes)
        frames = []
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_rgb().to_ndarray().astype(np.float32) / 255.0).unsqueeze(0))
        container.close()

        if not frames: raise Exception("Failed to decode video frames")

        return ([torch.cat(frames, dim=0)],)

NODE_CLASS_MAPPINGS = {"VeoGeminiVideoGenerator": VeoGeminiVideoGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"VeoGeminiVideoGenerator": "Veo (Gemini API)"}