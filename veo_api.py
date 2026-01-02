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
                "model": (["veo-2.0-generate-001"], {"default": "veo-2.0-generate-001"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration_seconds": ("INT", {"default": 8, "min": 5, "max": 8, "step": 1}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
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
    
    def generate_video(self, seed, prompt, model, aspect_ratio, duration_seconds, api_key,
                      negative_prompt=None, image=None):
        
        # 1. Setup Client
        key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
        if not key: raise ValueError("API Key required")
        client = genai.Client(http_options={"api_version": "v1beta"}, api_key=key)

        # 2. Prepare Config (Seed used for ComfyUI caching only)
        gen_kwargs = {
            "model": model, 
            "prompt": prompt, 
            "config": types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
                negative_prompt=negative_prompt.strip() if negative_prompt else None,
            )
        }

        # 3. Handle Image (In-Memory, ComfyUI Tensors are always [B,H,W,C])
        if image is not None:
            buf = io.BytesIO()
            Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8)).save(buf, format="PNG")
            gen_kwargs["image"] = types.Image(image_bytes=buf.getvalue(), mime_type="image/png")

        # 4. Generate & Poll
        op = client.models.generate_videos(**gen_kwargs)
        print(f"Gemini Veo Operation: {op.name}")
        
        while not op.done:
            time.sleep(4)
            op = client.operations.get(op)
        
        if not op.response or not op.response.generated_videos:
            raise Exception("No videos generated")

        # 5. Download & Decode (Direct Memory Stream)
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