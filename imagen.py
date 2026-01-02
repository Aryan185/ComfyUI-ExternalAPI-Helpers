import os
import io
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

class GoogleImagenNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (["models/imagen-4.0-ultra-generate-001", "models/imagen-4.0-generate-001", "models/imagen-4.0-fast-generate-001", "models/imagen-3.0-generate-002"],),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "aspect_ratio": (["1:1", "9:16", "16:9", "4:3", "3:4"],),
                "image_size": (["1K", "2K"],),
                "seed": ("INT", {"default": 69, "min": 1, "max": 2147483646, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                },
                "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "image/generation"
    
    def generate_images(self, prompt, api_key, model, number_of_images, aspect_ratio, image_size, seed, guidance_scale, negative_prompt=""):
        key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
        if not key: raise ValueError("No API key provided.")
        
        client = genai.Client(api_key=key)
        
        config = types.GenerateImagesConfig(
            number_of_images=number_of_images,
            aspect_ratio=aspect_ratio,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt.strip() if negative_prompt.strip() else None
        )
        
        if "imagen-4.0" in model and "fast" not in model:
            config.image_size = image_size

        try:
            result = client.models.generate_images(model=model, prompt=prompt, config=config)
            if not result.generated_images: raise ValueError("No images generated")
            
            tensors = []
            for item in result.generated_images:
                img_data = item.image
                
                if hasattr(img_data, "image_bytes"):
                    pil_img = Image.open(io.BytesIO(img_data.image_bytes))
                elif hasattr(img_data, "convert"):
                    pil_img = img_data
                else:
                    # Fallback for raw bytes
                    pil_img = Image.open(io.BytesIO(img_data))
                
                pil_img = pil_img.convert("RGB")
                tensors.append(torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0))
            
            return (torch.stack(tensors),)
            
        except Exception as e:
            print(f"Google Imagen Error: {e}")
            raise RuntimeError(f"Google Imagen Error: {e}")
            
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

NODE_CLASS_MAPPINGS = {"GoogleImagenNode": GoogleImagenNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GoogleImagenNode": "Google Imagen Generator"}