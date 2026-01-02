import os
import io
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

class NanoBananaNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (["gemini-3-pro-image-preview", "gemini-2.5-flash-image"],),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"],),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    
    def _convert_tensor_to_bytes(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='PNG')
        return buf.getvalue()

    def generate(self, api_key, model, aspect_ratio, resolution, temperature, top_p, seed,
                 prompt="", system_instruction="", **kwargs):
        
        key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("No API key provided.")
        
        client = genai.Client(api_key=key)
        
        parts = []
        input_images = [kwargs.get(f"image_{i}") for i in range(1, 6)]
        for img in input_images:
            if img is not None:
                img_bytes = self._convert_tensor_to_bytes(img)
                parts.append(types.Part.from_bytes(mime_type="image/png", data=img_bytes))
        
        if prompt.strip():
            parts.append(types.Part.from_text(text=prompt))
            
        if not parts:
            raise ValueError("At least one image or prompt must be provided.")


        img_config_params = {"aspect_ratio": aspect_ratio}
        if "gemini-3-pro" in model:
            img_config_params["image_size"] = resolution
            
        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(**img_config_params),
            system_instruction=system_instruction.strip() if system_instruction.strip() else None
        )
        
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")
        
        try:
            img_data = response.candidates[0].content.parts[0].inline_data.data
            result_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            result_tensor = torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0)
            return (result_tensor,)
            
        except (AttributeError, IndexError, TypeError):
            raise ValueError("API returned a response, but no valid image data was found.")

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

NODE_CLASS_MAPPINGS = {"NanoBananaNode": NanoBananaNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaNode": "Nano Banana"}