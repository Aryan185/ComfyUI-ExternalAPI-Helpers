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
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Directly put Gemini API key or .env variable name (GEMINI_API_KEY)"}),
                "model": (["gemini-3-pro-image-preview", "gemini-2.5-flash-image", "gemini-3.1-flash-image-preview"],),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"],),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "google_search": ("BOOLEAN", {"default": False}),
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

    def generate(self, api_key, model, aspect_ratio, resolution, temperature, top_p, google_search, seed,
                 prompt="", system_instruction="", **kwargs):

        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("No API key provided.")

        client = genai.Client(api_key=key)

        parts = []
        for i in range(1, 6):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                parts.append(types.Part.from_bytes(mime_type="image/png", data=self._convert_tensor_to_bytes(img)))

        if prompt.strip():
            parts.append(types.Part.from_text(text=prompt))

        if not parts:
            raise ValueError("At least one image or prompt must be provided.")

        tools = None
        if google_search:
            if "gemini-2.5" in model:
                print(f"Ignoring google_search: {model} does not support it.")
            else:
                tools = [types.Tool(googleSearch=types.GoogleSearch())]

        img_config_params = {"aspect_ratio": aspect_ratio}
        if "gemini-3-pro" in model:
            img_config_params["image_size"] = resolution

        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(**img_config_params),
            system_instruction=system_instruction.strip() if system_instruction.strip() else None,
            tools=tools
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
            return (torch.from_numpy(np.array(Image.open(io.BytesIO(img_data)).convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0),)
        except (AttributeError, IndexError, TypeError):
            raise ValueError("API returned a response, but no valid image data was found.")

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed


NODE_CLASS_MAPPINGS = {"NanoBananaNode": NanoBananaNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaNode": "Nano Banana"}