import os
import io
import requests
import base64
import torch
import numpy as np
from PIL import Image

class GPTImageNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5", "gpt-image-2"],),
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Directly put OpenAI API key or .env variable name (OPENAI_API_KEY)"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "size": (["1024x1024", "1536x1024", "1024x1536", "auto"], {"default": "1024x1024"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "n_images": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "mask": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    
    def _tensor_to_bytes(self, tensor, is_mask=False):
        if tensor.dim() == 4:
            tensor = tensor[0]

        array = (tensor.cpu().numpy() * 255).astype(np.uint8)

        if is_mask:
            array = array.squeeze()
            if array.ndim != 2:
                raise ValueError(f"GPT Image: Mask must be 2D after squeeze, got shape {array.shape}")

            mask_l = Image.fromarray(array, mode='L')
            mask_rgba = mask_l.convert("RGBA")
            inverted = Image.eval(mask_l, lambda x: 255 - x) 
            mask_rgba.putalpha(inverted)
            pil_image = mask_rgba
        else:
            pil_image = Image.fromarray(array)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def generate(self, model, prompt, api_key, background, quality, size, 
                output_format, output_compression, n_images, mask=None, **kwargs):
        
        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("GPT Image: No API Key provided.")

        input_list = []
        for i in range(1, 6):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                input_list.append(img)
        
        is_edit = len(input_list) > 0
        url = "https://api.openai.com/v1/images/edits" if is_edit else "https://api.openai.com/v1/images/generations"
        
        # Standard Payload
        data = {
            "model": model,
            "prompt": prompt,
            "background": background,
            "n": n_images,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "output_compression": output_compression,
        }

        try:
            if is_edit:
                # Use Multipart Form for Edits
                files = [('image[]', (f'input_{i}.png', self._tensor_to_bytes(img), 'image/png')) for i, img in enumerate(input_list)]
                if mask is not None:
                    files.append(('mask', ('mask.png', self._tensor_to_bytes(mask, is_mask=True), 'image/png')))
                
                response = requests.post(url, headers={"Authorization": f"Bearer {key}"}, data=data, files=files)
            else:
                # Use JSON for Generations
                response = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=data)
            
            if response.status_code != 200:
                raise RuntimeError(f"GPT Image API Error: {response.status_code} - {response.text}")

            result = response.json()
            if result.get('data'):
                output_tensors = []
                for img_obj in result['data']:
                    b64_data = img_obj['b64_json']
                    image_bytes = base64.b64decode(b64_data)
                    pil_out = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    out_array = np.array(pil_out).astype(np.float32) / 255.0
                    output_tensors.append(torch.from_numpy(out_array).unsqueeze(0))
                
                try:
                    return (torch.cat(output_tensors, dim=0),)
                except RuntimeError:
                    sizes = [tuple(t.shape[1:]) for t in output_tensors]
                    msg = f"GPT Image: Cannot batch {n_images} images — API returned mixed sizes: {sizes}. Set 'size' to a fixed value (e.g. 1024x1024) instead of 'auto'."
                    print(msg)
                    raise RuntimeError(msg)
            else:
                raise RuntimeError("GPT Image: API returned no data.")

        except Exception as e:
            raise RuntimeError(f"GPT Image Exception: {str(e)}")

NODE_CLASS_MAPPINGS = {"GPTImageNode": GPTImageNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GPTImageNode": "GPT-Image"}