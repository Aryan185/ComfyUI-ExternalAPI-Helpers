import os
import io
import requests
import base64
import torch
import numpy as np
from PIL import Image

class GPTImageEditNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "model": (["gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5"],),
                "prompt": ("STRING", {"multiline": True, "default": "Edit this image"}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "n_images": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "mask": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit_image"
    CATEGORY = "image/edit"
    
    def _tensor_to_bytes(self, tensor, is_mask=False):
        if tensor.dim() == 4:
            tensor = tensor[0]
            
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if is_mask:
            pil_image = Image.fromarray(array, mode='L')
            pil_image = Image.eval(pil_image, lambda x: 255 - x)
        else:
            pil_image = Image.fromarray(array)
            
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def edit_image(self, image_1, model, prompt, api_key, background, quality, size, 
                   output_format, output_compression, n_images, mask=None, **kwargs):
        
        key = api_key.strip() or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("GPT Image Edit: No API Key provided.")

        files = []
        input_list = [image_1]
        input_list.extend([kwargs.get(f"image_{i}") for i in range(2, 6)])
        
        for idx, img_tensor in enumerate(input_list):
            if img_tensor is not None:
                img_bytes = self._tensor_to_bytes(img_tensor, is_mask=False)
                files.append(('image[]', (f'input_{idx}.png', img_bytes, 'image/png')))

        if mask is not None:
            mask_bytes = self._tensor_to_bytes(mask, is_mask=True)
            files.append(('mask', ('mask.png', mask_bytes, 'image/png')))

        data = {
            "model": model,
            "prompt": prompt,
            "background": background,
            "n": n_images,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "output_compression": output_compression
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/images/edits",
                headers={"Authorization": f"Bearer {key}"},
                data=data,
                files=files
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"GPT Image Edit Error: {response.status_code} - {response.text}")

            result = response.json()
            if result.get('data'):
                b64_data = result['data'][0]['b64_json']
                image_bytes = base64.b64decode(b64_data)
                
                pil_out = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                out_array = np.array(pil_out).astype(np.float32) / 255.0
                out_tensor = torch.from_numpy(out_array).unsqueeze(0)
                
                return (out_tensor,)
            else:
                raise RuntimeError("GPT Image Edit: API returned no data.")

        except Exception as e:
            raise RuntimeError(f"GPT Image Edit Exception: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('prompt')}-{kwargs.get('model')}"

NODE_CLASS_MAPPINGS = {"GPTImageEditNode": GPTImageEditNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GPTImageEditNode": "GPT Image Edit"}