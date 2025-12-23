import replicate
import os
import requests
import torch
import numpy as np
from PIL import Image
import io

class Flux2Replicate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "api_key": ("STRING", {"default": ""}),
                "model": (["flux-2-max", "flux-2-pro", "flux-2-dev"], {"default": "flux-2-max"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9", "9:21", "2:1", "1:2"], {"default": "1:1"}),
                "output_format": (["webp", "jpg", "png"], {"default": "webp"}),
                "output_quality": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "image/generation"
    
    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        t = tensor.squeeze(0) if len(tensor.shape) == 4 else tensor
        if t.max() <= 1.0:
            t = (t * 255).clamp(0, 255).byte()
        return Image.fromarray(t.cpu().numpy(), 'RGB')
    
    def pil_to_buffer(self, pil_image):
        """Convert PIL Image to BytesIO buffer"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer
    
    def generate_image(self, prompt, api_key, model, aspect_ratio, output_format, output_quality, 
                      image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        try:
            os.environ["REPLICATE_API_TOKEN"] = api_key
            
            input_images = []
            for img in [image_1, image_2, image_3, image_4, image_5]:
                if img is not None:
                    pil_image = self.tensor_to_pil(img)
                    img_buffer = self.pil_to_buffer(pil_image)
                    input_images.append(img_buffer)
            
            replicate_input = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "output_quality": output_quality,
                "input_images": input_images
            }
            
            # Add safety_tolerance for models that support it (flux-2-max and flux-2-pro)
            if model in ["flux-2-max", "flux-2-pro"]:
                replicate_input["safety_tolerance"] = 0
            
            # Run Replicate model
            output = replicate.run(
                f"black-forest-labs/{model}",
                input=replicate_input
            )
            
            # Get URL from output
            output_url = output.url if hasattr(output, 'url') else (
                output if isinstance(output, str) else (
                    output[0] if isinstance(output, list) and output else str(output)
                )
            )
            
            # Download and convert back to tensor
            response = requests.get(output_url, timeout=60)
            response.raise_for_status()
            
            downloaded_image = Image.open(io.BytesIO(response.content))
            if downloaded_image.mode != 'RGB':
                downloaded_image = downloaded_image.convert('RGB')
            
            np_image = np.array(downloaded_image).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(np_image).unsqueeze(0)
            
            return (output_tensor,)
            
        except Exception as e:
            raise RuntimeError(f"Error in Flux.2 generation: {str(e)}")

NODE_CLASS_MAPPINGS = {"Flux2Replicate": Flux2Replicate}
NODE_DISPLAY_NAME_MAPPINGS = {"Flux2Replicate": "Flux.2 (Replicate)"}