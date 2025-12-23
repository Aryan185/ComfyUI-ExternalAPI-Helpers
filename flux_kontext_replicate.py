import replicate
import os
import requests
import torch
import numpy as np
from PIL import Image
import io

class FluxKontextReplicate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Make this a 90s cartoon"}),
                "api_key": ("STRING", {"default": ""}),
                "model": (["flux-kontext-dev", "flux-kontext-max", "flux-kontext-pro"], {"default": "flux-kontext-dev"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9", "9:21", "2:1", "1:2", "match_input_image"], {"default": "match_input_image"}),
                "output_format": (["jpg", "png"], {"default": "jpg"}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6, "step": 1}),
                "seed": ("INT", {"default": 69, "min": 1, "max": 2147483646, "step": 1}),
                "prompt_upsampling": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "image/edit"
    
    def generate_image(self, image, prompt, api_key, model, aspect_ratio, output_format, safety_tolerance, seed, prompt_upsampling):
        try:
            os.environ["REPLICATE_API_TOKEN"] = api_key
            
            # Convert tensor to PIL and save to buffer
            tensor = image.squeeze(0) if len(image.shape) == 4 else image
            if tensor.max() <= 1.0:
                tensor = (tensor * 255).clamp(0, 255).byte()
            pil_image = Image.fromarray(tensor.cpu().numpy(), 'RGB')
            
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Build input dict with all parameters for both models
            replicate_input = {
                "prompt": prompt,
                "input_image": img_buffer,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "safety_tolerance": safety_tolerance,
                "seed": seed,
                "prompt_upsampling": prompt_upsampling
            }
            
            # Run Replicate model with selected model
            output = replicate.run(
                f"black-forest-labs/{model}",
                input=replicate_input
            )
            
            # Get URL from output
            output_url = output if isinstance(output, str) else (output[0] if isinstance(output, list) and output else str(output))
            
            # Download and convert back to tensor
            response = requests.get(output_url, timeout=30)
            response.raise_for_status()
            
            downloaded_image = Image.open(io.BytesIO(response.content))
            if downloaded_image.mode != 'RGB':
                downloaded_image = downloaded_image.convert('RGB')
            
            np_image = np.array(downloaded_image).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(np_image).unsqueeze(0)
            
            return (output_tensor,)
            
        except Exception as e:
            raise RuntimeError(f"Flux Kontext generation failed: {str(e)}") from e

NODE_CLASS_MAPPINGS = {"FluxKontextReplicate": FluxKontextReplicate}
NODE_DISPLAY_NAME_MAPPINGS = {"FluxKontextReplicate": "Flux Kontext (Replicate)"}