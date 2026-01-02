import os
import io
import base64
import tempfile
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types


class GoogleImagenEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"multiline": True, "default": "Edit this image"}),
                "project_id": ("STRING", {"multiline": False, "default": ""}),
                "location": (["global", "us-central1", "us-east1", "us-east4", "us-east5", "us-south1", "us-west1", "us-west2", "us-west3", "us-west4", "northamerica-northeast1", "northamerica-northeast2", "southamerica-east1", "southamerica-west1", "africa-south1", "europe-west1", "europe-north1", "europe-west2", "europe-west3", "europe-west4", "europe-west6", "europe-west8", "europe-west9", "europe-west12", "europe-southwest1", "europe-central2", "asia-east1", "asia-east2", "asia-northeast1", "asia-northeast2", "asia-northeast3", "asia-south1", "asia-south2", "asia-southeast1", "asia-southeast2", "australia-southeast1", "australia-southeast2", "me-central1", "me-central2", "me-west1"], {"default": "us-central1"}),
                "service_account": ("STRING", {"multiline": True, "default": ""}),
                "edit_mode": (["EDIT_MODE_INPAINT_INSERTION", "EDIT_MODE_INPAINT_REMOVAL", "EDIT_MODE_OUTPAINT", "EDIT_MODE_BGSWAP"], {"default": "EDIT_MODE_INPAINT_INSERTION"}),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "seed": ("INT", {"default": 69, "min": 1, "max": 2147483646, "step": 1}),
                "base_steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "mask_dilation": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_images",)
    FUNCTION = "edit_image"
    CATEGORY = "image/edit"
    
    def edit_image(self, image, mask, prompt, project_id, location, service_account, 
                  edit_mode, number_of_images, seed, base_steps, guidance_scale, mask_dilation, negative_prompt=""):
        
        creds_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        creds_file.write(service_account.strip())
        creds_file.close()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file.name
        
        try:
            client = genai.Client(vertexai=True, project=project_id.strip(), location=location.strip())
            
            def to_b64(img):
                b = io.BytesIO()
                img.save(b, format='PNG')
                return base64.b64encode(b.getvalue()).decode('utf-8')

            img_pil = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
            
            mask_np = mask.cpu().numpy()
            if mask_np.ndim == 3: mask_np = mask_np[0]
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

            config_dict = {
                "edit_mode": edit_mode,
                "number_of_images": number_of_images,
                "base_steps": base_steps,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "output_mime_type": "image/jpeg",
                "include_rai_reason": True,
            }
            
            if negative_prompt.strip():
                config_dict["negative_prompt"] = negative_prompt.strip()
            
            response = client.models.edit_image(
                model="imagen-3.0-capability-001",
                prompt=prompt,
                reference_images=[
                    types.RawReferenceImage(reference_id=0, reference_image={'image_bytes': to_b64(img_pil)}),
                    types.MaskReferenceImage(reference_id=1, reference_image={'image_bytes': to_b64(mask_pil)},
                                             config=types.MaskReferenceConfig(mask_mode="MASK_MODE_USER_PROVIDED", mask_dilation=mask_dilation))
                ],
                config=types.EditImageConfig(**config_dict)
            )

            if not response.generated_images: raise ValueError("No images generated")

            output_tensors = []
            for item in response.generated_images:
                img_bytes = item.image.image_bytes
                res_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                output_tensors.append(torch.from_numpy(np.array(res_img).astype(np.float32) / 255.0))
            
            return (torch.stack(output_tensors),)

        except Exception as e:
            print(f"Google Imagen Edit Error: {e}")
            raise RuntimeError(f"Google Imagen Edit Error: {e}")
        finally:
            if os.path.exists(creds_file.name): os.remove(creds_file.name)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

NODE_CLASS_MAPPINGS = {"GoogleImagenEditNode": GoogleImagenEditNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GoogleImagenEditNode": "Google Imagen Edit (Vertex AI only)"}