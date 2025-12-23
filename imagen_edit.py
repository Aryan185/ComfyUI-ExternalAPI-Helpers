import os
import json
import tempfile
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from google import genai
from google.genai import types
from google.genai.types import RawReferenceImage, MaskReferenceImage

class GoogleImagenEditNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"multiline": True, "default": "Edit this image"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "project_id": ("STRING", {"multiline": False, "default": ""}),
                "location": (["global", "us-central1", "us-east1", "us-east4", "us-east5", "us-south1", "us-west1", "us-west2", "us-west3", "us-west4", "northamerica-northeast1", "northamerica-northeast2", "southamerica-east1", "southamerica-west1", "africa-south1", "europe-west1", "europe-north1", "europe-west2", "europe-west3", "europe-west4", "europe-west6", "europe-west8", "europe-west9", "europe-west12", "europe-southwest1", "europe-central2", "asia-east1", "asia-east2", "asia-northeast1", "asia-northeast2", "asia-northeast3", "asia-south1", "asia-south2", "asia-southeast1", "asia-southeast2", "australia-southeast1", "australia-southeast2", "me-central1", "me-central2", "me-west1"], {"default": "us-central1"}),
                "service_account": ("STRING", {"multiline": True, "default": ""}),
                "edit_mode": (["EDIT_MODE_INPAINT_INSERTION", "EDIT_MODE_INPAINT_REMOVAL", "EDIT_MODE_OUTPAINT", "EDIT_MODE_BGSWAP"], {"default": "EDIT_MODE_INPAINT_INSERTION"}),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "seed": ("INT", {"default": 69, "min": 1, "max": 2147483646, "step": 1}),
                "base_steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "mask_dilation": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01}),

            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_images",)
    FUNCTION = "edit_image"
    CATEGORY = "image/edit"
    
    def setup_client(self, service_account_json, project_id, location):
        """Setup Vertex AI client with service account JSON content"""
        if not service_account_json.strip():
            raise ValueError("Service account JSON content is required.")
        
        if not project_id.strip():
            raise ValueError("Project ID is required.")
        
        # Validate and write JSON content to temporary file
        try:
            json.loads(service_account_json)  # Validate JSON format
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {str(e)}")
        
        # Create temporary file with JSON content
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write(service_account_json.strip())
        temp_file.close()
        
        # Set credentials path
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
        
        return genai.Client(vertexai=True, project=project_id.strip(), location=location.strip())
    
    def tensor_to_pil(self, tensor):
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)
    
    def mask_to_pil(self, mask):
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        array = (mask.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode='L')
    
    def pil_to_tensor(self, images):
        if not isinstance(images, list):
            images = [images]
        tensors = []
        for image in images:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            array = np.array(image).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(array))
        return torch.stack(tensors)
    
    def edit_image(self, image, mask, prompt, project_id, location, service_account, edit_mode, number_of_images, negative_prompt, seed, base_steps, guidance_scale, mask_dilation):
        try:
            # Initialize Vertex AI client with improved authentication
            client = self.setup_client(service_account, project_id, location)
            
            input_image = self.tensor_to_pil(image[0])
            input_mask = self.mask_to_pil(mask)
            
            img_buffer = BytesIO()
            input_image.save(img_buffer, format='PNG')
            img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            mask_buffer = BytesIO()
            input_mask.save(mask_buffer, format='PNG')
            mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
            
            raw_ref_image = RawReferenceImage(
                reference_image={'image_bytes': img_b64},
                reference_id=0
            )
            
            mask_ref_image = MaskReferenceImage(
                reference_id=1,
                reference_image={'image_bytes': mask_b64},
                config=types.MaskReferenceConfig(
                    mask_mode="MASK_MODE_USER_PROVIDED",
                    mask_dilation=mask_dilation,
                ),
            )
            
            config_params = {
                "edit_mode": edit_mode,
                "number_of_images": number_of_images,
                "include_rai_reason": True,
                "output_mime_type": "image/jpeg",
                "base_steps": base_steps,
                "seed": seed,
                "guidance_scale": guidance_scale,
            }
            
            if negative_prompt.strip():
                config_params["negative_prompt"] = negative_prompt.strip()
            
            response = client.models.edit_image(
                model="imagen-3.0-capability-001",
                prompt=prompt,
                reference_images=[raw_ref_image, mask_ref_image],
                config=types.EditImageConfig(**config_params),
            )
            
            if not response.generated_images:
                raise ValueError("No images generated by the API")
            
            pil_images = []
            for generated_image in response.generated_images:
                image_data = generated_image.image
                
                if hasattr(image_data, 'mode') and hasattr(image_data, 'size'):
                    pil_images.append(image_data)
                elif hasattr(image_data, '_pil_image'):
                    pil_images.append(image_data._pil_image)
                elif hasattr(image_data, 'show'):
                    try:
                        buffer = BytesIO()
                        image_data.save(buffer, format='PNG')
                        buffer.seek(0)
                        pil_images.append(Image.open(buffer))
                    except:
                        pil_images.append(Image.new('RGB', (512, 512), color='gray'))
                elif hasattr(image_data, 'read') or isinstance(image_data, bytes):
                    image_bytes = image_data.read() if hasattr(image_data, 'read') else image_data
                    pil_images.append(Image.open(BytesIO(image_bytes)))
                else:
                    try:
                        pil_images.append(Image.open(image_data))
                    except:
                        pil_images.append(Image.new('RGB', (512, 512), color='gray'))
            
            return (self.pil_to_tensor(pil_images),)
            
        except Exception as e:
            print(f"Google Imagen Edit Error: {str(e)}")
            error_image = Image.new('RGB', (512, 512), color='black')
            return (self.pil_to_tensor([error_image]),)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('prompt', '')}-{kwargs.get('negative_prompt', '')}-{kwargs.get('edit_mode', '')}-{kwargs.get('number_of_images', 1)}-{kwargs.get('seed', 12345)}-{kwargs.get('base_steps', 50)}"

NODE_CLASS_MAPPINGS = {"GoogleImagenEditNode": GoogleImagenEditNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GoogleImagenEditNode": "Google Imagen Edit (Vertex AI only)"}