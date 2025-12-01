import time
import os
import io
import json
import tempfile
import torch
import numpy as np
from PIL import Image
import uuid
from typing import Optional
from google import genai
from google.genai import types
import cv2


class VeoVertexVideoGenerator:    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a cat reading a book"}),
                "project_id": ("STRING", {"multiline": False, "default": ""}),
                "location": ([
                    "global", "us-central1", "us-east1", "us-east4", "us-east5", "us-south1", 
                    "us-west1", "us-west2", "us-west3", "us-west4", 
                    "northamerica-northeast1", "northamerica-northeast2", 
                    "southamerica-east1", "southamerica-west1", "africa-south1", 
                    "europe-west1", "europe-north1", "europe-west2", "europe-west3", 
                    "europe-west4", "europe-west6", "europe-west8", "europe-west9", 
                    "europe-west12", "europe-southwest1", "europe-central2", 
                    "asia-east1", "asia-east2", "asia-northeast1", "asia-northeast2", 
                    "asia-northeast3", "asia-south1", "asia-south2", "asia-southeast1", 
                    "asia-southeast2", "australia-southeast1", "australia-southeast2", 
                    "me-central1", "me-central2", "me-west1"
                ], {"default": "us-central1"}),
                "service_account": ("STRING", {"multiline": True, "default": ""}),
                "model": ([
                    "veo-2.0-generate-001",
                    "veo-2.0-generate-exp",
                    "veo-2.0-generate-preview",
                    "veo-3.0-generate-001",
                    "veo-3.0-fast-generate-001",
                    "veo-3.1-generate-001",
                    "veo-3.1-fast-generate-001"
                ], {"default": "veo-3.0-generate-001"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration_seconds": ("INT", {"default": 4, "min": 4, "max": 8, "step": 1}),
                "seed": ("INT", {"default": 69, "min": 1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"
    OUTPUT_IS_LIST = (True,)
    
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
        
        return genai.Client(
            vertexai=True, 
            project=project_id.strip(), 
            location=location.strip()
        )
    
    def pil_to_tensor(self, pil_image):
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(numpy_image).unsqueeze(0)
    
    def video_to_frames(self, video_bytes):
        temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_video_{uuid.uuid4().hex}.mp4")
        
        try:
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
            
            cap = cv2.VideoCapture(temp_video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor_frame = self.pil_to_tensor(Image.fromarray(frame_rgb))
                frames.append(tensor_frame)
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            return torch.cat(frames, dim=0)
            
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    def tensor_to_image_bytes(self, image_tensor):
        """Convert ComfyUI image tensor to bytes"""
        img_array = image_tensor.cpu().numpy() if isinstance(image_tensor, torch.Tensor) else image_tensor
        
        if len(img_array.shape) == 4:
            img_array = img_array[0]
        
        if img_array.dtype in [np.float32, np.float64]:
            img_array = (img_array * 255).astype(np.uint8)
        
        buffered = io.BytesIO()
        Image.fromarray(img_array).save(buffered, format="PNG")
        return buffered.getvalue()
    
    def generate_video(self, prompt: str, project_id: str, location: str, 
                      service_account: str, model: str, resolution: str, aspect_ratio: str, 
                      duration_seconds: int, seed: int,
                      negative_prompt: Optional[str] = None, 
                      first_frame: Optional[torch.Tensor] = None,
                      last_frame: Optional[torch.Tensor] = None):
        
        # Initialize Vertex AI client
        client = self.setup_client(service_account, project_id, location)
        
        # Configure video generation
        config_params = {
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration_seconds,
        }
        
        if seed != -1:
            config_params["seed"] = seed
        
        if negative_prompt and negative_prompt.strip():
            config_params["negative_prompt"] = negative_prompt.strip()
        
        video_config = types.GenerateVideosConfig(**config_params)
        
        # Prepare generation parameters
        generation_params = {
            "model": model,
            "prompt": prompt,
            "config": video_config,
        }
        
        # Handle first frame image
        if first_frame is not None:
            image_bytes = self.tensor_to_image_bytes(first_frame)
            generation_params["image"] = types.Image(
                image_bytes=image_bytes,
                mime_type="image/png"
            )
            print("First frame image provided for video generation")
        
        # Handle last frame image (dynamic attribute for preview SDK)
        if last_frame is not None:
            last_image_bytes = self.tensor_to_image_bytes(last_frame)
            last_frame_img = types.Image(
                image_bytes=last_image_bytes,
                mime_type="image/png"
            )
            setattr(video_config, 'last_frame', last_frame_img)
            print("Last frame image provided for video generation")
        
        print(f"Starting video generation with model {model}...")
        operation = client.models.generate_videos(**generation_params)
        print(f"Operation started: {operation.name}")
        
        # Poll for completion
        print("Waiting for video generation to complete...")
        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)
            print(".", end="", flush=True)
        print("")
        
        # Check for errors
        if operation.error:
            raise Exception(f"Operation failed: {operation.error}")
        
        # Retrieve video
        if not operation.result or not operation.result.generated_videos:
            raise Exception("No videos were generated.")
        
        video_result = operation.result.generated_videos[0].video
        
        if not video_result.video_bytes:
            raise Exception("No video bytes returned from API")
        
        print("Video generated successfully. Extracting frames...")
        frames_tensor = self.video_to_frames(video_result.video_bytes)
        print(f"Extracted {frames_tensor.shape[0]} frames.")
        
        return ([frames_tensor],)


NODE_CLASS_MAPPINGS = {
    "VeoVertexVideoGenerator": VeoVertexVideoGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VeoVertexVideoGenerator": "Veo (Vertex AI)"
}