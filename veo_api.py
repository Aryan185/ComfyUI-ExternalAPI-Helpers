import os
import io
import torch
import numpy as np
from PIL import Image
import tempfile
import uuid
from typing import Optional
from google import genai
from google.genai import types
import cv2


class VeoGeminiVideoGenerator:    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a cat reading a book"}),
                "model": (["veo-2.0-generate-001"], {"default": "veo-2.0-generate-001"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "number_of_videos": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "duration_seconds": ("INT", {"default": 8, "min": 5, "max": 8, "step": 1}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"
    OUTPUT_IS_LIST = (True,)
    
    def pil_to_tensor(self, pil_image):
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(numpy_image).unsqueeze(0)
    
    def video_to_frames(self, video_file):
        temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_video_{uuid.uuid4().hex}.mp4")
        
        try:
            video_file.save(temp_video_path)
            
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
    
    def generate_video(self, seed: int, prompt: str, model: str, aspect_ratio: str, number_of_videos: int,
                      duration_seconds: int, api_key: str,
                      negative_prompt: Optional[str] = None, image: Optional[torch.Tensor] = None):
        
        try:
            key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError("Error: No API key provided. Set GEMINI_API_KEY or provide api_key parameter.")
            
            client = genai.Client(http_options={"api_version": "v1beta"}, api_key=key)
            comfy_seed = seed
            config_params = {
                "aspect_ratio": aspect_ratio,
                "number_of_videos": number_of_videos,
                "duration_seconds": duration_seconds,
            }
            
            if negative_prompt and negative_prompt.strip():
                config_params["negative_prompt"] = negative_prompt.strip()
            
            generation_params = {
                "model": model,
                "prompt": prompt,
                "config": types.GenerateVideosConfig(**config_params),
            }
            
            if image is not None:
                generation_params["image"] = types.Image(
                    image_bytes=self.tensor_to_image_bytes(image),
                    mime_type="image/png"
                )
                print("Image input provided for video generation")
            
            print(f"Starting video generation with model {model}...")
            operation = client.models.generate_videos(**generation_params)
            print(f"Operation started: {operation.name}")
            
            # Wait for operation to complete by polling
            print("Waiting for video generation to complete...")
            import time
            while not operation.done:
                time.sleep(10)
                operation = client.operations.get(operation)
            
            if not operation.response or not operation.response.generated_videos:
                raise Exception("No videos were generated.")
            
            generated_videos = operation.response.generated_videos
            print(f"Generated {len(generated_videos)} video(s).")
            
            all_frames = []
            for n, generated_video in enumerate(generated_videos):
                print(f"Processing video {n+1}/{len(generated_videos)}")
                client.files.download(file=generated_video.video)
                frames_tensor = self.video_to_frames(generated_video.video)
                all_frames.append(frames_tensor)
                print(f"Video {n+1} processed. Extracted {frames_tensor.shape[0]} frames.")
            
            return (all_frames,)
            
        except Exception as e:
            print(f"Error in video generation: {str(e)}")
            return ([torch.zeros((1, 64, 64, 3), dtype=torch.float32)],)


NODE_CLASS_MAPPINGS = {
    "VeoGeminiVideoGenerator": VeoGeminiVideoGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VeoGeminiVideoGenerator": "Veo Video Generator (Gemini API)"
}