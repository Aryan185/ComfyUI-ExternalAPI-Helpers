import time
import os
import io
import tempfile
import torch
import numpy as np
import av
from PIL import Image
from google import genai
from google.genai import types

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
                    "veo-2.0-generate-001", "veo-2.0-generate-exp", "veo-2.0-generate-preview",
                    "veo-3.0-generate-001", "veo-3.0-fast-generate-001", 
                    "veo-3.1-generate-001", "veo-3.1-fast-generate-001"
                ], {"default": "veo-3.0-generate-001"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration_seconds": ("INT", {"default": 4, "min": 4, "max": 8, "step": 1}),
                "seed": ("INT", {"default": 69, "min": 1, "max": 2147483646, "step": 1}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "fps": (["24"], {"default": "24"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"
    OUTPUT_IS_LIST = (True, False)
    
    def generate_video(self, prompt, project_id, location, service_account, model, resolution, aspect_ratio, 
                      duration_seconds, seed, generate_audio, fps, negative_prompt=None, 
                      first_frame=None, last_frame=None):
        
        creds_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        creds_file.write(service_account.strip())
        creds_file.close()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file.name

        try:
            client = genai.Client(vertexai=True, project=project_id, location=location)
            
            config = types.GenerateVideosConfig(
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
                generate_audio=generate_audio,
                fps=int(fps),
                seed=seed if seed != -1 else None,
                negative_prompt=negative_prompt.strip() if negative_prompt else None
            )

            # Helper: Tensor -> Bytes
            def tensor_to_bytes(t):
                arr = (t.cpu().numpy()[0] * 255).astype(np.uint8) if len(t.shape) == 4 else (t.cpu().numpy() * 255).astype(np.uint8)
                b = io.BytesIO()
                Image.fromarray(arr).save(b, format="PNG")
                return b.getvalue()

            gen_kwargs = {"model": model, "prompt": prompt, "config": config}
            
            if first_frame is not None:
                gen_kwargs["image"] = types.Image(image_bytes=tensor_to_bytes(first_frame), mime_type="image/png")
            if last_frame is not None:
                setattr(config, 'last_frame', types.Image(image_bytes=tensor_to_bytes(last_frame), mime_type="image/png"))

            op = client.models.generate_videos(**gen_kwargs)
            print(f"Veo Operation: {op.name}")
            
            while not op.done:
                time.sleep(5)
                op = client.operations.get(op)
            
            if op.error: raise Exception(f"Veo Error: {op.error}")
            if not op.result.generated_videos: raise Exception("No videos generated")

            video_bytes = io.BytesIO(op.result.generated_videos[0].video.video_bytes)
            
            container = av.open(video_bytes)
            frames = []
            for frame in container.decode(video=0):
                img = frame.to_rgb().to_ndarray().astype(np.float32) / 255.0
                frames.append(torch.from_numpy(img).unsqueeze(0))
            container.close()

            # Decode Audio
            audio = None
            if generate_audio:
                video_bytes.seek(0)
                container = av.open(video_bytes)
                if container.streams.audio:
                    audio_data = [f.to_ndarray() for f in container.decode(audio=0)]
                    if audio_data:
                        waveform = torch.from_numpy(np.concatenate(audio_data, axis=1)).float()
                        # Normalize 16/32-bit audio
                        if audio_data[0].dtype == np.int16: waveform /= 32768.0
                        elif audio_data[0].dtype == np.int32: waveform /= 2147483648.0
                        
                        audio = {
                            "waveform": waveform.unsqueeze(0),
                            "sample_rate": container.streams.audio[0].rate
                        }
                container.close()

            if not frames: raise Exception("Failed to decode video frames")
            
            return ([torch.cat(frames, dim=0)], audio)

        finally:
            # Cleanup Auth File
            if os.path.exists(creds_file.name):
                os.remove(creds_file.name)

NODE_CLASS_MAPPINGS = {"VeoVertexVideoGenerator": VeoVertexVideoGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"VeoVertexVideoGenerator": "Veo (Vertex AI)"}