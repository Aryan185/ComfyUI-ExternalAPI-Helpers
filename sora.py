import time
import io
import torch
import numpy as np
import av
from PIL import Image
from openai import OpenAI

class SoraGen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A calico cat playing a piano on stage"}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (["sora-2", "sora-2-pro"], {"default": "sora-2"}),
                "size": (["720x1280", "1280x720", "1024x1792", "1792x1024"], {"default": "1280x720"}),
                "duration": (["4", "8", "12"], {"default": "4"}),
                "seed": ("INT", {"default": 69, "min": 1, "max": 2147483646, "step": 1}),
            },
            "optional": {"input_image": ("IMAGE",)}
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"
    OUTPUT_IS_LIST = (True, False)

    def generate_video(self, prompt, api_key, model, size, duration, seed, input_image=None):
        client = OpenAI(api_key=api_key)
        api_args = {"prompt": prompt, "model": model, "size": size, "seconds": duration}
        
        img_buf = None
        if input_image is not None:
            img_buf = io.BytesIO()
            # Convert Tensor (Batch, H, W, C) -> Numpy (H, W, C) -> PIL
            arr = (input_image.cpu().numpy()[0] * 255).astype(np.uint8)
            Image.fromarray(arr).save(img_buf, format="JPEG", quality=95)
            img_buf.seek(0)
            api_args["input_reference"] = ("ref.jpg", img_buf, "image/jpeg")

        try:
            job = client.videos.create(**api_args)
            print(f"Job started: {job.id}")
            
            while (status := client.videos.retrieve(job.id)).status not in ["completed", "failed"]:
                time.sleep(4)
            
            if status.status == "failed":
                raise Exception(f"API Error: {status.error.message}")

            video_bytes = io.BytesIO(client.videos.download_content(video_id=job.id).read())
            
            container = av.open(video_bytes)
            frames = []
            for frame in container.decode(video=0):
                img = frame.to_rgb().to_ndarray().astype(np.float32) / 255.0
                frames.append(torch.from_numpy(img).unsqueeze(0))
            container.close()
            
            video_bytes.seek(0) 
            container = av.open(video_bytes)
            audio = None
            
            if container.streams.audio:
                audio_data = [f.to_ndarray() for f in container.decode(audio=0)]
                if audio_data:
                    waveform = torch.from_numpy(np.concatenate(audio_data, axis=1)).float()
                    # Normalize Audio
                    if audio_data[0].dtype == np.int16: waveform /= 32768.0
                    elif audio_data[0].dtype == np.int32: waveform /= 2147483648.0
                    
                    audio = {
                        "waveform": waveform.unsqueeze(0), 
                        "sample_rate": container.streams.audio[0].rate
                    }
            container.close()

            return ([torch.cat(frames, dim=0)], audio)

        finally:
            if img_buf:
                img_buf.close()

NODE_CLASS_MAPPINGS = {"SoraGen": SoraGen}
NODE_DISPLAY_NAME_MAPPINGS = {"SoraGen": "Sora 2 (OpenAI)"}