import os
import io
import torch
import soundfile as sf
from openai import OpenAI

class GroqOrpheusTTSNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "model": (["canopylabs/orpheus-v1-english", "canopylabs/orpheus-arabic-saudi"], {"default": "canopylabs/orpheus-v1-english"}),
                "voice": (["autumn", "diana", "hannah", "austin", "daniel", "troy", "abdullah", "fahad", "sultan", "lulwa", "noura", "aisha"], {"default": "troy"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483646, "step": 1}),
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Directly put Groq API key or .env variable name (GROQ_API_KEY)"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/generation"

    def generate_speech(self, text, model, voice, speed, seed, api_key):

        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("No API key provided.")

        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="wav",
            speed=speed,
        )

        waveform, sample_rate = sf.read(io.BytesIO(response.content), dtype='float32')
        waveform = torch.from_numpy(waveform)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)
    
    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed


NODE_CLASS_MAPPINGS = {"GroqOrpheusTTSNode": GroqOrpheusTTSNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GroqOrpheusTTSNode": "Groq Orpheus TTS"}