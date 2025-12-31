import os
import io
import torch
import requests
import soundfile as sf

class OpenAITTSNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "model": ([
                    "gpt-4o-mini-tts",
                    "tts-1",
                    "tts-1-hd"
                ],),
                "voice": ([
                    "alloy",
                    "ash",
                    "coral",
                    "echo",
                    "fable",
                    "onyx",
                    "nova",
                    "sage",
                    "shimmer"
                ],),
                "response_format": ([
                    "mp3",
                    "opus",
                    "aac",
                    "flac",
                    "wav",
                    "pcm"
                ],),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "instructions": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/generation"
    
    def generate_speech(self, text, api_key, model, voice, response_format, speed, instructions=""):
        
        if not text.strip():
            raise ValueError("Text input cannot be empty.")
        
        key = api_key.strip() or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("No API key provided. Set OPENAI_API_KEY environment variable or provide it in the node.")
        
        # Check if instructions are used with incompatible models
        if instructions.strip() and model in ["tts-1", "tts-1-hd"]:
            raise ValueError(f"Instructions parameter is not supported with model '{model}'. Please use 'gpt-4o-mini-tts' instead.")
        
        url = "https://api.openai.com/v1/audio/speech"
        
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed
        }
        
        # Add instructions only if provided and model supports it
        if instructions.strip() and model not in ["tts-1", "tts-1-hd"]:
            data["instructions"] = instructions
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API Error: {response.status_code}, {response.text}")
        
        # Decode audio with soundfile
        audio_buffer = io.BytesIO(response.content)
        waveform, sample_rate = sf.read(audio_buffer, dtype='float32')
        waveform = torch.from_numpy(waveform)
        
        # Ensure correct shape [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
        
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('text', '')}-{kwargs.get('voice', '')}-{kwargs.get('model', '')}"

NODE_CLASS_MAPPINGS = {"OpenAITTSNode": OpenAITTSNode}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenAITTSNode": "OpenAI Text-to-Speech"}