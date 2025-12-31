import os
import io
import numpy as np
import torch
import wave
from PIL import Image
from typing import Optional
from google import genai
from google.genai import types

class GeminiChatNode:
    """ComfyUI Node for Gemini API Chat with optional image and audio input"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {"default": "gemini-2.5-flash"}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1}),
                "thinking": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
                "api_key": ("STRING", {"default": "", "multiline": False})
            },
            "optional": {
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "thinking_budget": ("INT", {"default": -1, "min": -1, "max": 24576, "step": 1}),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "text/generation"
    
    def audio_to_bytes(self, audio):
        if isinstance(audio, dict):
            audio_data = audio.get("waveform")
            sr = audio.get("sample_rate", 44100)
        elif isinstance(audio, (list, tuple)) and len(audio) >= 2:
            audio_data, sr = audio[0], audio[1]
        else:
            raise ValueError(f"Invalid audio input format: {type(audio)}")
        
        if audio_data is None:
            raise ValueError("Missing audio data")
        
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        
        # Convert to WAV bytes
        audio_data = np.squeeze(audio_data)
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sr))
            wav_file.writeframes(audio_data.tobytes())
        
        return wav_buffer.getvalue()
    
    def generate(self, prompt: str, model: str, temperature: float, thinking: bool, seed: int, api_key: str,
                 system_instruction: Optional[str] = None, thinking_budget: int = -1, 
                 image: Optional[torch.Tensor] = None, audio: Optional[dict] = None) -> tuple:   

        key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Error: No API key provided.")

        
        # Initialize client and build parts
        client = genai.Client(api_key=key, http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=3, jitter=10)))
        parts = [types.Part.from_text(text=prompt)]
        
        # Handle image input
        if image is not None:
            img_array = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
            if len(img_array.shape) == 4:
                img_array = img_array[0]
            if img_array.dtype in [np.float32, np.float64]:
                img_array = (img_array * 255).astype(np.uint8)
            
            buffered = io.BytesIO()
            Image.fromarray(img_array).save(buffered, format="PNG")
            parts.append(types.Part.from_bytes(mime_type="image/png", data=buffered.getvalue()))
        
        # Handle audio input
        if audio is not None:
            audio_bytes = self.audio_to_bytes(audio)
            parts.append(types.Part.from_bytes(mime_type="audio/wav", data=audio_bytes))
        
        model_lower = model.lower()
        
        if "gemini-2.0" in model_lower:
            print("Gemini-2.0 models do not support thinking - disabling thinking config")
            final_thinking_budget = None
        elif not thinking:
            final_thinking_budget = 0
            if "gemini-2.5-pro" in model_lower:
                print("Gemini-2.5-Pro cannot have thinking turned off - defaulting thinking budget to -1")
                final_thinking_budget = -1
        else:
            final_thinking_budget = thinking_budget
            if "gemini-2.5-pro" in model_lower and final_thinking_budget == 0:
                print("Gemini-2.5-Pro cannot have thinking turned off - defaulting thinking budget to -1")
                final_thinking_budget = -1
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_mime_type="text/plain"
        )
        
        if "gemini-2.0" not in model_lower:
            config.thinking_config = types.ThinkingConfig(thinking_budget=final_thinking_budget)
        
        if system_instruction and system_instruction.strip():
            config.system_instruction = [types.Part.from_text(text=system_instruction.strip())]
        
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=config
        )
        
        return (response.text,)
            

# Node mappings
NODE_CLASS_MAPPINGS = {"GeminiChatNode": GeminiChatNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiChatNode": "Gemini Chat"}