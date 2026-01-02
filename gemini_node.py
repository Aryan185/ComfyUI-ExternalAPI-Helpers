import os
import io
import wave
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

class GeminiChatNode:
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
                "thinking_budget": ("INT", {"default": 0, "min": -1, "max": 24576, "step": 1}),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "text/generation"
    
    def generate(self, prompt, model, temperature, thinking, seed, api_key,
                 system_instruction=None, thinking_budget=-1, image=None, audio=None):   

        key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
        if not key: raise ValueError("Error: No API key provided.")
        
        client = genai.Client(api_key=key, http_options={'api_version': 'v1beta'})
        parts = [types.Part.from_text(text=prompt)]
        
        if image is not None:
            arr = (image[0].cpu().numpy() * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            parts.append(types.Part.from_bytes(mime_type="image/png", data=buf.getvalue()))
        
        if audio is not None:
            wf = audio.get("waveform") if isinstance(audio, dict) else audio[0]
            sr = audio.get("sample_rate", 44100) if isinstance(audio, dict) else audio[1]
            
            wf = wf.cpu().numpy() if isinstance(wf, torch.Tensor) else wf
            if wf.ndim > 1: wf = wf.mean(axis=0) if wf.shape[0] > 1 else wf.squeeze()
            wf_int16 = (np.clip(wf, -1, 1) * 32767).astype(np.int16)
            
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes(wf_int16.tobytes())
            parts.append(types.Part.from_bytes(mime_type="audio/wav", data=buf.getvalue()))
        
        model_lower = model.lower()
        t_config = None

        if "gemini-2.0" in model_lower:
            print("Gemini-2.0 models do not support thinking - disabling thinking config")
        else:
            final_budget = 0 # Default disabled
            
            if not thinking:
                if "gemini-2.5-pro" in model_lower or "gemini-3-pro-preview" in model_lower:
                    print("Pro models cannot have thinking turned off - defaulting thinking budget to -1")
                    final_budget = -1
            else:
                final_budget = thinking_budget
                if ("gemini-2.5-pro" in model_lower or "gemini-3-pro-preview" in model_lower) and final_budget == 0:
                    print("Pro models cannot have thinking turned off - defaulting thinking budget to -1")
                    final_budget = -1
            
            t_config = types.ThinkingConfig(thinking_budget=final_budget)

        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            system_instruction=system_instruction.strip() if system_instruction else None,
            thinking_config=t_config
        )
        
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=config
        )
        
        return (response.text,)

NODE_CLASS_MAPPINGS = {"GeminiChatNode": GeminiChatNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiChatNode": "Gemini Chat"}