import os
import torch
from google.genai import Client, types

class GeminiTTSNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"],),
                "voice_id": (["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Achernar", "Laomedeia", "Rasalgethi", "Algenib", "Achird", "Pulcherrima", "Gacrux", "Schedar", "Alnilam", "Sulafat", "Sadaltager", "Sadachbia", "Vindemiatrix", "Zubenelgenubi"],),                
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/generation"
    
    def generate_speech(self, text, api_key, voice_id, temperature, model, seed, system_prompt=""):
        
        if not text.strip():
            raise ValueError("Text input cannot be empty.")
        
        key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("No API key provided.")
        
        client = Client(api_key=key)
        
        final_prompt = text
        if system_prompt.strip():
            final_prompt = f"{system_prompt.strip()}\n\n{text}"

        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_id)
            )
        )
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        )
        
        try:
            response = client.models.generate_content(
                model=model,
                contents=final_prompt,
                config=config
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")

        try:
            inline_data = response.candidates[0].content.parts[0].inline_data
            audio_bytes = inline_data.data
        except (AttributeError, IndexError, TypeError):
            raise ValueError("API returned a response, but it contained no audio data.")

        waveform = torch.frombuffer(bytearray(audio_bytes), dtype=torch.int16)
        waveform = waveform.to(torch.float32) / 32768.0
        waveform = waveform.unsqueeze(0).unsqueeze(0)
        
        return ({"waveform": waveform, "sample_rate": 24000},)
    
    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

NODE_CLASS_MAPPINGS = {"GeminiTTSNode": GeminiTTSNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiTTSNode": "Gemini TTS"}