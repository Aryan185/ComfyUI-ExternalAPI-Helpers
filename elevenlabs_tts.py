import os
import io
import torch
import requests
import soundfile as sf

class ElevenLabsTTSNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "model_id": ([
                    "eleven_multilingual_v2",
                    "eleven_turbo_v2_5",
                    "eleven_flash_v2_5", 
                    "eleven_flash_v2",
                    "eleven_turbo_v2",
                    "eleven_multilingual_v1",
                    "eleven_v3"
                ],),
                "output_format": ([
                    "mp3_44100_128",
                    "mp3_22050_32",
                    "mp3_44100_32",
                    "mp3_44100_64",
                    "mp3_44100_96",
                    "mp3_44100_192",
                    "pcm_8000",
                    "pcm_16000",
                    "pcm_22050",
                    "pcm_24000",
                    "pcm_44100",
                    "pcm_48000",
                    "ulaw_8000",
                    "alaw_8000",
                    "opus_48000_32",
                    "opus_48000_64",
                    "opus_48000_96",
                    "opus_48000_128",
                    "opus_48000_192"
                ],),
                "voice_id": ("STRING", {"multiline": False, "default": "oPM3trUCF4e0vTcsrMQr"}),
                "stability": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "similarity_boost": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.01}),
                "style": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_speaker_boost": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 40, "min": 0, "max": 4294967294}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "previous_text": ("STRING", {"multiline": True, "default": ""}),
                "next_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/generation"
    
    def generate_speech(self, text, api_key, voice_id, model_id, output_format, 
                       stability, similarity_boost, speed, style, use_speaker_boost, seed,
                       previous_text="", next_text=""):
        
        if not text.strip():
            raise ValueError("Text input cannot be empty.")
        
        key = api_key.strip() or os.environ.get("XI_API_KEY")
        if not key:
            raise ValueError("No API key provided.")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "xi-api-key": key,
            "Content-Type": "application/json"
        }
        
        if model_id == "eleven_v3":
            allowed_stabilities = [0.0, 0.5, 1.0]
            original_stability = stability
            stability = min(allowed_stabilities, key=lambda x: abs(x - original_stability))
            if stability != original_stability:
                print(f"For 'eleven_v3' model, stability value must be one of: [0.0, 0.5, 1.0] (0.0 = Creative, 0.5 = Natural, 1.0 = Robust). Rounding stability to {stability}.")

        voice_settings = {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "speed": speed,
            "style": style,
            "use_speaker_boost": use_speaker_boost
        }
        
        data = {
            "text": text,
            "voice_settings": voice_settings,
            "model_id": model_id,
            "seed": seed,
            "output_format": output_format
        }
        
        if model_id == "eleven_v3":
            if previous_text.strip() or next_text.strip():
                print("Providing previous_text or next_text is not yet supported with the 'eleven_v3' model. Ignoring these inputs.")
        else:
            if previous_text.strip():
                data["previous_text"] = previous_text
            if next_text.strip():
                data["next_text"] = next_text
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"ElevenLabs API Error: {response.status_code}, {response.text}")
        
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
        return f"{kwargs.get('text', '')}-{kwargs.get('voice_id', '')}-{kwargs.get('seed', 40)}"

NODE_CLASS_MAPPINGS = {"ElevenLabsTTSNode": ElevenLabsTTSNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ElevenLabsTTSNode": "ElevenLabs TTS"}