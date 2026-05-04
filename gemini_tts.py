import os
import torch
from google.genai import Client, types

class GeminiTTSNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Directly put Gemini API key or .env variable name (GEMINI_API_KEY)"}),
                "model": (["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts", "gemini-3.1-flash-tts-preview"],),
                "voice_id": (["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Achernar", "Laomedeia", "Rasalgethi", "Algenib", "Achird", "Pulcherrima", "Gacrux", "Schedar", "Alnilam", "Sulafat", "Sadaltager", "Sadachbia", "Vindemiatrix", "Zubenelgenubi"],),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "audio_profile": ("STRING", {"multiline": False, "default": ""}),
                "style": (["None", "Vocal Smile", "Newscaster", "Whisper", "Empathetic", "Promo/Hype", "Deadpan"], {"default": "None"}),
                "pace": (["None", "Natural", "Rapid Fire", "The Drift", "Staccato"], {"default": "None"}),
                "accent": (["None", "Neutral", "American (Gen)", "American (Valley)", "American (South)", "British (RP)", "Transatlantic", "Australian"], {"default": "None"}),
                "scene": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/generation"

    def generate_speech(self, text, api_key, voice_id, temperature, model, seed,
                        audio_profile="", style="None", pace="None", accent="None", scene=""):

        if not text.strip():
            raise ValueError("Text input cannot be empty.")

        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("No API key provided.")

        client = Client(api_key=key)

        director_parts = []
        if style not in ("", "None"): director_parts.append(f"Style: {style}")
        if pace not in ("", "None"): director_parts.append(f"Pace: {pace}")
        if accent not in ("", "None"): director_parts.append(f"Accent: {accent}")

        has_director = any([audio_profile.strip(), director_parts, scene.strip()])

        if has_director:
            sections = ["Read the following transcript based on the audio profile and director's note."]
            if audio_profile.strip():
                sections.append(f"# Audio Profile\n{audio_profile.strip()}")
            if director_parts:
                sections.append(f"# Director's note\n{'. '.join(director_parts)}.")
            if scene.strip():
                sections.append(f"## Scene:\n{scene.strip()}")
            sections.append(f"## Transcript:\n{text.strip()}")
            prompt = "\n\n".join(sections)
        else:
            prompt = f"## Transcript:\n{text.strip()}"

        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_id)
                )
            )
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")

        try:
            audio_bytes = response.candidates[0].content.parts[0].inline_data.data
        except (AttributeError, IndexError, TypeError):
            raise ValueError("API returned a response, but it contained no audio data.")

        waveform = torch.frombuffer(bytearray(audio_bytes), dtype=torch.int16)
        waveform = waveform.to(torch.float32) / 32768.0
        waveform = waveform.unsqueeze(0).unsqueeze(0)

        return ({"waveform": waveform, "sample_rate": 24000},)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('text', '')}-{kwargs.get('voice_id', '')}-{kwargs.get('temperature', 1.0)}-{kwargs.get('model', '')}-{kwargs.get('seed', 69)}-{kwargs.get('audio_profile', '')}-{kwargs.get('style', '')}-{kwargs.get('pace', '')}-{kwargs.get('accent', '')}-{kwargs.get('scene', '')}"

NODE_CLASS_MAPPINGS = {"GeminiTTSNode": GeminiTTSNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiTTSNode": "Gemini TTS"}