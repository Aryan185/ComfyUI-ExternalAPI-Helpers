import os
import io
import wave
import torch
from google import genai
from google.genai import types

class GeminiTTS3Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Transcript text. Supports inline tags: <argh>, <breath>, <cackle>, <cheer>, <chuckle>, <chuckles>, <cough>, <cry>, <exhales>, <gasp>, <giggle>, <groan>, <growl>, <grr>, <grunt>, <heavy breath>, <hiss>, <laugh>, <laughter>, <long pause>, <moan>, <pant>, <pff>, <phew>, <scream>, <short pause>, <shout>, <shriek>, <sigh>, <sighs>, <sneeze>, <snicker>, <snort>, <sob>, <throat-clearing>, <tsk>, <whimper>, <whispering>, <whispers>, <yawn>"
                }),
                "voice": ("STRING", {
                    "default": "Bodi", 
                    "multiline": False, 
                    "tooltip": "Voice name from AI Studio (e.g. Bodi, Lumi, Sola, Koda, Nika, Tavi, Kira, Ludo, or legacy celestial names like Puck, Zephyr)"
                }),
                "model": ([
                    "gemini-3.8-flash-tts",
                    "gemini-3.8-flash-lite-tts",
                ], {"default": "gemini-3.8-flash-tts"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "tooltip": "Directly put Gemini API key or .env variable name (GEMINI_API_KEY)"}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "style": ("STRING", {
                    "default": "", 
                    "multiline": True, 
                    "tooltip": "Acting style passed to speech_metadata (e.g. 'Bright, sunny morning greeting with a subtle vocal smile and crisp pacing.')"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/generation"

    def parse_audio_rate(self, mime_type: str) -> int:
        rate = 24000
        if not mime_type:
            return rate
        for param in mime_type.split(";"):
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate = int(param.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
        return rate

    def generate_speech(self, text, voice, model, api_key, seed, temperature, style=""):
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Text input cannot be empty.")

        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("No API key provided.")

        client = genai.Client(api_key=key)

        transcript = clean_text if clean_text.startswith("## Transcript:") else f"## Transcript:\n{clean_text}"

        metadata = None
        if style.strip():
            style_str = style.strip()
            if not style_str.lower().startswith("style:"):
                style_str = f"Style: {style_str}"
            metadata = types.SpeechMetadata(style=style_str)

        part = types.Part(text=transcript, speech_metadata=metadata) if metadata else types.Part(text=transcript)

        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice.strip())
                )
            )
        )

        audio_data = bytearray()
        mime_type = ""

        try:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=[types.Content(role="user", parts=[part])],
                config=config,
            ):
                if chunk.parts is None:
                    continue
                if chunk.parts[0].inline_data and chunk.parts[0].inline_data.data:
                    inline_data = chunk.parts[0].inline_data
                    audio_data.extend(inline_data.data)
                    mime_type = inline_data.mime_type
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")

        if not audio_data:
            raise ValueError("API returned a response, but it contained no audio data.")

        sample_rate = self.parse_audio_rate(mime_type)

        if audio_data.startswith(b"RIFF"):
            with wave.open(io.BytesIO(audio_data), "rb") as wf:
                raw_pcm = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
            waveform = torch.frombuffer(bytearray(raw_pcm), dtype=torch.int16)
        else:
            waveform = torch.frombuffer(bytearray(audio_data), dtype=torch.int16)

        waveform = waveform.to(torch.float32) / 32768.0
        waveform = waveform.unsqueeze(0).unsqueeze(0)

        return ({"waveform": waveform, "sample_rate": sample_rate},)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('text', '')}-{kwargs.get('voice', '')}-{kwargs.get('model', '')}-{kwargs.get('seed', 69)}-{kwargs.get('temperature', 1.0)}-{kwargs.get('style', '')}"

NODE_CLASS_MAPPINGS = {"GeminiTTS3Node": GeminiTTS3Node}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiTTS3Node": "Gemini TTS"}