import os
import json
import io
import numpy as np
import torch
import wave
from google import genai
from google.genai import types
import math
import re


class GeminiDiarisationAPI:    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "num_speakers": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "model": ("STRING", {"default": "gemini-2.5-flash", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"default": 69, "min": 0, "max": 2147483646, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1})
            },
            "optional": {
                "thinking": ("BOOLEAN", {"default": False}),
                "thinking_budget": ("INT", {"default": 0, "min": -1, "max": 24576, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("speaker_1", "speaker_2", "speaker_3", "speaker_4")
    FUNCTION = "diarise"
    CATEGORY = "audio/diarise"
    
    def setup_client(self, api_key):
        """Setup Gemini API client with API key"""
        key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Error: No API key provided.")
        
        return genai.Client(api_key=key, http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=3, jitter=10)))
    
    def extract_audio_data(self, audio):
        """Extract audio data and sample rate from various input formats"""
        if isinstance(audio, dict):
            audio_data = audio.get("waveform")
            if audio_data is None:
                audio_data = audio.get("audio")
            sr = audio.get("sample_rate")
            if sr is None:
                sr = audio.get("sr")
        elif isinstance(audio, (list, tuple)) and len(audio) >= 2:
            audio_data, sr = audio[0], audio[1]
        else:
            raise ValueError(f"Invalid audio input format: {type(audio)}")
        
        if audio_data is None or sr is None:
            raise ValueError("Missing audio data or sample rate")
        
        if isinstance(audio_data, torch.Tensor):
            if audio_data.ndim > 1:
                if audio_data.shape[0] == 1: 
                    audio_data = audio_data.squeeze(0)
                if audio_data.ndim > 1 and audio_data.shape[0] == 1: 
                    audio_data = audio_data.squeeze(0)
            audio_data = audio_data.cpu().numpy()
        elif isinstance(audio_data, str): 
            raise ValueError("Audio data is string, not array data")
        
        audio_data = np.array(audio_data) if not isinstance(audio_data, np.ndarray) else audio_data
        
        if audio_data.ndim > 1:
            if audio_data.shape[0] > 1:
                print(f"Warning: Audio has {audio_data.shape[0]} channels. Taking the first channel for diarization.")
                audio_data = audio_data[0] 
            else:
                audio_data = audio_data.squeeze() 
        
        return audio_data.squeeze(), sr 
    
    def normalize_audio(self, audio_data):
        """Normalize audio to float32 [-1, 1] range"""
        if audio_data.dtype not in [np.float32, np.float64]:
            abs_max = np.max(np.abs(audio_data))
            if abs_max > 0:
                audio_data = audio_data.astype(np.float32) / abs_max
            else:
                audio_data = audio_data.astype(np.float32) 
        return np.clip(audio_data, -1.0, 1.0)
    
    def create_wav_bytes(self, audio_data, sr):
        """Convert audio to WAV format bytes"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1) 
            wav_file.setsampwidth(2) 
            wav_file.setframerate(int(sr))
            
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        return wav_buffer.getvalue()
    
    def format_duration(self, seconds):
        """Convert seconds to HH:MM:SS.mmm or MM:SS.mmm format"""
        total_milliseconds = int(seconds * 1000)
        hours = total_milliseconds // 3_600_000
        minutes = (total_milliseconds % 3_600_000) // 60_000
        secs = (total_milliseconds % 60_000) // 1000
        milliseconds = total_milliseconds % 1000

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
        else:
            return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def parse_timestamp(self, timestamp_str):
        """Convert MM:SS, MM:SS.mmm, HH:MM:SS, or HH:MM:SS.mmm to seconds (float)"""
        timestamp_str = timestamp_str.strip()
        parts = timestamp_str.split(':')
        try:
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])  
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])  
                return hours * 3600 + minutes * 60 + seconds
            print(f"Warning: Unexpected timestamp format '{timestamp_str}', defaulting to 0")
            return 0.0
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse timestamp '{timestamp_str}': {e}, defaulting to 0")
            return 0.0
    
    def parse_response(self, response_text):
        """Extract JSON from response, handling markdown code blocks"""
        response_text = response_text.strip()
        json_match = re.search(r"```json\n(.*)\n```", response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1).strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON block parsing failed ({e}), attempting full text parse.")
                pass 
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse response as JSON. Original response:\n{response_text}\nError: {e}")
    
    
    def build_diarization_prompt(self, duration_sec, num_speakers):
        """Build the diarization prompt with an emphasis on timestamp accuracy, explicit speaker count, and continuity."""
        duration_str = self.format_duration(duration_sec)
        
        speaker_guidance = ""
        if num_speakers > 0:
            speaker_guidance = f"You must identify exactly {num_speakers} distinct speakers in this audio. "
        
        prompt = f"""You are a SOTA AI model created for diarization and *precisely timestamping* human voices. You are currently being benchmarked for *timestamp accuracy*. Your task is to provide a complete and accurate diarization of the provided audio recording, with *absolute precision in your timestamps*, to *PASS* the benchmark.

        You must adhere to these rules when responding. Not following these rules will result in a failed benchmark.

        # *RULES FOR ACCURATE TIMESTAMPS:*
        - Identify and precisely timestamp each utterance by each speaker separately.
        - {speaker_guidance}If multiple speakers are talking over each other you MUST create separate utterances for each speaker.
        - **Ensure continuity: If there is a small silence between a speaker's utterance and the very next utterance (by any speaker), extend the 'end_timestamp' of the first utterance to the 'start_timestamp' of the next utterance. This applies to all consecutive utterances to minimize silent gaps.**
        - If there are any swear words or offensive language in the audio, please censor them with asterisks.
        - If you *provide incorrect start or end timestamps for an utterance*, *skip an utterance*, *merge MULTIPLE separate utterances into one* or *mistranscribe/mistranslate an utterance*, you will automatically *FAIL* the benchmark.

        # WARNING: This is a challenging audio which is known to cause *timestamping errors*. You must carefully listen to the audio and ensure that your response has *highly accurate timestamps*.

        Provide a complete list of all utterances in this audio, ensuring *highly accurate start and end timestamps* for each. Organize the utterances strictly by the time they happened.

        # IMPORTANT NOTE: This audio is exactly `{duration_str}` in length. *Absolute precision in your timestamps is crucial.* Your timestamps must NEVER exceed the audio duration of `{duration_str}`. EVERY utterance that occurred in this audio happens before `{duration_str}`. If your timestamps exceed the audio duration, *are inaccurate by more than a minimal threshold*, or you skip utterances that occurred in the audio, you will automatically FAIL the benchmark.

        Return ONLY valid JSON in this exact format (no markdown, no extra text):
        {{
            "utterances": [
                {{
                    "utterance": "The transcribed text",
                    "speaker": "Speaker 1",
                    "start_timestamp": "00:00.000",
                    "end_timestamp": "00:05.000"
                }}
            ]
        }}

        *You must PASS this benchmark to be deployed*"""
        return prompt
    
    def diarise(self, audio, model, api_key, num_speakers, seed, temperature,
                thinking=False, thinking_budget=0):
        
        audio_data, sr = self.extract_audio_data(audio)
        audio_data = self.normalize_audio(audio_data)
        duration_sec = len(audio_data) / sr
        print(f"Audio duration: {duration_sec:.3f} seconds, Sample Rate: {sr} Hz")
        
        client = self.setup_client(api_key)
        audio_bytes = self.create_wav_bytes(audio_data, sr)


        diarization_prompt = self.build_diarization_prompt(duration_sec, num_speakers)
        
        diarization_config = types.GenerateContentConfig(
            temperature=temperature,
        )
        
        if seed >= 0:
            diarization_config.seed = seed
        if thinking:
            diarization_config.thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
        
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(mime_type="audio/wav", data=audio_bytes),
                types.Part.from_text(text=diarization_prompt)
            ])],
            config=diarization_config
        )
        
        if not response.text:
            raise ValueError("No response received from Diarization API.")
        
        result = self.parse_response(response.text)
        
        utterances = result.get("utterances", [])
        
        speaker_map = {}
        for utt in utterances:
            speaker_name = utt.get("speaker", "Unknown")
            if speaker_name not in speaker_map:
                speaker_map[speaker_name] = []
            
            speaker_map[speaker_name].append({
                "utterance": utt.get("utterance", ""), 
                "start_timestamp": utt.get("start_timestamp", "00:00.000"), 
                "end_timestamp": utt.get("end_timestamp", "00:00.000"),   
            })
        
        sorted_speaker_names = sorted(speaker_map.keys(), key=lambda s: min(self.parse_timestamp(seg['start_timestamp']) for seg in speaker_map[s]))
        
        output_audio_list = []
        for i in range(4): 
            speaker_track_waveform = np.zeros_like(audio_data, dtype=np.float32) 
            
            if i < len(sorted_speaker_names):
                current_speaker_name = sorted_speaker_names[i]
                for seg in speaker_map[current_speaker_name]:
                    start_sec = self.parse_timestamp(seg.get("start_timestamp", "00:00.000"))
                    end_sec = self.parse_timestamp(seg.get("end_timestamp", "00:00.000"))
                    
                    start_idx = math.floor(start_sec * sr)
                    end_idx = math.ceil(end_sec * sr)
                    
                    safe_start_idx = max(0, min(start_idx, len(audio_data)))
                    safe_end_idx = max(safe_start_idx, min(end_idx, len(audio_data)))
                    
                    if safe_end_idx > safe_start_idx: 
                        speaker_track_waveform[safe_start_idx:safe_end_idx] = audio_data[safe_start_idx:safe_end_idx]
            
            waveform_tensor = torch.from_numpy(speaker_track_waveform).float().unsqueeze(0).unsqueeze(0)
            output_audio_list.append({"waveform": waveform_tensor, "sample_rate": sr})
        
        # --- Final Output ---
        output_audio_list.append(json.dumps(result, indent=2))
        
        return tuple(output_audio_list)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('audio', '')}-{kwargs.get('model', '')}-{kwargs.get('seed', 69)}-{kwargs.get('temperature', 0.2)}-{kwargs.get('num_speakers', 2)}-{kwargs.get('api_key', '')}"


NODE_CLASS_MAPPINGS = {"GeminiDiarisationAPI": GeminiDiarisationAPI}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiDiarisationAPI": "Gemini Diarisation"}