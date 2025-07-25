import base64
import os
import io
import numpy as np
import torch
from PIL import Image
from typing import Optional
from google import genai
from google.genai import types

class GeminiChatNode:
    """ComfyUI Node for Gemini API Chat with optional image input"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gemini-2.5-pro", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1}),
                "thinking": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 4294967295, "step": 1}),
                "api_key": ("STRING", {"default": "", "multiline": False})
            },
            "optional": {
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "thinking_budget": ("INT", {"default": -1, "min": -1, "max": 24576, "step": 1}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "AI/Gemini"
    
    def generate(self, prompt: str, model: str, temperature: float, thinking: bool, seed: int, api_key: str,
                 system_instruction: Optional[str] = None, thinking_budget: int = -1, 
                 image: Optional[torch.Tensor] = None) -> tuple:
        
        try:
            # API key handling
            key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
            if not key:
                return ("Error: No API key provided.",)
            
            # Initialize client and build parts
            client = genai.Client(api_key=key)
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
            
            model_lower = model.lower()
            
            if "gemini-2.0" in model_lower:
                final_thinking_budget = None
            elif not thinking:
                final_thinking_budget = 0
                if "gemini-2.5-pro" in model_lower:
                    final_thinking_budget = -1
            else:
                final_thinking_budget = thinking_budget
                if "gemini-2.5-pro" in model_lower and final_thinking_budget == 0:
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
            
        except Exception as e:
            return (f"Error: {str(e)}",)

# Node mappings
NODE_CLASS_MAPPINGS = {"GeminiChatNode": GeminiChatNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiChatNode": "Gemini Chat"}