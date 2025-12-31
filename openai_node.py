import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from openai import OpenAI

class OpenAILLMNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["gpt-4.1","gpt-4.1-mini","gpt-5","gpt-5.2","gpt-5-mini","gpt-5-nano","gpt-5.2-pro","o1","o3-mini"],),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "reasoning_effort": (["low", "medium", "high"],),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "max_output_tokens": ("INT", {"default": 16384, "min": 1, "max": 32768, "step": 1})
            },
            "optional": {
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "text/generation"
    
    def generate(self, prompt, model, temperature, reasoning_effort, api_key, max_output_tokens,
                 system_instruction="", image=None):
        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        
        key = api_key.strip() or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("No API key provided.")
        
        client = OpenAI(api_key=key)
        
        # Build input content
        if image is not None:
            img_array = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
            if len(img_array.shape) == 4:
                img_array = img_array[0]
            if img_array.dtype in [np.float32, np.float64]:
                img_array = (img_array * 255).astype(np.uint8)
            
            buffered = io.BytesIO()
            Image.fromarray(img_array).save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            input_content = [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"}
                ]
            }]
        else:
            input_content = prompt
        
        # Build request
        request_params = {
            "model": model,
            "input": input_content,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens
        }
        
        # Only add reasoning for o-series and gpt-5+ models
        if not model.startswith("gpt-4"):
            request_params["reasoning"] = {"effort": reasoning_effort}
        else:
            print(f"Skipping reasoning parameter for {model} (not supported)")
        
        if system_instruction.strip():
            request_params["instructions"] = system_instruction
        
        response = client.responses.create(**request_params)
        
        return (response.output_text,)

NODE_CLASS_MAPPINGS = {"OpenAILLMNode": OpenAILLMNode}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenAILLMNode": "OpenAI LLM"}