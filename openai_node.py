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
                "model": ([
                    "gpt-6",
                    "gpt-6.1-sol",
                    "gpt-6-luna",
                    "gpt-5.6-sol",
                    "gpt-5.6-terra",
                    "gpt-5.6-luna",
                    "gpt-5.5",
                    "gpt-5.5-pro",
                    "gpt-5.4",
                    "gpt-5.4-pro",
                    "gpt-5.2",
                    "gpt-5.2-pro",
                    "gpt-5",
                    "gpt-5-mini",
                    "gpt-5-nano",
                    "gpt-4.1",
                    "gpt-4.1-mini",
                ], {"default": "gpt-6"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "reasoning_effort": (["none", "low", "medium", "high"], {"default": "medium"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Directly put OpenAI API key or .env variable name (OPENAI_API_KEY)"}),
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

        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("No API key provided.")

        client = OpenAI(api_key=key)

        if image is not None:
            content = [{"type": "input_text", "text": prompt}]
            for i in range(image.shape[0]):
                arr = (image[i].cpu().numpy() * 255).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
            input_content = [{"role": "user", "content": content}]
        else:
            input_content = prompt

        request_params = {
            "model": model,
            "input": input_content,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens
        }

        # Pass reasoning effort only if supported and not set to "none"
        if not model.startswith("gpt-4"):
            if reasoning_effort != "none":
                request_params["reasoning"] = {"effort": reasoning_effort}
        else:
            print(f"Skipping reasoning parameter for {model} (not supported)")

        if system_instruction.strip():
            request_params["instructions"] = system_instruction

        response = client.responses.create(**request_params)

        return (response.output_text,)


NODE_CLASS_MAPPINGS = {"OpenAILLMNode": OpenAILLMNode}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenAILLMNode": "OpenAI LLM"}