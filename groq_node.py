import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from openai import OpenAI

VISION_MODELS = {"meta-llama/llama-4-scout-17b-16e-instruct"}
GPT_OSS_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
QWEN_MODELS = {"qwen/qwen3-32b"}

class GroqLLMNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ([
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "qwen/qwen3-32b",
                    "groq/compound",
                    "groq/compound-mini",
                ], {"default": "llama-3.3-70b-versatile"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_completion_tokens": ("INT", {"default": 8192, "min": 1, "max": 131072, "step": 1}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483646, "step": 1}),
                "reasoning_effort": (["disabled", "default", "low", "medium", "high"], {"default": "disabled"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Directly put Groq API key or .env variable name (GROQ_API_KEY)"}),
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

    def generate(self, prompt, model, temperature, max_completion_tokens, seed, reasoning_effort, api_key,
                 system_instruction="", image=None):

        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("No API key provided.")

        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

        # Build messages
        messages = []
        if system_instruction.strip():
            messages.append({"role": "system", "content": system_instruction.strip()})

        if image is not None:
            if model not in VISION_MODELS:
                raise ValueError(f"Model '{model}' does not support image input. Use meta-llama/llama-4-scout-17b-16e-instruct for vision.")

            img_array = image[0].cpu().numpy() if isinstance(image, torch.Tensor) else image[0]
            if img_array.dtype in [np.float32, np.float64]:
                img_array = (img_array * 255).astype(np.uint8)

            buf = io.BytesIO()
            Image.fromarray(img_array).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            messages.append({"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]})
        else:
            messages.append({"role": "user", "content": prompt})

        # Build request params
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }

        if seed != -1:
            params["seed"] = seed

        # Reasoning mapping
        if model in GPT_OSS_MODELS:
            if reasoning_effort != "disabled":
                effort = "medium" if reasoning_effort == "default" else reasoning_effort
                params["extra_body"] = {"reasoning_effort": effort}
        elif model in QWEN_MODELS:
            params["extra_body"] = {"reasoning_effort": "none" if reasoning_effort == "disabled" else "default"}
        else:
            if reasoning_effort != "disabled":
                print(f"GroqLLMNode: '{model}' does not support reasoning_effort — ignoring.")

        response = client.chat.completions.create(**params)
        return (response.choices[0].message.content,)


NODE_CLASS_MAPPINGS = {"GroqLLMNode": GroqLLMNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GroqLLMNode": "Groq LLM"}