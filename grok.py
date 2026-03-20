import os
import io
import base64
import requests
import torch
import numpy as np
from PIL import Image


XAI_GENERATIONS_URL = "https://api.x.ai/v1/images/generations"
XAI_EDITS_URL = "https://api.x.ai/v1/images/edits"


def _tensor_to_base64(tensor) -> str:
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_obj(tensor) -> dict:
    frame = tensor[0] if tensor.dim() == 4 else tensor
    return {"url": f"data:image/png;base64,{_tensor_to_base64(frame)}", "type": "image_url"}


def _url_to_tensor(url: str) -> torch.Tensor:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class GrokImageAPINode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (
                    ["grok-imagine-image", "grok-imagine-image-beta"],
                    {"default": "grok-imagine-image"},
                ),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9",
                     "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1"],
                    {"default": "1:1",
                     "tooltip": "Applies to generation and multi-image edits. "
                                "Ignored for single-image edits (output follows input aspect ratio)."},
                ),
                "resolution": (["1k", "2k"], {"default": "1k"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                              "tooltip": "Number of images to generate or edited variations to produce"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1,
                                 "tooltip": "Used to trigger re-runs in ComfyUI; not sent to the API"}),
                "api_key": ("STRING", {"multiline": False, "default": "",
                                       "tooltip": "Directly put xAI API key or .env variable name (XAI_API_KEY)"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit_image"
    CATEGORY = "image/generation"

    def edit_image(self, prompt, model, aspect_ratio, resolution, n, seed, api_key,
            image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        if not prompt.strip():
            raise ValueError("Grok: prompt cannot be empty.")

        key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("XAI_API_KEY")
        if not key:
            raise ValueError("Grok: no API key provided.")

        input_images = [img for img in [image_1, image_2, image_3, image_4, image_5] if img is not None]
        num_images = len(input_images)

        if num_images == 0:
            endpoint = XAI_GENERATIONS_URL
            payload = {
                "model": model,
                "prompt": prompt.strip(),
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "n": n,
            }
        else:
            endpoint = XAI_EDITS_URL
            single = num_images == 1
            payload = {
                "model": model,
                "image" if single else "images": _image_obj(input_images[0]) if single else [_image_obj(img) for img in input_images],
                "prompt": prompt.strip(),
                "resolution": resolution,
                "n": n,
            }
            if single:
                print("Grok: aspect_ratio ignored for single-image edits (output follows input aspect ratio).")
            else:
                payload["aspect_ratio"] = aspect_ratio

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Grok API error {resp.status_code} – {resp.text}")

        data = resp.json().get("data", [])
        if not data:
            raise RuntimeError("Grok: API returned no image data.")

        tensors = []
        for item in data:
            if item.get("url"):
                tensors.append(_url_to_tensor(item["url"]))
            elif item.get("b64_json"):
                pil = Image.open(io.BytesIO(base64.b64decode(item["b64_json"]))).convert("RGB")
                tensors.append(torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0))
        if not tensors:
            raise RuntimeError("Grok: could not retrieve any output images.")

        return (torch.cat(tensors, dim=0),)

    @classmethod
    def IS_CHANGED(cls, prompt, seed, **kwargs):
        return f"{prompt}-{seed}"


NODE_CLASS_MAPPINGS = {"GrokImageAPINode": GrokImageAPINode}
NODE_DISPLAY_NAME_MAPPINGS = {"GrokImageAPINode": "Grok Image API"}