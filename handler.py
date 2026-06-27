import base64
import gc
import io
import os
import threading
import uuid
from typing import Dict, Tuple

import runpod
import torch
from diffusers import Krea2Pipeline

MODEL_ID = os.environ.get("MODEL_ID", "krea/Krea-2-Turbo")
HF_TOKEN = os.environ.get("HF_TOKEN")
ENGINE_NAME = os.environ.get("ENGINE_NAME", "krea-2-turbo")

_pipe = None
_pipe_lock = threading.Lock()

ASPECT_SIZES: Dict[str, Tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1024, 1024),
    "4:3": (1024, 768),
    "21:9": (1536, 640),
}


def get_dtype():
    dtype_name = os.environ.get("TORCH_DTYPE", "bfloat16").lower()

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32

    return torch.bfloat16


def get_pipe():
    global _pipe

    if _pipe is not None:
        return _pipe

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This endpoint must run on a GPU worker.")

    load_kwargs = {
        "torch_dtype": get_dtype(),
    }

    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    _pipe = Krea2Pipeline.from_pretrained(
        MODEL_ID,
        **load_kwargs,
    ).to("cuda")

    try:
        _pipe.enable_attention_slicing()
    except Exception:
        pass

    return _pipe


def health():
    return {
        "status": "ok",
        "worker": "hermes-hero-image-krea-serverless",
        "engine": ENGINE_NAME,
        "model": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_dtype": str(get_dtype()),
        "supported_aspect_ratios": list(ASPECT_SIZES.keys()),
        "hf_token_present": bool(HF_TOKEN),
    }


def build_prompt(prompt: str, aspect_ratio: str) -> str:
    base = prompt.strip()

    quality_tail = (
        ", premium editorial hero image, cinematic lighting, clean composition, "
        "professional commercial-grade visual, no text, no words, no letters, "
        "no logo, no watermark"
    )

    if aspect_ratio == "16:9":
        layout_tail = ", wide horizontal composition, strong focal point, negative space for headline overlay"
    elif aspect_ratio == "9:16":
        layout_tail = ", vertical mobile composition, centered subject, top and bottom breathing room"
    elif aspect_ratio == "1:1":
        layout_tail = ", square composition, centered focal subject, balanced clean layout"
    elif aspect_ratio == "21:9":
        layout_tail = ", ultra-wide cinematic masthead composition, generous negative space"
    else:
        layout_tail = ", clean balanced editorial composition"

    return base + quality_tail + layout_tail


def generate(job_input):
    prompt = (job_input.get("prompt") or "").strip()
    aspect_ratio = job_input.get("aspect_ratio") or "16:9"
    filename = job_input.get("filename") or f"hero_{uuid.uuid4().hex}.png"

    if not prompt:
        raise ValueError("Prompt is required.")

    if aspect_ratio not in ASPECT_SIZES:
        raise ValueError(
            f"Unsupported aspect ratio: {aspect_ratio}. "
            f"Supported: {list(ASPECT_SIZES.keys())}"
        )

    width, height = ASPECT_SIZES[aspect_ratio]

    steps = int(job_input.get("steps") or os.environ.get("DEFAULT_STEPS", "8"))
    guidance_scale = float(
        job_input.get("guidance_scale") or os.environ.get("DEFAULT_GUIDANCE_SCALE", "0.0")
    )

    seed = job_input.get("seed")
    generator = None

    if seed is not None:
        seed = int(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)

    final_prompt = build_prompt(prompt, aspect_ratio)

    pipe = get_pipe()

    call_kwargs = {
        "prompt": final_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
    }

    if generator is not None:
        call_kwargs["generator"] = generator

    result = pipe(**call_kwargs)
    image = result.images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "status": "ok",
        "job_id": str(uuid.uuid4()),
        "engine": ENGINE_NAME,
        "model": MODEL_ID,
        "filename": filename,
        "mime_type": "image/png",
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "size_bytes": len(image_bytes),
        "image_base64": image_b64,
    }


def handler(job):
    job_input = job.get("input", {}) or {}
    action = job_input.get("action")

    try:
        if action == "health":
            return health()

        if action == "generate":
            return generate(job_input)

        return {
            "status": "error",
            "error": f"Unknown action: {action}",
            "supported_actions": ["health", "generate"],
        }

    except Exception as e:
        return {
            "status": "error",
            "action": action,
            "error": str(e),
            "engine": ENGINE_NAME,
            "model": MODEL_ID,
        }


runpod.serverless.start({"handler": handler})
