"""
Stable Diffusion Benchmark Script.

Prerequisites:
- See https://github.com/pytorch/serve/tree/master/examples/usecases/llm_diffusion_serving_app/README.md
- This script assumes models are available in the mounted volume at
      /home/model-server/model-store/stabilityai---stable-diffusion-xl-base-1.0/model

This script benchmarks Stable Diffusion model inference across different execution modes:
- Eager mode (standard PyTorch)
- Torch.compile with Inductor backend
- Torch.compile with OpenVINO backend

Results are saved in a timestamped directory including:
- JSON file with complete benchmark data
- Generated images for each mode
- Profiling data when enabled

Usage:
    python sd-benchmark.py [--num_iter N] [--run_profiling]
"""

import time
from enum import Enum
from typing import Dict, Tuple
import torch
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
from sd_utils import load_fx_pipeline

class RunMode(Enum):
    EAGER = "eager FP"
    TC_INDUCTOR = "tc_inductor FP"
    TC_OPENVINO_FP = "tc_openvino FP"
    TC_OPENVINO_INT8 = "tc_openvino I8"

def main():
    # Number of benchmark iterations
    num_iter = 1


    run_modes = [
        # RunMode.EAGER.value,
        # RunMode.TC_INDUCTOR.value,
        # RunMode.TC_OPENVINO_FP.value,
        RunMode.TC_OPENVINO_INT8.value,
    ]

    params = {
        "ckpt": "/home/user/Documents/serve2/serve/examples/usecases/llm_diffusion_serving_app/docker/sd3-fx/",
        "guidance_scale": 5.0,
        # "num_inference_steps": 28,
        "num_inference_steps": 7,
        "height": 512,
        "width": 512,
        "prompt": "A close-up HD shot of a vibrant macaw parrot perched on a branch in a lush jungle ",
        "dtype": torch.float16,
    }
    # params["prompt"] = "A close-up of a blooming cherry blossom tree in full bloom"
    with torch.no_grad():
        for mode in run_modes:
            print("\n" + "=" * 80)
            print(
                f"Running benchmark with run mode: {mode}, num_iter: {num_iter}"
            )
            from diffusers import StableDiffusionXLPipeline
            # pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", text_encoder_3=None, tokenizer_3=None)
            pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
            # pipe = load_fx_pipeline(params["ckpt"])
            compile_options = {
                "backend": "openvino",
                "options": {"device": "CPU", "config": {"PERFORMANCE_HINT": "LATENCY"}, "aot_autograd": True},
            }
            pipe.text_encoder = torch.compile(pipe.text_encoder, **compile_options)
            pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, **compile_options)
            # pipe.transformer = torch.compile(pipe.transformer, **compile_options)
            pipe.unet = torch.compile(pipe.unet, **compile_options)
            pipe.vae.decoder = torch.compile(pipe.vae.decoder, **compile_options)
            start_time = time.time()
            try:
                image = pipe(
                    params["prompt"],
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    height=params["height"],
                    width=params["width"],
                ).images[0]
            except Exception as error:
                with open('stderr.txt', 'w') as f:
                    f.write(str(error))
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"Iteration 1 execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
