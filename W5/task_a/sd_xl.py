# Example taken from https://huggingface.co/stabilityai/stable-diffusion-2-1

import torch
import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = "A realistic of a panda with glasses studying at a desk, surrounded by bookshelves in a library."
negative_prompt = "cartoon"
# negative_prompt = "smile, smiling, happy"

times = []
for i in range(1):
    start_time = time.time()
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=500).images[0] 
    end_time = time.time()
    times.append(end_time - start_time)
    image.save(f"./results_sd_xl/ddpm/sd_xl_{i}.png")

print(f"[SD XL] Average inference time: {sum(times) / len(times)} s")


