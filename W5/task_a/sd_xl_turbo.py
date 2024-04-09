# Example taken from https://huggingface.co/stabilityai/stable-diffusion-2-1

import torch
import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/sdxl-turbo"

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = "A realistic panda with glasses studying at a desk, surrounded by bookshelves in a library."
#negative_prompt = "cartoon"
negative_prompt = "smile, smiling, happy"

times = []
for i in range(20):
    start_time = time.time()
    image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0] 
    end_time = time.time()
    times.append(end_time - start_time)
    image.save(f"./results_sd_xl_turbo/negative_list/sd_xl_{i}.png")

print(f"[SD XL Turbo] Average inference time: {sum(times) / len(times)} s")
