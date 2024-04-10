# Example taken from https://huggingface.co/stabilityai/stable-diffusion-2-1

import torch
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# If you don't swap the scheduler it will run with the default DDIM
# in this example we are swapping it to DPMSolverMultistepScheduler)
pipe = StableDiffusionPipeline.from_pretrained(model_id)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "A realistic of a panda with glasses studying at a desk, surrounded by bookshelves in a library."
negative_prompt = "cartoon"
# negative_prompt = "smile, smiling, happy"

times = []
for i in range(1):
    start_time = time.time()
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=999).images[0] 
    end_time = time.time()
    times.append(end_time - start_time)
    image.save(f"./results_sd_2_1/ddpm/sd_2_1_{i}.png")

print(f"[SD 2.1] Average inference time: {sum(times) / len(times)} s")
