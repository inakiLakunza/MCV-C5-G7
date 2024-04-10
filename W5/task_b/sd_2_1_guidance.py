# Example taken from https://huggingface.co/stabilityai/stable-diffusion-2-1

import sys
import os

import torch
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"
# HYPERPARAMETERS
#===================================
USE_DDPM = True
GUIDANCE_SCALE = 25
N_STEPS = 50
N_IMAGES = 5
#====================================

# If you don't swap the scheduler it will run with the default DDIM
# in this example we are swapping it to DPMSolverMultistepScheduler)
pipe = StableDiffusionPipeline.from_pretrained(model_id)
if USE_DDPM:
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "A realistic photo of a panda with glasses studying at a desk, surrounded by bookshelves in a library."
#negative_prompt = "cartoon"
negative_prompt = "smile, smiling, happy"
negative_prompt = None

FOLDER_NAME = str(GUIDANCE_SCALE)+"_guidance"

times = []
for i in range(N_IMAGES):
    start_time = time.time()
    image = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                 guidance_scale=GUIDANCE_SCALE,
                 num_inference_steps=N_STEPS).images[0] 
    end_time = time.time()
    times.append(end_time - start_time)
    FOLDER_PATH = os.path.join("./", "results_sd_2_1", FOLDER_NAME)
    os.makedirs(FOLDER_PATH, exist_ok=True)
    img_save_path = os.path.join(FOLDER_PATH, f"sd_sd_2_1_{i}.png")
    image.save(img_save_path)

print(f"[SD 2.1] Average inference time: {sum(times) / len(times)} s")
