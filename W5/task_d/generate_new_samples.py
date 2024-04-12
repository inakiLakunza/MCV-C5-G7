import sys
import os
import tqdm
import json
import torch
from torch.utils.data import DataLoader
from dataloader_original_animal import OriginalAnimalDataset
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def generate_subset_animal(subset_animal):
    res = []

    total=10000-len(subset_animal)
    if total > 2000:
        total = 2000

    for i, element in enumerate(tqdm.tqdm(subset_animal, total=total)):
        # random <animal >caption
        random_element = animal_subset.get_wanted_random_caption_of_subset(
            subset = subset_animal
        )

        # guardar image
        prompt = random_element['caption']
        print(f'prompt: {prompt}')
        negative_prompt = 'cartoon'
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=30).images[0] 
        os.makedirs(f'./augmented_2_1/{element["animal"]}', exist_ok=True)
        image.save(f'./augmented_2_1/{element["animal"]}/{element["animal"]}_{random_element["id"]}_{i}.png')

        res.append({'image_id': f'{element["animal"]}_{i}', 'id': str(i) + '_', 'caption': prompt})

        if i > missing_to10000[element['animal']]:
            break
        if i > 2000:
            break

    return res

if __name__ == '__main__':
    model_id = "stabilityai/stable-diffusion-2-1"

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    missing_to10000 = {
        "horse":    10000-9437,
        "zebra":    10000-6555,
        "cow":      10000-4967,
        "elephant": 10000-7799,
        "sheep":    10000-3994,
        "giraffe":  10000-8943
    }


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The device we are working on is: {DEVICE}")

    animal_subset = OriginalAnimalDataset(True)
    print("Animal set created")

    dataloader_animal_subset = DataLoader(animal_subset,
                                          batch_size=16,
                                          drop_last=False,
                                          )

    results = []
    # results += generate_subset_animal(animal_subset.horse_subset)
    results += generate_subset_animal(animal_subset.cow_subset)
    results += generate_subset_animal(animal_subset.zebra_subset)
    # results += generate_subset_animal(animal_subset.elephant_subset)
    results += generate_subset_animal(animal_subset.sheep_subset)
    # results += generate_subset_animal(animal_subset.giraffe_subset)

    with open("augmented_dataset_2_1.json", "w") as f:
        json.dump(results, f)
        
    




