import sys
import os
import json
import tqdm


animals = [
    "horse",
    "zebra",
    "cow",
    "elephant",
    "sheep",
    "giraffe"
]


if __name__ == '__main__':
    PATH_PARENT_DIRECTORY = "/ghome/group07/mcv/datasets/C5/COCO"
    PATH_TRAINING_SET = os.path.join(PATH_PARENT_DIRECTORY, "train2014")
    PATH_VAL_SET = os.path.join(PATH_PARENT_DIRECTORY, "val2014")
    PATH_CAPTIONS_TRAIN = os.path.join(PATH_PARENT_DIRECTORY, "captions_train2014.json")
    f = open(PATH_CAPTIONS_TRAIN)
    config = json.load(f)

    # initialize dict
    animals_subset = dict()
    for animal in animals:
        animals_subset[animal] = []

    # build animals subset
    dataset = []
    for element in tqdm.tqdm(config["annotations"]):
        image_id = element['image_id']
        id = element['id']
        caption = element['caption']
        for animal in animals:
            if animal in caption.lower():
                # animals_subset[animal].append({'image_id': image_id, 'id': id, 'caption': caption})
                dataset.append({'image_id': image_id, 'id': id, 'caption': caption, 'animal': animal})

    #print(animals_subset['horse'])

    with open("animal_dataset.json", "w") as f:
        json.dump(dataset, f)



    