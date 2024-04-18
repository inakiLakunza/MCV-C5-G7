from dataloader import Dataset
from torch.utils.data import ConcatDataset, DataLoader
import sys


age_mapping = {
    "1": "[7,13]",
    "2": "[14,18]",
    "3": "[19,24]",
    "4": "[25,32]",
    "5": "[33,45]",
    "6": "[46,60]",
    "7": "61+"
}

gender_mapping = {
    "1": "Male",
    "2": "Female"
}

ethnicity_mapping = {
    "1": "Asian",
    "2": "Caucasian",
    "3": "African-American",
}


def get_frequencies_of_labels(dataloader):
    counts_age = {}
    counts_gender = {}
    counts_ethnicity = {}
    for data in dataloader:
        ages, genders, ethnicities = data
        for age in ages: 
            counts_age[age_mapping[age]] = counts_age.get(age_mapping[age], 0) + 1
        for gender in genders: 
            counts_gender[gender_mapping[gender]] = counts_gender.get(gender_mapping[gender], 0) + 1
        for ethnicity in ethnicities: 
            counts_ethnicity[ethnicity_mapping[ethnicity]] = counts_ethnicity.get(ethnicity_mapping[ethnicity], 0) + 1
    print(f'counts_age: {counts_age}')
    print(f'counts_gender: {counts_gender}')
    print(f'counts_ethnicity: {counts_ethnicity}')


def get_age_distribution_by_gender(dataloader):
    counts_age = {}
    for data in dataloader:
        ages, genders, _ = data
        for age, gender in zip(ages, genders):
            if gender_mapping[gender] not in counts_age:
                counts_age[gender_mapping[gender]] = {}
            age_category = age_mapping[age]
            if age_category not in counts_age[gender_mapping[gender]]:
                counts_age[gender_mapping[gender]][age_category] = 0
            counts_age[gender_mapping[gender]][age_category] += 1
    print(f'counts_age: {counts_age}')


def get_age_distribution_by_ethnicity(dataloader):
    counts_age = {}
    for data in dataloader:
        ages, _, ethnicities = data
        for age, ethnicity in zip(ages, ethnicities):
            if ethnicity_mapping[ethnicity] not in counts_age:
                counts_age[ethnicity_mapping[ethnicity]] = {}
            age_category = age_mapping[age]
            if age_category not in counts_age[ethnicity_mapping[ethnicity]]:
                counts_age[ethnicity_mapping[ethnicity]][age_category] = 0
            counts_age[ethnicity_mapping[ethnicity]][age_category] += 1
    print(f'counts_age: {counts_age}')


def get_age_distribution_by_gender_and_ethnicity(dataloader):
    counts_age = {}
    for data in dataloader:
        ages, genders, ethnicities = data
        for age, gender, ethnicity in zip(ages, genders, ethnicities):
            if gender_mapping[gender] not in counts_age:
                counts_age[gender_mapping[gender]] = {}
            if ethnicity_mapping[ethnicity] not in counts_age[gender_mapping[gender]]:
                counts_age[gender_mapping[gender]][ethnicity_mapping[ethnicity]] = {}
            if age_mapping[age] not in counts_age[gender_mapping[gender]][ethnicity_mapping[ethnicity]]:
                counts_age[gender_mapping[gender]][ethnicity_mapping[ethnicity]][age_mapping[age]] = 0
            counts_age[gender_mapping[gender]][ethnicity_mapping[ethnicity]][age_mapping[age]] += 1
    print(counts_age)


if __name__ == '__main__':
    batch_size = 32

    print("Loading Datasets...")
    train_dataset = Dataset(regime='train')
    val_dataset = Dataset(regime='val')
    test_dataset = Dataset(regime='test')

    print("Loading Dataloaders...")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    # If we want to use the overall dataset!!!
    total_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    total_dataloader = DataLoader(dataset=total_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    print("\n============== FREQUENCIES OF EACH LABEL FOR [train + val + test] ==================")
    get_frequencies_of_labels(total_dataloader)

    print("\n============== AGE DISTRIBUTION BY GENDER [train + val + test] ==================")
    get_age_distribution_by_gender(total_dataloader)
    
    print("\n============== AGE DISTRIBUTION BY ETHNICITY [train + val + test] ==================")
    get_age_distribution_by_ethnicity(total_dataloader)

    print("\n============== AGE DISTRIBUTION BY GENDER AND ETHNICITY [train + val + test] ==================")
    get_age_distribution_by_gender_and_ethnicity(total_dataloader)