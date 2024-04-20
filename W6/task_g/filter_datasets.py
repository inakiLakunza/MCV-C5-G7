import os
from torchvision import datasets,transforms


class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, 
                 dataset, 
                 transform=transforms.Compose([transforms.ToTensor(), 
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                    std=[0.229, 0.224, 0.225])]), 
                 target_transform=None, 
                 filter_classes=[]):
        
        self.root = dataset.root
        super().__init__(root=self.root, transform=transform, target_transform=target_transform)
        self.dataset = dataset
        self.filter_classes = set(filter_classes)
        self.filtered_samples = self._filter_samples()

    def _filter_samples(self):
        filtered_samples = []
        for idx, (sample, target) in enumerate(self.dataset.samples):
            if self.dataset.classes[target] not in self.filter_classes:
                filtered_samples.append((sample, target))
        return filtered_samples

    def __getitem__(self, index):
        path, target = self.filtered_samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.filtered_samples)