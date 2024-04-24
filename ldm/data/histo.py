import bisect
import os
import json
import albumentations
import cv2
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, ConcatDataset


CRC9CLASS2IDX = {
    'ADI': 0,
    'BACK': 1,
    'DEB': 2,
    'LYM': 3,
    'MUC': 4,
    'MUS': 5,
    'NORM': 6,
    'STR': 7,
    'TUM': 8
}

PANNUKECLASS2IDX = {
    'Adrenal_gland': 0,
    'Bladder': 1,
    'Cervix': 2,
    'Esophagus': 3,
    'Kidney': 4,
    'Lung': 5,
    'Pancreatic': 6,
    'Skin': 7,
    'Testis': 8,
    'Uterus': 9,
    'Bile-duct': 10,
    'Breast': 11,
    'Colon': 12,
    'HeadNeck': 13,
    'Liver': 14,
    'Ovarian': 15,
    'Prostate': 16,
    'Stomach': 17,
    'Thyroid': 18
}

# Benign  InSitu  Invasive  Normal
BACHCLASS2IDX = {
    'Benign': 0,
    'InSitu': 1,
    'Invasive': 2,
    'Normal': 3
}

# adenosis          fibroadenoma       mucinous_carcinoma   phyllodes_tumor
# ductal_carcinoma  lobular_carcinoma  papillary_carcinoma  tubular_adenoma
BREAKHISCLASS2IDX = {
    'adenosis': 0,
    'fibroadenoma': 1,
    'mucinous_carcinoma': 2,
    'phyllodes_tumor': 3,
    'ductal_carcinoma': 4,
    'lobular_carcinoma': 5,
    'papillary_carcinoma': 6,
    'tubular_adenoma': 7
}

# 0  1  2  3  4  5
PANDACLASS2IDX = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5
}



class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None, num_sample_per_class=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels

        self.labels["file_path_"] = []
        self.labels["class_label"] = [] # for conditioning
        self.labels["human_label"] = [] # for image logging
        self._length = 0
        for path in paths:
            if 'vgh' in path or ('gpuhome' in path and 'data' in path):
                self.labels["file_path_"].extend([os.path.join(path, i) for i in os.listdir(path)])
                self._length += len(os.listdir(path))
            elif 'CRC' in path or 'PanNuke' in path or 'BACH' in path or 'BreakHis' in path or 'PANDA' in path:
                if 'CRC' in path:
                    CLASS2IDX = CRC9CLASS2IDX
                elif 'PanNuke' in path:
                    CLASS2IDX = PANNUKECLASS2IDX
                elif 'BACH' in path:
                    CLASS2IDX = BACHCLASS2IDX
                elif 'BreakHis' in path:
                    CLASS2IDX = BREAKHISCLASS2IDX
                elif 'PANDA' in path:
                    CLASS2IDX = PANDACLASS2IDX

                train_path = os.path.join(path, 'train')
                tissue_type = os.listdir(train_path)
                human_labels, class_labels, train_files = [], [], []
                for tt in tissue_type:
                    filenames = os.listdir(os.path.join(train_path, tt))
                    if num_sample_per_class is not None and num_sample_per_class < len(filenames):
                        filenames = filenames[:num_sample_per_class]
                    train_files.extend([os.path.join(train_path, tt, fn) for fn in filenames])
                    human_labels.extend([tt] * len(filenames))
                    class_labels.extend([CLASS2IDX[tt]] * len(filenames))
                self.labels["file_path_"].extend(train_files)
                self.labels["class_label"].extend(class_labels)
                self.labels["human_label"].extend(human_labels)
                self._length += len(train_files)
            else:
                filenames = [os.path.join(os.path.join(path, patient), i) for patient in
                             os.listdir(path) for i in os.listdir(os.path.join(path, patient))]
                self.labels["file_path_"].extend(filenames)
                self._length += len(filenames)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        # Increase the pixel limit of images to avoid DecompressionBombError
        Image.MAX_IMAGE_PIXELS = None  # This disables the limit check
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # This helps to avoid issues with damaged images

        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


class HistoBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class HistoTrain(HistoBase):
    def __init__(self, dataset, size, num_sample_per_class=None):
        super().__init__()
        if dataset == 'CRC':
            self.data = ImagePaths(
                paths=['/data/karenyyy/PathologyDatasets/classification/CRC_Data'],
                size=size, random_crop=False, num_sample_per_class=num_sample_per_class)
        elif dataset == 'PanNuke':
            self.data = ImagePaths(
                paths=['/data/karenyyy/PathologyDatasets/classification/PanNuke'],
                size=size, random_crop=False, num_sample_per_class=num_sample_per_class)
        elif dataset == 'BACH':
            self.data = ImagePaths(
                paths=['/data/karenyyy/PathologyDatasets/classification/BACH/ICIAR2018_BACH_Challenge'],
                size=size, random_crop=False, num_sample_per_class=num_sample_per_class)
        elif dataset == 'BreakHis':
            self.data = ImagePaths(
                paths=['/data/karenyyy/PathologyDatasets/classification/BreakHis/BreaKHis_v1/histology_slides/breast'],
                size=size, random_crop=False, num_sample_per_class=num_sample_per_class)
        elif dataset == 'PANDA':
            self.data = ImagePaths(
                paths=['/data/karenyyy/PathologyDatasets/classification/PANDA'],
                size=size, random_crop=False, num_sample_per_class=num_sample_per_class)
        else:
            raise ValueError(f'Unknown dataset: {dataset}')



class LizardDataset(Dataset):
    def __init__(self, val=False):
        super().__init__()
        images = np.load('/data/karenyyy/HistoDiffAug2/Lizard/images.npy').transpose(0, 3, 1, 2)
        labels = np.load('/data/karenyyy/HistoDiffAug2/Lizard/labels.npy').transpose(0, 3, 1, 2)[:, [1]]
        self.images = (images/127.5 - 1.0).astype(np.float32)
        self.labels = (labels/3 - 1.0).astype(np.float32)
        # self.labels = labels.astype(np.float32)
        self.val = val

    def __len__(self):
        return 4000 if not self.val else 981

    def __getitem__(self, i):
        if self.val:
            i += 4000
        return self.images[i], self.labels[i]
        example = dict()
        example['image'] = self.images[i]
        example['mask'] = self.labels[i]
        return example




class ShiftDataset(Dataset):
    def __init__(self, val=False):
        super().__init__()
        part = 'train' if val else 'test'
        self.root_dir = f'/data/Peng/histo_datasets/SHIFT_dataset/{part}'
        self.fnames = os.listdir(self.root_dir+'A')  #H&E

    def __len__(self):
        return len(self.fnames)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).transpose(2, 0, 1)
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        he_img = self.preprocess_image(f'{self.root_dir}A/{self.fnames[i]}')
        patient_id, idx = self.fnames[i][:2], self.fnames[i][26:-4]
        if_img = self.preprocess_image(f'{self.root_dir}B/{patient_id}_AF488_FITC_CROP_{idx}.png')
        return he_img, if_img


class VirtualStainingDataset(Dataset):
    def __init__(self, dataset='skin', mode='train'):
        self.data = []
        self.dataset = dataset
        if mode == 'train':
            self.path = '/gpuhome/jxy225/ControlNet/training'
        elif mode == 'test':
            self.path = '/gpuhome/jxy225/ControlNet/testing'

        prompt_path = f'{self.path}/{self.dataset}/prompt.json'

        with open(prompt_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']

        if self.dataset == 'skin':
            source = cv2.imread(f'{self.path}/{self.dataset}/' + source_filename)
        elif self.dataset == 'lizard' or self.dataset == 'shift' or self.dataset == 'pannuke':
            source = cv2.imread(f'{self.path}/{self.dataset}/' + source_filename,
                                cv2.IMREAD_GRAYSCALE)

        target = cv2.imread(f'{self.path}/{self.dataset}/' + target_filename)

        # resize images
        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))

        if self.dataset == 'skin':
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        elif self.dataset == 'lizard' or self.dataset == 'shift' or self.dataset == 'pannuke':
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)

        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, hint=source)

