import os
import json
import random
from functools import partial
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
from copy import deepcopy
import torch

# Hugging Face datasets
from datasets import load_dataset, Dataset, ClassLabel
from datasets import Image as HF_Image

import torchvision.datasets as datasets

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2
except Exception as e:
    import warnings
    warnings.warn("albumentations and OpenCV are not installed.")


def get_multiprocessing_context(args):
    """
    Running multiple jobs in parallel (using joblib) fails when num_workers in DataLoaders is > 0.
    
    This is mainly for backward compatibility in case. It does not hurt when working with the latest version.
    
    https://github.com/pytorch/pytorch/issues/44687
    """
    multiprocessing_context = None
    if hasattr(args, "n_jobs") and args.n_jobs is not None and args.n_jobs > 1 \
            and hasattr(args, "num_workers") and args.num_workers is not None \
        and args.num_workers > 1:
        from joblib.externals.loky.backend.context import get_context
        multiprocessing_context = get_context('loky')
    return multiprocessing_context


def worker_init_fn(worker_id):
    """
    useful for PyTorch version < 1.9
    
    This is mainly for backward compatibility in case (PyTorch 1.8 is still widely used these days).
    It does not hurt when working with the latest version.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_augmentation(dataset):
    """
    albumentations works with OpenCV format
    """
    proba = 1

    if dataset == "imagenet":
        transform = A.Compose(
        [
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.4)
        ]
        )

    else:
        list_albumentations = [
            A.Affine(shear=10, p=proba),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.01,
                            rotate_limit=30, p=proba),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                    b_shift_limit=15, p=proba),
            A.RandomBrightnessContrast(p=proba),
            A.Posterize(p=proba),
            A.MotionBlur(p=proba),
            A.ImageCompression(p=proba),
            A.GaussNoise(p=proba),
            A.Equalize(p=proba),
            A.Solarize(p=proba),
            A.Sharpen(p=proba),
            A.InvertImg(p=proba),
        ]

        # With HuggingFace, resizing and normalization will be done by preprocessors
        transform = A.Compose(
            [
                A.OneOf(list_albumentations, p=1),
                A.OneOf(list_albumentations, p=1),
                ToTensorV2(), # from OpenCV format to torch.Tensor
            ]
        )
    
    return transform


def pretrained_preprocess_with_data_augmentation(examples, preprocessor, data_augmentation, device):
    """
    assume examples["image"] are PIL images
    
    OpenCV format are required by the albumentations library
    """
    def convert_PIL_to_OpenCV_format(pil_image):
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
        return img

    # examples is a python dict with keys'image' (type: list) and 'label' (type: list)
    
    # convert_PIL_to_OpenCV_format(): 
    #   from PIL to OpenCV format
    # data_augmentation(image=...)["image"]: 
    #   from OpenCV format to torch.Tensor
    # We do not directly convert back to PIL because of potential errors. 
    examples["pixel_values"] = [
        data_augmentation(image=convert_PIL_to_OpenCV_format(
            image.convert("RGB")))["image"] for image in examples["image"]
    ]
    
    # from torch.Tensor to PIL image (intentionally commented out)
    #examples["image"] = [to_pil_image(xx) for xx in examples["image"]]
    
    batch_feature = pretrained_preprocess(
        examples, preprocessor, device, target_field="pixel_values")
    
    return batch_feature


def pretrained_preprocess(examples, preprocessor, device, target_field):
    """
    apply the pretrained preprocessor to all examples in the dataset on the fly
    
    examples: python dict with keys "image" and "label"
        examples["image"]: list of PIL image
        examples["label"]: list of int
        (potentially) examples["pixel_values"]: list of torch.Tensor
        
    target_field: str
        Either "image" or "pixel_values" (field of examples to preprocess)
        Use target_field="image" when pretrained_preprocess is 
            used alone (without data augmentation)
        Use target_field="pixel_values" when pretrained_preprocess is 
            used inside pretrained_preprocess_with_data_augmentation 
            (chained with data augmentation)
        
    batch_feature: transformers.image_processing_utils.BatchFeature
        keys: 
            "pixel_values": torch.Tensor
            "labels": int      (not "label", because it must match the argument of forward() )
            
            "image": PIL image (removed, otherwise forward() cannot be done)
    
    To move (model's) input data to (GPU) device, simply call batch_feature.to(device)
            
    x.convert("RGB") is necessary for HuggingFace because it (at 
    least some preprocessors) does not support grayscale images. 
    
    One drawback of including preprocessor into set_transform() of the dataset:
        when inference with batch_size=1, the batch dimension no longer exists which 
        is a problem for models' forward() method. To solve this problem, 
        add .unsqueeze(0) for pixel_values. The following is an example:
            model(pixel_values=train_dataloader.dataset[0]["pixel_values"].unsqueeze(0))
    """
    if target_field == "image":
        # Take a list of PIL images, results are stored in batch_feature["pixel_values"]
        batch_feature = preprocessor(
            [x.convert("RGB") for x in examples[target_field]],
            return_tensors="pt"
        )
    elif target_field == "pixel_values":
        # Take a list of torch.Tensor, results are stored in batch_feature["pixel_values"]
        batch_feature = preprocessor(
            [x for x in examples[target_field]],
            return_tensors="pt"
        )
    else:
        raise Exception("Target field {} not recognized.".format(target_field))

    batch_feature["labels"] = torch.tensor(examples["label"], dtype=torch.long)
    
    # the following line is intentionally commented out.
    #batch_feature["image"] = examples["image"]
    
    # move input data to device 
    # the following line is intentionally commented out because otherwise, it is 
    # not compatible with torch.utils.data.DataLoader(..., pin_memory=True)
    # See: https://pytorch.org/docs/stable/data.html#memory-pinning 
    
    #batch_feature = batch_feature.to(device)
    
    return batch_feature


def load_imagenette(imagenette_version="320px"):
    """
    imagenette does not have a test split
    """
    train_dataset = load_dataset(
        "frgfm/imagenette", name=imagenette_version, split="train")
    validation_dataset = load_dataset(
        "frgfm/imagenette", name=imagenette_version, split="validation")

    return train_dataset, validation_dataset

def load_imagenet(root):
    if root is None:
        train_dataset = load_dataset("imagenet-1k", split="train", use_auth_token=True)
        val_dataset = load_dataset("imagenet-1k", split="validation", use_auth_token=True)
    else:
        train_dataset = load_dataset(root, split="train")
        val_dataset = load_dataset(root, split="validation")

    return train_dataset, val_dataset

def hf_image_collate_fn(batch):
    """
    This collate_fn is useful for HuggingFace model (forward() with labels) and Trainer.
    
    batch: list of length "batch size"
        each element is a python dict with keys: 
            "pixel_values" (torch.Tensor)
            "image" (PIL image)
            "labels" (int)
    """
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }


def torch_dataloader_image_collate_fn(batch):
    """
    This collate_fn is useful to create "standard" PyTorch dataloader.
    
    batch: list of length "batch size"
        each element is a python dict with keys: 
            "pixel_values" (torch.Tensor)
            "image" (PIL image)
            "labels" (int)
    """
    dict_ = hf_image_collate_fn(batch)

    X = dict_["pixel_values"]
    y = dict_["labels"]

    return X, y


def get_dataloaders(args, do_data_augmentation_for_training_split, 
                    image_processor=None, collate_type="hf", device="cpu"):
    """
    get_dataloaders is the unique access point for data, e.g.
        if you need the training dataset, then you can just call train_dataloader.dataset, 
        where train_dataloader.dataset is the plain HuggingFace dataset equiped with the 
        pretrained preprocessor. 
    
    image_processor is integrated into the datasets.
    
    set device in get_dataloaders() will avoid the labor of manual data transfer
        the above sentence is legacy?
    """

    train_dataset, val_dataset, test_dataset = None, None, None
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    
    if args.dataset == "imagenette":
        train_dataset, val_dataset = load_imagenette()
    elif args.dataset == "Icdar_Micro":
        train_dataset, val_dataset, test_dataset = get_ICDAR_2003_character_datasets()
    elif args.dataset == "imagenet":
        train_dataset, val_dataset = load_imagenet(args.imagenet_root)
    else:
        train_dataset, val_dataset, test_dataset = get_Meta_Album_datasets(args)
        
    # apply preprocess to datasets
    partial_pretrained_preprocess = partial(
        pretrained_preprocess, preprocessor=image_processor, device=device, target_field="image")
    
    if train_dataset is not None:
        # optionally apply data augmentation to the training split
        if do_data_augmentation_for_training_split:
            
            data_augmentation = get_data_augmentation(args.dataset)
            
            train_image_processor = deepcopy(image_processor)
            if args.dataset == "imagenet":
                train_image_processor.do_resize = False
                train_image_processor.do_crop = False
            
            train_partial_pretrained_preprocess = partial(
                pretrained_preprocess_with_data_augmentation, 
                preprocessor=train_image_processor,
                data_augmentation=data_augmentation,
                device=device,
            )
        else:
            train_partial_pretrained_preprocess = partial_pretrained_preprocess
        train_dataset.set_transform(train_partial_pretrained_preprocess)
    if val_dataset is not None:
        val_dataset.set_transform(partial_pretrained_preprocess)
    if test_dataset is not None:
        test_dataset.set_transform(partial_pretrained_preprocess)
    
    # if you run into errors in the following line related to key errors "label"/"labels", 
    # consider clear cache by deleting ~/.cache/huggingface/datasets/generator/
    num_classes = len(train_dataset.features["label"].names)
    
    multiprocessing_context = get_multiprocessing_context(args)
    
    collate_fn = hf_image_collate_fn if collate_type == "hf" else torch_dataloader_image_collate_fn

    if args.dataset == "imagenet":
        pin_memory_ = True
    else:
        pin_memory_ = False

    if train_dataset is not None:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            collate_fn=collate_fn,
            pin_memory=pin_memory_,  # the default setting pin_memory=False
        )
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            collate_fn=collate_fn,
            pin_memory=pin_memory_,  # the default setting pin_memory=False
        )
    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            collate_fn=collate_fn,
            pin_memory=pin_memory_,  # the default setting pin_memory=False
        )

    return num_classes, train_dataloader, val_dataloader, test_dataloader


s0_list = ["BCT_Micro", "BRD_Micro", "CRS_Micro", "FLW_Micro", "MD_MIX_Micro",
           "PLK_Micro", "PLT_VIL_Micro", "RESISC_Micro", "SPT_Micro", "TEX_Micro"]

s1_list = ["ACT_40_Micro", "APL_Micro", "DOG_Micro", "INS_2_Micro", "MD_5_BIS_Micro",
           "MED_LF_Micro", "PLT_NET_Micro", "PNU_Micro", "RSICB_Micro", "TEX_DTD_Micro"]

s2_list = ["ACT_410_Micro", "AWA_Micro", "BTS_Micro", "FNG_Micro", "INS_Micro",
           "MD_6_Micro", "PLT_DOC_Micro", "PRT_Micro", "RSD_Micro", "TEX_ALOT_Micro"]

s3_list = ["ARC_Micro", "ASL_ALP_Micro", "BFY_Micro", "BRK_Micro", "MD_5_T_Micro",
           "PLT_LVS_Micro", "POT_TUB_Micro", "SNK_Micro", "UCMLU_Micro", "VEG_FRU_Micro"]


def get_Meta_Album_datasets(args):
    """
    train/val/test split per class: 20/0/20
    """

    dataset_name = args.dataset

    if dataset_name in s0_list:
        subfolder_name = "Set0_Micro"
    elif dataset_name in s1_list:
        subfolder_name = "Set1_Micro"
    elif dataset_name in s2_list:
        subfolder_name = "Set2_Micro"
    elif dataset_name in s3_list:
        subfolder_name = "Set3_Micro"
    else:
        raise Exception("Dataset {} not recognized.".format(args.dataset))
    
    dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "datasets", 
                                     subfolder_name, 
                                     dataset_name)

    train_dataset = MetaAlbumMicroTorchDataset(
        dataset_directory=dataset_directory, split="train")
    
    # because of 20/0/20 split
    val_dataset = None

    test_dataset = MetaAlbumMicroTorchDataset(
        dataset_directory=dataset_directory, split="test")
    
    # convert plain PyTorch datasets to HuggingFace datasets
    train_dataset = convert_torch_dataset_to_hf_dataset(train_dataset)
    val_dataset = convert_torch_dataset_to_hf_dataset(val_dataset)
    test_dataset = convert_torch_dataset_to_hf_dataset(test_dataset)

    return train_dataset, val_dataset, test_dataset


def _hf_dataset_conversion_helper_generator(torch_dataset):
    for idx in range(len(torch_dataset)):
        X, y = torch_dataset[idx]
        yield {"image": X, "label": y}


def convert_torch_dataset_to_hf_dataset(torch_dataset):
    """
    converts a plain PyTorch dataset (with the 
        attribute "raw_labels") to a HuggingFace dataset.
    """
    if torch_dataset is None:
        return None
    
    label_names = torch_dataset.raw_labels
    
    partial_generator = partial(
        _hf_dataset_conversion_helper_generator, torch_dataset=torch_dataset)
    
    hf_dataset = Dataset.from_generator(partial_generator).cast_column(
        "label", ClassLabel(names=label_names)).cast_column("image", HF_Image())
    
    return hf_dataset


class MetaAlbumMicroTorchDataset(torch.utils.data.Dataset):
    """
    This is the plain PyTorch dataset format (with the field "targets") 
    for the Meta-Album micro datasets. 
    
    This dataset loads image paths (strings, not PIL image objects) into RAM 
    at the initialization phase.
    """

    def __init__(self, dataset_directory, split):
        super().__init__()

        self.dataset_directory = dataset_directory

        assert split in ["train", "val", "test"]
        self.split = split
        
        self.train_examples_per_class = 20
        self.val_examples_per_class = 0
        self.test_examples_per_class = 20

        assert os.path.exists(dataset_directory), \
            "Dataset path {} not found.".format(dataset_directory)

        self.items = self.construct_items()
        self.divide_items_into_split()

        self.add_field_targets()

        self.n_classes = int(max(list(set(self.targets)))) + 1

        # intentionally commented out, because it's handled on the HF side
        #self.transform_paths_to_pil_images_in_RAM()
        
        self.add_raw_labels()
        
    def add_raw_labels(self):
        """
        useful for HF datasets' ClassLabel
        """
        tuples = []
        for raw_label, label in self.raw_label2label.items():
            tuples.append((raw_label, label))
        # make sure that the order is correct
        tuples = sorted(tuples, key=lambda x: x[1])
        
        self.raw_labels = []
        for raw_label, label in tuples:
            self.raw_labels.append(raw_label)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, target = self.items[idx]
        return img, target

    def transform_paths_to_pil_images_in_RAM(self):
        """
        modifies self.items in place so that instead of storing image path 
        in RAM, we store PIL image in RAM (the dataset should 
        not be too large compared to RAM)
        """
        new_items = []
        for img_path, target in self.items:
            img = Image.open(img_path) 
            new_items.append([img, target])
        self.items = new_items

    def divide_items_into_split(self):
        """
        modifies self.items in place so that each class contains 
        20/0/20 examples per class for train/val/test split
        """
        # reshuffle to break the order of examples or classes
        rng = np.random.default_rng(88)
        rng.shuffle(self.items)

        items_per_class = defaultdict(list)

        for img_path, target in self.items:
            items_per_class[target].append([img_path, target])

        pruned_items = []
        for class_ in items_per_class.keys():

            assert len(items_per_class[class_]) >= self.train_examples_per_class + \
                self.val_examples_per_class + self.test_examples_per_class

            if self.split == "train":
                pruned_items.extend(
                    items_per_class[class_][:self.train_examples_per_class])
            elif self.split == "val":
                pruned_items.extend(
                    items_per_class[class_][self.train_examples_per_class:
                                            (self.train_examples_per_class + self.val_examples_per_class)])
            elif self.split == "test":
                pruned_items.extend(
                    items_per_class[class_][(self.train_examples_per_class + self.val_examples_per_class):
                                            (self.train_examples_per_class + self.val_examples_per_class
                                                + self.test_examples_per_class)])
            else:
                raise Exception

        # reshuffle to break the order of examples or classes
        rng = np.random.default_rng(66)
        rng.shuffle(pruned_items)

        self.items = pruned_items

    def construct_items(self):
        """
        returns items: 
            a list of lists, each inner-list has 2 elements.
            the first is img_path, the second is the 
            classification label (integer).
        """
        # get raw labels
        self.read_info_json()
        df = self.read_labels_csv()

        tmp_items = []
        for idx_value, row in df.iterrows():
            img_path = os.path.join(
                self.dataset_directory, "images", idx_value)
            tmp_items.append([img_path,
                              df.loc[idx_value, self.category_column_name]])
        
        # map each raw label to a label (int starting from 0)
        self.raw_label2label = dict()
        items = []
        for item in tmp_items:
            if item[1] not in self.raw_label2label:
                self.raw_label2label[item[1]] = len(self.raw_label2label)
            items.append([item[0], self.raw_label2label[item[1]]])
        return items

    def read_info_json(self) -> None:
        info_json_path = os.path.join(
            self.dataset_directory, "info.json")
        with open(info_json_path, "r") as f:
            info_json = json.load(f)
        # "FILE_NAME"
        self.image_column_name = info_json["image_column_name"]
        # "CATEGORY"
        self.category_column_name = info_json["category_column_name"]

    def read_labels_csv(self):
        csv_path = os.path.join(
            self.dataset_directory, "labels.csv")
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8")
        df = df.loc[:, [self.image_column_name, self.category_column_name]]
        
        # convert potential int64 columns to str columns
        df[self.category_column_name] = df[self.category_column_name].astype(str)
        
        df.set_index(self.image_column_name, inplace=True)
        return df

    def add_field_targets(self):
        """
        The targets field is available in nearly all torchvision datasets. 
        It must be a list containing the label for each data point (usually the y value).
        
        https://avalanche.continualai.org/how-tos/avalanchedataset/creating-avalanchedatasets
        """
        self.targets = []
        for item in self.items:
            self.targets.append(item[1])
        self.targets = torch.tensor(self.targets, dtype=torch.int64)


def get_ICDAR_2003_character_datasets():
    """
    train/val/test split per class: 15/5/20 
    """
    
    dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "datasets", "ICDAR_2003_character_dataset")
    
    
    train_dataset = Icdar03MicroTorchDataset(micro_split="train", 
                                             dataset_root=dataset_root)
    val_dataset = Icdar03MicroTorchDataset(micro_split="val", 
                                           dataset_root=dataset_root)
    test_dataset = Icdar03MicroTorchDataset(micro_split="test", 
                                            dataset_root=dataset_root)
    
    # convert plain PyTorch datasets to HuggingFace datasets
    train_dataset = convert_torch_dataset_to_hf_dataset(train_dataset)
    val_dataset = convert_torch_dataset_to_hf_dataset(val_dataset)
    test_dataset = convert_torch_dataset_to_hf_dataset(test_dataset)
    
    return train_dataset, val_dataset, test_dataset


class Icdar03MicroTorchDataset(torch.utils.data.Dataset):
    """
    This is the plain PyTorch dataset format (with the field "targets") 
    for the ICDAR-micro datasets (from ICDAR 2003 character dataset). 
    
    This dataset loads image paths (strings, not PIL image objects) into RAM 
    at the initialization phase.
    """
    
    shared_classes_unicode = [33, 34, 38, 39, 40, 41, 45, 46, 48, 49,
                              50, 51, 52, 53, 54, 55, 56, 57, 63, 65,
                              66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                              76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                              86, 87, 88, 89, 90, 97, 98, 99, 100, 101,
                              102, 103, 104, 105, 106, 107, 108, 109,
                              110, 111, 112, 113, 114, 115, 116, 117,
                              118, 119, 120, 121, 163]

    alphanumeric = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67,
                    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99,
                    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122]

    all_test_set_classes = [33, 34, 38, 39, 40, 41, 44, 45, 46, 48, 49,
                            50, 51, 52, 53, 54, 55, 56, 57, 63, 65, 66,
                            67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                            89, 90, 97, 98, 99, 100, 101, 102, 103, 104,
                            105, 106, 107, 108, 109, 110, 111, 112, 113,
                            114, 115, 116, 117, 118, 119, 120, 121, 122,
                            163]

    # ['0', '1', '2', 'B', 'D', 'E', 'I', 'K', 'L', 'P', 'W',
    # 'b', 'd', 'f', 'i', 'm', 'o', 'p', 's', 'v']
    micro_version_classes_unicode = [
        48, 49, 50, 66, 68, 69, 73, 75, 76, 80, 87, 98, 100, 102, 105,
        109, 111, 112, 115, 118]

    def __init__(self, train_split=True,
                 enforce_all_alphanumeric_classes=False,
                 only_keep_test_set_classes=False,
                 only_keep_shared_classes=False,
                 only_alphanumeric=False,
                 dataset_root="ICDAR_2003_character_dataset",
                 training_set_folder="TrialTrain_Set_6185_Characters",
                 test_set_folder="TrialTest_Set_5430_Characters",
                 micro_split="train"):
        super().__init__()

        assert micro_split in ["train", "val", "test"]

        # enforce_all_alphanumeric_classes overrides attributes
        if enforce_all_alphanumeric_classes:
            only_alphanumeric = True
            only_keep_test_set_classes = False
            only_keep_shared_classes = False

        assert not (only_keep_test_set_classes is True and
                    only_keep_shared_classes is True), \
            "only_keep_test_set_classes and only_keep_shared_classes cannot be both True"

        self.train_split = train_split

        self.enforce_all_alphanumeric_classes = enforce_all_alphanumeric_classes
        self.only_keep_test_set_classes = only_keep_test_set_classes
        self.only_keep_shared_classes = only_keep_shared_classes
        self.only_alphanumeric = only_alphanumeric

        self.use_micro_version = True
        self.micro_split = micro_split

        if self.micro_split in ["train", "val"]:
            self.train_split = True
        elif self.micro_split in ["test"]:
            self.train_split = False

        self.dataset_root = dataset_root
        self.training_set_folder = training_set_folder
        self.test_set_folder = test_set_folder

        if self.train_split:
            self.folder = os.path.join(
                self.dataset_root, self.training_set_folder)
        else:
            self.folder = os.path.join(
                self.dataset_root, self.test_set_folder)

        self.df = self.convert_ICDAR_2013_char_labels()
        self.set_classes_attributes()
        self.unicode2label = self.get_unicode2label_dict()
        self.items = self.df2items()

        if self.use_micro_version:
            # modifies self.items so that each class contains
            # 15/5/20 examples per class for train/val/test split
            self.prune_micro_examples()
            
        self.classes_ = self.get_pytorch_classes_attributes()
        
        # necessary to build HuggingFace dataset
        self.add_raw_labels()

    def set_classes_attributes(self):
        """
        self.classes contains a sorted list of classes 
        (in the form of rendered characters)
        """
        if self.use_micro_version:
            self.classes = sorted(self.df["raw_label"].unique().tolist())
        elif self.enforce_all_alphanumeric_classes:
            self.classes = [chr(x) for x in self.alphanumeric]
        else:
            self.classes = sorted(self.df["raw_label"].unique().tolist())

        self.n_classes = len(self.classes)
        self.classes_unicode = [ord(x) for x in self.classes]

    def get_pytorch_classes_attributes(self):
        """
        classes_ is usually called classes in torchvision datasets
        classes_ is a list, not numpy array nor torch tensor
        
        classes_ has the same length as the total number of examples
        """
        classes_ = []
        for idx in range(len(self.items)):
            X, y = self.items[idx]
            classes_.append(y)
        return classes_

    def get_unicode2label_dict(self):
        unicode2label = dict()
        for unicode_ in self.classes_unicode:
            if unicode_ not in unicode2label:
                unicode2label[unicode_] = len(unicode2label)
        return unicode2label

    def convert_ICDAR_2013_char_labels(self):
        """
        file: 
            char/1/1.jpg, char/1/2.jpg, char/1/3.jpg, ..., char/62/6181.jpg
        path:
            ICDAR_2003_character_dataset/TrialTrain_Set_6185_Characters/char/1/1.jpg,
            ICDAR_2003_character_dataset/TrialTrain_Set_6185_Characters/char/1/2.jpg, ...
        raw_label:
            s, e, l, f, a, ..., S, O, R, ..., 3, 1, 4, A, m, i, e, a, r
        raw_label_unicode:
            115, 101, 108, 102, 97, ...
        """
        df = []
        tree = ET.parse(os.path.join(self.folder, "char.xml"))
        root = tree.getroot()
        for i in range(len(root)):
            attrib_ = root[i].attrib
            df.append([attrib_["file"],
                       os.path.join(self.folder, attrib_["file"]),
                       attrib_["tag"]])
        df = pd.DataFrame(df, columns=["file", "path", "raw_label"])
        df["raw_label_unicode"] = df["raw_label"].apply(lambda x: ord(x))

        df = self.clean_classes(df)

        return df

    def clean_classes(self, df):
        if self.use_micro_version:
            df = df.drop(df[~df["raw_label_unicode"].isin(
                self.micro_version_classes_unicode)].index)
        else:
            if self.only_keep_test_set_classes:
                df = df.drop(df[~df["raw_label_unicode"].isin(
                    self.all_test_set_classes)].index)
            if self.only_keep_shared_classes:
                df = df.drop(df[~df["raw_label_unicode"].isin(
                    self.shared_classes_unicode)].index)
            if self.only_alphanumeric:
                df = df.drop(df[~df["raw_label_unicode"].isin(
                    self.alphanumeric)].index)
        df.reset_index(drop=True, inplace=True)
        return df

    def df2items(self):
        items = []
        for idx in range(self.df.shape[0]):
            img_path = self.df.loc[idx, "path"]
            target_unicode = self.df.loc[idx, "raw_label_unicode"]
            target = self.unicode2label[target_unicode]
            items.append([img_path, target])
        return items

    def prune_micro_examples(self,
                             train_examples_per_class=15,
                             val_examples_per_class=5,
                             test_examples_per_class=20):
        """
        modifies self.items so that each class contains 
        15/5/20 examples per class for train/val/test split
        """
        # reshuffle to break the order of examples or classes
        rng = np.random.default_rng(88)
        rng.shuffle(self.items)

        items_per_class = defaultdict(list)

        for img_path, target in self.items:
            items_per_class[target].append([img_path, target])

        pruned_items = []
        for class_ in items_per_class.keys():
            if self.micro_split == "train":
                pruned_items.extend(
                    items_per_class[class_][:train_examples_per_class])
            elif self.micro_split == "val":
                pruned_items.extend(
                    items_per_class[class_][train_examples_per_class:
                                            (train_examples_per_class + val_examples_per_class)])
            elif self.micro_split == "test":
                # micro test split is from the original test split,
                # micro train, micro val splits are from the original train split,
                # so it is OK to just take the first test_examples_per_class examples.
                pruned_items.extend(
                    items_per_class[class_][:test_examples_per_class])

        # reshuffle to break the order of examples or classes
        rng = np.random.default_rng(66)
        rng.shuffle(pruned_items)

        self.items = pruned_items

    def __getitem__(self, idx):
        img_path, target = self.items[idx]
        return img_path, target

    def __len__(self):
        return len(self.items)
    
    def add_raw_labels(self):
        """
        useful for HF datasets' ClassLabel
        """
        tuples = []
        for raw_label_unicode, label in self.unicode2label.items():
            tuples.append((raw_label_unicode, label))
        # make sure that the order is correct
        tuples = sorted(tuples, key=lambda x: x[1])

        self.raw_labels = []
        for raw_label_unicode, label in tuples:
            self.raw_labels.append(raw_label_unicode)


