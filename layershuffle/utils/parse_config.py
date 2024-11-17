from transformers import ViTForImageClassification, DeiTForImageClassification, ViTImageProcessor, DeiTImageProcessor, ViTConfig, DeiTConfig
from ..models import PositionPredictingViTForImageClassification, PositionPredictingDeiTForImageClassification, PositionEncodingViTForImageClassification, PositionEncodingDeiTForImageClassification, ShufflingViTForImageClassification,ShufflingDeiTForImageClassification
from safetensors import safe_open

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from torch.utils.data import Subset

import torchvision
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    Resize,
                                    ToTensor,
                                    Lambda)


def load_model_weights(model, checkpoint_path):
    state_dict = {}
    with safe_open(checkpoint_path ,framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    model.load_state_dict(state_dict)

def parse_preprocessor(config):
    # FIXXME: this is fine for now, but it doesn't scale
    prep_dict = {"ViTImageProcessor" : ViTImageProcessor, "DeiTImageProcessor": DeiTImageProcessor}
    prep_class = prep_dict[config.MODEL.PREPNAME]
    preprocessor = prep_class.from_pretrained(config.MODEL.CONF)
    return preprocessor


def parse_model(config):
    # FIXXME: this is fine for now, but it doesn't scale
    model_dict = {"ViTForImageClassification": ViTForImageClassification,
                  "DeiTForImageClassification": DeiTForImageClassification,
                  "PositionPredictingViTForImageClassification" : PositionPredictingViTForImageClassification,
                  "PositionPredictingDeiTForImageClassification": PositionPredictingDeiTForImageClassification,
                  "PositionEncodingViTForImageClassification" : PositionEncodingViTForImageClassification,
                  "PositionEncodingDeiTForImageClassification" : PositionEncodingDeiTForImageClassification,
                  "ShufflingViTForImageClassification" : ShufflingViTForImageClassification,
                  "ShufflingDeiTForImageClassification" : ShufflingDeiTForImageClassification}
    model_class = model_dict[config.MODEL.NAME]
    config_dict = {"ViTForImageClassification": ViTConfig,
                   "DeiTForImageClassification": DeiTConfig,
                  "PositionPredictingViTForImageClassification" : ViTConfig,
                  "PositionPredictingDeiTForImageClassification": DeiTConfig,
                  "PositionEncodingViTForImageClassification" : ViTConfig,
                  "PositionEncodingDeiTForImageClassification" : DeiTConfig,
                  "ShufflingViTForImageClassification" : ViTConfig,
                  "ShufflingDeiTForImageClassification" : DeiTConfig}

    if config.MODEL.PRETRAINED:
        model = model_class.from_pretrained(config.MODEL.CONF)
    else:
        model = model_class(config_dict[config.MODEL.NAME].from_pretrained(config.MODEL.CONF))

    if config.MODEL.WEIGHTS is not None:
        load_model_weights(model, config.MODEL.WEIGHTS)
    #FIXXXXME: this is hacky but as always, we are short on time
    if "DeiT" in  config.MODEL.NAME:
        model.deit.encoder.shuffle = config.MODEL.SHUFFLE
    else:
        model.vit.encoder.shuffle = config.MODEL.SHUFFLE
    return model

def parse_dataset(config,processor):
    # FIXXME: same here, no harm in making it simple but it doesn't scale
    image_mean = processor.image_mean
    print(image_mean)
    image_std = processor.image_std
    print(image_std)
    normalize = Normalize(mean=image_mean, std=image_std)
    # TODO there is probably a better way to read out the image size directly from hf config (reading from prep is faulty)
    size = config.MODEL.IMG_SIZE

    _train_transforms = Compose(
        [
            Lambda(lambda img:img.convert("RGB")),
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Lambda(lambda img:img.convert("RGB")),
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    if config.DATA.DATASET=="CIFAR-10":
        train_set = torchvision.datasets.CIFAR10(root=config.DATA.TRAIN_PATH, train=True, download=True, transform=_train_transforms)
        val_set = torchvision.datasets.CIFAR10(root=config.DATA.TEST_PATH, train=False, download=True, transform=_val_transforms)
    elif config.DATA.DATASET=="CIFAR-100":
        train_set = torchvision.datasets.CIFAR100(root=config.DATA.TRAIN_PATH, train=True, download=True, transform=_train_transforms)
        val_set = torchvision.datasets.CIFAR100(root=config.DATA.TEST_PATH, train=False, download=True, transform=_val_transforms)
    elif config.DATA.DATASET=="Caltech256":
        dataset = torchvision.datasets.Caltech256(root=config.DATA.TRAIN_PATH, download=True,transform=_val_transforms)
        labels = [label for label in dataset.y] 
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
        train_indices, test_indices = next(stratified_split.split(np.zeros(len(labels)), labels))

        # Create subsets for training and testing
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, test_indices)

    elif config.DATA.DATASET=="EuroSAT":
        dataset = torchvision.datasets.EuroSAT(root=config.DATA.TRAIN_PATH, download=True,transform=_val_transforms)
        labels = [label for label in dataset.targets] 
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
        train_indices, test_indices = next(stratified_split.split(np.zeros(len(labels)), labels))

        # Create subsets for training and testing
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, test_indices)

        
    else: # default dataset is imagenet
        train_set = torchvision.datasets.ImageFolder(root=config.DATA.TRAIN_PATH, transform=_train_transforms)
        val_set = torchvision.datasets.ImageFolder(root=config.DATA.TEST_PATH, transform=_val_transforms)

    return train_set,val_set
