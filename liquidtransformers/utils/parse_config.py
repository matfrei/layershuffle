
from transformers import ViTForImageClassification
from ..models import PositionPredictingViTForImageClassification,PositionEncodingViTForImageClassification,ShufflingViTForImageClassification
from safetensors import safe_open

import torchvision
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    Resize,
                                    ToTensor)

def load_model_weights(model, checkpoint_path):
    state_dict = {}
    with safe_open(checkpoint_path ,framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    model.load_state_dict(state_dict)

def parse_model(config):
    model_dict = {"ViTForImageClassification": ViTForImageClassification, "PositionPredictingViTForImageClassification": PositionPredictingViTForImageClassification,"PositionEncodingViTForImageClassification": PositionEncodingViTForImageClassification, "ShufflingViTForImageClassification":ShufflingViTForImageClassification}
    model_class = model_dict[config.MODEL.NAME]
    model = model_class.from_pretrained(config.MODEL.CONF)
    if config.MODEL.WEIGHTS is not None:
        load_model_weights(model, config.MODEL.WEIGHTS)
    model.vit.encoder.shuffle = config.MODEL.SHUFFLE

    return model

def parse_dataset(config,processor):
    image_mean = processor.image_mean
    image_std = processor.image_std
    normalize = Normalize(mean=image_mean, std=image_std)
    size = processor.size["height"]

    _train_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )


    if config.DATA.DATASET=="CIFAR-100":
        train_set = torchvision.datasets.CIFAR100(root=config.DATA.TRAIN_PATH, train=True, download=True, transform=_train_transforms)
        val_set = torchvision.datasets.CIFAR100(root=config.DATA.TEST_PATH, train=False, download=True, transform=_val_transforms)
    else:
        train_set = torchvision.datasets.ImageFolder(root=config.DATA.TRAIN_PATH, transform=_train_transforms)
        val_set = torchvision.datasets.ImageFolder(root=config.DATA.TEST_PATH, transform=_val_transforms)

    return train_set,val_set