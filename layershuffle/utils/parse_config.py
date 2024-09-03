from transformers import ViTForImageClassification, ViTImageProcessor, DeiTImageProcessor, ViTConfig
from ..models import PositionPredictingViTForImageClassification, PositionPredictingDeiTForImageClassification, PositionEncodingViTForImageClassification, PositionEncodingDeiTForImageClassification, ShufflingViTForImageClassification,ShufflingDeiTForImageClassification
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

def parse_preprocessor(config):
    # FIXXME: this is fine for now, but it doesn't scale
    prep_dict = {"ViTImageProcessor" : ViTImageProcessor, "DeiTImageProcessor": DeiTImageProcessor}
    prep_class = prep_dict[config.MODEL.PREPNAME]
    preprocessor = prep_class.from_pretrained(config.MODEL.CONF)
    return preprocessor


def parse_model(config):
    # FIXXME: this is fine for now, but it doesn't scale
    model_dict = {"ViTForImageClassification": ViTForImageClassification,
                  "PositionPredictingViTForImageClassification" : PositionPredictingViTForImageClassification,
                  "PositionPredictingDeiTForImageClassification": PositionPredictingDeiTForImageClassification,
                  "PositionEncodingViTForImageClassification" : PositionEncodingViTForImageClassification,
                  "PositionEncodingDeiTForImageClassification" : PositionEncodingDeiTForImageClassification,
                  "ShufflingViTForImageClassification" : ShufflingViTForImageClassification,
                  "ShufflingDeiTForImageClassification" : ShufflingDeiTForImageClassification}
    model_class = model_dict[config.MODEL.NAME]

    if config.MODEL.PRETRAINED:
        model = model_class.from_pretrained(config.MODEL.CONF)
    else:
        print(ViTConfig.from_pretrained(config.MODEL.CONF))
        #model = model_class(ViTConfig(hidden_size = 768, num_hidden_layers = 12, num_attention_heads = 12, intermediate_size = 3072, hidden_act = 'gelu', hidden_dropout_prob = 0.0, attention_probs_dropout_prob = 0.0, initializer_range = 0.02, layer_norm_eps = 1e-12, image_size = 224, patch_size = 16, num_channels = 3, qkv_bias = True, encoder_stride = 16))
        model = model_class(ViTConfig.from_pretrained(config.MODEL.CONF))

    if config.MODEL.WEIGHTS is not None:
        load_model_weights(model, config.MODEL.WEIGHTS)
    model.vit.encoder.shuffle = config.MODEL.SHUFFLE
    return model

def parse_dataset(config,processor):
    # FIXXME: same here, no harm in making it simple but it doesn't scale
    image_mean = processor.image_mean
    image_std = processor.image_std
    normalize = Normalize(mean=image_mean, std=image_std)
    # TODO there is probably a better way to read out the image size directly from hf config (reading from prep is faulty)
    size = config.MODEL.IMG_SIZE

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

    if config.DATA.DATASET=="CIFAR-10":
        train_set = torchvision.datasets.CIFAR10(root=config.DATA.TRAIN_PATH, train=True, download=True, transform=_train_transforms)
        val_set = torchvision.datasets.CIFAR10(root=config.DATA.TEST_PATH, train=False, download=True, transform=_val_transforms)
    elif config.DATA.DATASET=="CIFAR-100":
        train_set = torchvision.datasets.CIFAR100(root=config.DATA.TRAIN_PATH, train=True, download=True, transform=_train_transforms)
        val_set = torchvision.datasets.CIFAR100(root=config.DATA.TEST_PATH, train=False, download=True, transform=_val_transforms)
    else: # default dataset is imagenet
        train_set = torchvision.datasets.ImageFolder(root=config.DATA.TRAIN_PATH, transform=_train_transforms)
        val_set = torchvision.datasets.ImageFolder(root=config.DATA.TEST_PATH, transform=_val_transforms)

    return train_set,val_set