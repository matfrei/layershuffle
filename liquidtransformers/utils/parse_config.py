from transformers import ViTForImageClassification
from ..models import PositionPredictingViTForImageClassification,PositionEncodingViTForImageClassification,ShufflingViTForImageClassification
from safetensors import safe_open

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