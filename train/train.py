import os
import argparse
from config.defaults import get_cfg_defaults

from datasets import load_metric
from safetensors import safe_open

import torch,torchvision
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    Resize,
                                    ToTensor)

from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer

from liquidtransformers.utils import Experiment

metric = load_metric("accuracy")

def load_model_weights(model,checkpoint_path):
    state_dict = {}
    with safe_open(checkpoint_path ,framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    model.load_state_dict(state_dict)

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x[0] for x in batch]),
        'labels': torch.tensor([x[1] for x in batch])
    }

def parse_args():
    parser = argparse.ArgumentParser(
        prog='train',
        description='Train a model',
        epilog='')
    parser.add_argument('--config-path', help='path to config file')
    args = parser.parse_args()
    return args
def train_model():
    #config_path = parse_args().config_path
    config = get_cfg_defaults()
    #config.merge_from_file(config_path)
    config.freeze()

    experiment = Experiment(config.EXPERIMENT_LOG.BASEPATH, config.EXPERIMENT_LOG.MODEL_NAME, config.EXPERIMENT_LOG.EXPERIMENT_NAME)

    device = torch.device(f'cuda:{config.GPU_ID}') if torch.cuda.is_available() else torch.device("cpu")
    processor = ViTImageProcessor.from_pretrained(config.MODEL.CONF)
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

    train_set = torchvision.datasets.ImageFolder(root=config.DATA.TRAIN_PATH,transform=_train_transforms)
    val_set = torchvision.datasets.ImageFolder(root=config.DATA.TEST_PATH,transform=_val_transforms)

    for seed in range(config.RANDOM_SEED,config.RANDOM_SEED+config.NUM_RUNS):
        training_args = TrainingArguments(
            output_dir=os.path.join(experiment.modelpath,"run_{seed}"),
            seed=seed,
            data_seed=seed,
            per_device_train_batch_size=config.OPTIMIZER.BATCH_SIZE_TRAIN,#432 for single gpu, 320 for 4x
            per_device_eval_batch_size=config.OPTIMIZER.BATCH_SIZE_EVAL, # 10x train batch size
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=config.OPTIMIZER.EPOCHS,
            fp16=True,
            logging_strategy="epoch",
            learning_rate=config.OPTIMIZER.LR,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to='tensorboard',
            load_best_model_at_end=True,
            dataloader_num_workers=config.N_WORKERS)

        model = ViTForImageClassification.from_pretrained(config.MODEL.CONF)
        if config.MODEL.WEIGHTS is not None:
            load_model_weights(model, config.MODEL.WEIGHTS)
        model.to(device)
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=train_set,
            eval_dataset=val_set,
            tokenizer=processor,
        )

        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

if __name__ == '__main__':
    train_model()