# === import necessary modules ===
import src.config as config
import src.trainer as trainer
import src.trainer.stats as trainer_stats

# === import necessary external modules ===
from typing import Dict, Optional, Tuple
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers
import torch

"""
Whisper model for audio classification using synthetic data.
Uses WhisperForAudioClassification from HuggingFace Transformers.
"""


def init_whisper_processor():
    processor = transformers.AutoProcessor.from_pretrained("openai/whisper-tiny")
    return processor


def init_whisper_model(num_labels=10):
    model_config = transformers.AutoConfig.from_pretrained("openai/whisper-tiny")
    model_config.num_labels = num_labels
    model = transformers.WhisperForAudioClassification.from_pretrained("openai/whisper-tiny", config=model_config)
    return model


def process_dataset(conf: config.Config, processor: transformers.WhisperProcessor, dataset: data.Dataset) -> data.Dataset:
    return dataset


def init_whisper_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=conf.learning_rate)


def pre_init_whisper(conf: config.Config, dataset: data.Dataset) -> Tuple[transformers.PreTrainedModel, data.Dataset, transformers.WhisperProcessor]:
    num_labels = conf.data_configs.synthetic_whisper.num_labels
    processor = init_whisper_processor()
    model = init_whisper_model(num_labels=num_labels)
    dataset = process_dataset(conf, processor, dataset)
    return model, dataset, processor


def simple_trainer(conf: config.Config, model: transformers.WhisperForAudioClassification, dataset: data.Dataset, processor: transformers.WhisperProcessor) -> Tuple[trainer.Trainer, Optional[Dict]]:
    def collate_fn(batch):
        input_features = torch.stack([item["input_features"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_features": input_features, "labels": labels}

    loader = data.DataLoader(dataset, batch_size=conf.batch_size, collate_fn=collate_fn)
    model = model.cuda()
    optimizer = init_whisper_optim(conf, model)
    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader),
    )

    return trainer.SimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=model.device,
        stats=trainer_stats.init_from_conf(conf=conf, device=model.device, num_train_steps=len(loader))
    ), None


def whisper_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    model, dataset, processor = pre_init_whisper(conf, dataset)
    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset, processor)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")
