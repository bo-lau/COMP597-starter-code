"""
Disk-backed synthetic Whisper classification data.

Use --data synthetic_whisper. Generates N_SAMPLES, caches features to .pt
(see data_path). Contrast with synthetic_whisper_milabench (in-memory, Milabench
pattern): docs/WHISPER_DATA_LOADING.md
"""
import os
import math
import torch
import torch.utils.data
import src.config as config
from transformers import WhisperFeatureExtractor

data_load_name = "synthetic_whisper"

N_SAMPLES = 5500
SAMPLE_RATE = 16000
PROGRESS_INTERVAL = 500
BATCH_SIZE = 64


def generate_samples(n, data_path, num_labels, batch_size=BATCH_SIZE):
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Using GPU for feature extraction (faster)")
    samples = []
    num_batches = math.ceil(n / batch_size)
    extract_kwargs = {"sampling_rate": SAMPLE_RATE, "return_tensors": "pt"}

    for batch_idx in range(num_batches):
        remaining = n - batch_idx * batch_size
        current_batch_size = min(batch_size, remaining)

        # Generate a batch of random waveforms in a single vectorized operation
        wavs = (torch.rand(current_batch_size, SAMPLE_RATE) * 2 - 1).tolist()

        # Run feature extraction in a single batched call
        try:
            batch_input_features = feature_extractor(wavs, device=device, **extract_kwargs)["input_features"]
        except TypeError:
            batch_input_features = feature_extractor(wavs, **extract_kwargs)["input_features"]

        # Generate labels for the whole batch at once
        batch_labels = torch.randint(0, num_labels, (current_batch_size,))

        for i in range(current_batch_size):
            samples.append({
                "input_features": batch_input_features[i],
                "labels": batch_labels[i]
            })

        generated = min(n, (batch_idx + 1) * batch_size)
        if generated % PROGRESS_INTERVAL == 0 or generated == n:
            print(f"Generated {generated}/{n} samples...")

    torch.save(samples, data_path)
    return samples


class SyntheticWhisperData(torch.utils.data.Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)


def load_data(conf: config.Config):
    sc = conf.data_configs.synthetic_whisper
    data_path = sc.data_path
    num_labels = getattr(sc, "num_labels", 10)
    force_regenerate = getattr(sc, "force_regenerate", 0)

    if os.path.exists(data_path) and not force_regenerate:
        print(f'=============================================================\nLoading Existing Data\n=============================================================')
        try:
            samples = torch.load(data_path, map_location="cpu", weights_only=False)
        except TypeError:
            samples = torch.load(data_path, map_location="cpu")
    else:
        if force_regenerate and os.path.exists(data_path):
            print(f'=============================================================\nForce Regenerate: Overwriting existing data\n=============================================================')
        else:
            print(f'=============================================================\nGenerating New Data\n=============================================================')
        samples = generate_samples(N_SAMPLES, data_path, num_labels)
    return SyntheticWhisperData(samples)
