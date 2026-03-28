"""
Synthetic Whisper data loader - Milabench-style.

Based on https://github.com/mila-iqia/milabench/blob/master/benchmarks/huggingface/bench/synth.py

Uses the same generator pattern as Milabench's SyntheticData and
gen_AutoModelForAudioClassification:
- generators: dict of {name: gen} where each gen() produces one sample field
- gen(): invokes all generators to produce one full sample dict
- data: [gen() for _ in range(n)] pre-generated in memory
- repeat: multiplier; effective __len__ = n * repeat, __getitem__(i) = data[i % n]
"""
import torch
import torch.utils.data
import src.config as config
from transformers import WhisperFeatureExtractor

data_load_name = "synthetic_whisper_milabench"

SAMPLE_RATE = 16000
# Milabench gen_AutoModelForAudioClassification uses 10000 samples (~0.6s at 16kHz)
AUDIO_LENGTH = 10000


class SyntheticWhisperDataMilabench(torch.utils.data.Dataset):
    """
    Milabench-style synthetic Whisper dataset.

    Mirrors Milabench's SyntheticData + gen_AutoModelForAudioClassification:
    - generators dict with per-field callables (igen, ogen)
    - gen() returns {name: gen() for name, gen in generators.items()}
    - data = [gen() for _ in range(n)]
    - __getitem__(i) returns data[i % n], __len__ = n * repeat
    """

    def __init__(self, n: int, repeat: int, num_labels: int = 10):
        """
        Parameters
        ----------
        n : int
            Number of unique samples to generate and store in memory.
        repeat : int
            Multiplier for effective dataset size. __len__ = n * repeat.
        num_labels : int
            Number of classification labels (0 to num_labels-1).
        """
        self.n = n
        self.repeat = repeat
        self.num_labels = num_labels
        extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        self.generators = self._build_generators(extractor)
        self.data = [self.gen() for _ in range(n)]

    def _build_generators(self, extractor):
        """Build per-field generators as in Milabench gen_AutoModelForAudioClassification."""
        sampling_rate = SAMPLE_RATE
        num_labels = self.num_labels

        def igen():
            wav = list(torch.rand(AUDIO_LENGTH) * 2 - 1)
            dat = extractor(wav, sampling_rate=sampling_rate, return_tensors="pt")[
                "input_features"
            ]
            return dat[0]

        def ogen():
            return torch.randint(0, num_labels, ())

        return {"input_features": igen, "labels": ogen}

    def gen(self):
        """Produce one sample by invoking each field generator (Milabench pattern)."""
        return {name: gen() for name, gen in self.generators.items()}

    def __getitem__(self, i: int):
        return self.data[i % self.n]

    def __len__(self) -> int:
        return self.n * self.repeat


def load_data(conf: config.Config) -> torch.utils.data.Dataset:
    sc = conf.data_configs.synthetic_whisper_milabench
    n = getattr(sc, "n", 100)
    repeat = getattr(sc, "repeat", 1)
    num_labels = getattr(sc, "num_labels", 10)
    return SyntheticWhisperDataMilabench(n=n, repeat=repeat, num_labels=num_labels)
