import json
import pdb
from importlib.resources import files
import os
from collections import defaultdict, Counter
import random

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk, concatenate_datasets
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        self.dialect_mapping = self.create_dialect_mapping()

        # self.dialect_mapping = {
        #     'chengdu':0,
        #     'gaoxiong':1,
        #     'shanghai':2,
        #     'shijiazhuang':3,
        #     'tianjin':4,
        #     'xian':5,
        #     'yue':6,
        #     'zh':7,
        #     'zhengzhou':8
        # }

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def create_dialect_mapping(self):
        # Generate a mapping list of dialect tags to integers automatically, and sort it in alphabetical order
        #dialect_labels = [item.get("dialect", 0) for item in self.data]
        dialect_labels = [item["dialect"] for item in self.data]
        return sorted(set(dialect_labels))

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def get_dialect(self,index):
        return self.data[index]["dialect"]

    def __len__(self):
        return len(self.data)
    def __getitem__pinjie(self, index):
        max_retries = 3
        for _ in range(max_retries):
            try:
                # main sample
                row1 = self.data[index]
                audio_path1 = row1["audio_path"]
                text1 = row1["text"]
                ori_text1 = row1["ori_text"]
                duration1 = row1["duration"]
                dialect1 = row1["dialect"]

                # random second sample
                index2 = random.randint(0, len(self.data) - 1)
                row2 = self.data[index2]
                audio_path2 = row2["audio_path"]
                text2 = row2["text"]
                ori_text2 = row2["ori_text"]
                duration2 = row2["duration"]
                dialect2 = row2["dialect"]

                # check legal duration
                if not (0.3 <= duration1 <= 30) or not (0.3 <= duration2 <= 30):
                    index = (index + 1) % len(self.data)
                    continue

                #  -> mel_spec1
                if self.preprocessed_mel:
                    mel_spec1 = torch.tensor(row1["mel_spec"])
                else:
                    audio1, sr1 = torchaudio.load(audio_path1)
                    if audio1.shape[0] > 1:
                        audio1 = torch.mean(audio1, dim=0, keepdim=True)
                    if sr1 != self.target_sample_rate:
                        resampler = torchaudio.transforms.Resample(sr1, self.target_sample_rate)
                        audio1 = resampler(audio1)
                    mel_spec1 = self.mel_spectrogram(audio1).squeeze(0)

                #  -> mel_spec2
                if self.preprocessed_mel:
                    mel_spec2 = torch.tensor(row2["mel_spec"])
                else:
                    audio2, sr2 = torchaudio.load(audio_path2)
                    if audio2.shape[0] > 1:
                        audio2 = torch.mean(audio2, dim=0, keepdim=True)
                    if sr2 != self.target_sample_rate:
                        resampler = torchaudio.transforms.Resample(sr2, self.target_sample_rate)
                        audio2 = resampler(audio2)
                    mel_spec2 = self.mel_spectrogram(audio2).squeeze(0)

                # concat at T [n_mel, T1] + [n_mel, T2] -> [n_mel, T1+T2]
                mel_spec = torch.cat([mel_spec1, mel_spec2], dim=1)

                # concat text
                text = text1 + " " + text2
                ori_text = ori_text1 + " " + ori_text2


                dialect = self.dialect_mapping.index(dialect1)

                return {
                    "mel_spec": mel_spec,        # [n_mel, T1 + T2]
                    "text": text,
                    "ori_text": ori_text,
                    "dialect": dialect,
                }

            except Exception as e:
                print(f"Error processing pair {index} + {index2}: {str(e)}")
                index = (index + 1) % len(self.data)
                continue

        # Return to single-sample fallback
        print(f"Failed to load pair after {max_retries} retries, falling back to single sample.")
        return self.get_single_item(index)
    def __getitem__(self, index):
        while True:
            try:
                row = self.data[index]
                audio_path = row["audio_path"]
                ori_text = row["ori_text"]
                text = row["text"]
                duration = row["duration"]
                dialect = row["dialect"]

                # filter by given length
                if 0.3 <= duration <= 30:
                    break  # valid

                index = (index + 1) % len(self.data)

            except Exception as e:
                print(f"Error processing index {index}: {str(e)}")
                index = (index + 1) % len(self.data)
                continue

        try:
            if self.preprocessed_mel:
                mel_spec = torch.tensor(row["mel_spec"])
            else:
                try:
                    audio, source_sample_rate = torchaudio.load(audio_path)
                except Exception as e:
                    print(f"Error loading audio file {audio_path}: {str(e)}")
                    # skip to the next sample
                    return self.__getitem__((index + 1) % len(self.data))

                # make sure mono input
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)

                # resample if necessary
                if source_sample_rate != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                    audio = resampler(audio)

                # to mel spectrogram
                mel_spec = self.mel_spectrogram(audio)
                mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'
                dialect = self.dialect_mapping.index(dialect)
                # dialect = self.dialect_mapping[dialect]

            return {
                "mel_spec": mel_spec,
                "text": text,
                "ori_text": ori_text,
                "dialect": dialect,
            }
            
        except Exception as e:
            print(f"Error processing audio {audio_path}: {str(e)}")
            # try next sample
            return self.__getitem__((index + 1) % len(self.data))


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source
        # sort by frame
        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        # batch_frames = 0
        # for idx, frame_len in tqdm(
        #     indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        # ):
        #     if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
        #         batch.append(idx)
        #         batch_frames += frame_len
        #     else:
        #         if len(batch) > 0:
        #             batches.append(batch)
        #         if frame_len <= self.frames_threshold:
        #             batch = [idx]
        #             batch_frames = frame_len
        #         else:
        #             batch = []
        #             batch_frames = 0
        max_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if max(max_frames, frame_len)*(len(batch)+1)  <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                max_frames = max(max_frames, frame_len)
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    max_frames = frame_len
                else:
                    batch = []
                    max_frames = 0
        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


class AutoRatioDialectSampler(Sampler):
    def __init__(self, sampler, frame_threshold, max_samples=0, random_seed=None, drop_residual=False):
        self.sampler = sampler
        self.frame_threshold = frame_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.drop_residual = drop_residual
        self.epoch = 0

        data_source = self.sampler.data_source

        # Construct a sample list corresponding to each dialect and determine the frame length
        self.dialect_to_samples = defaultdict(list)
        dialect_counter = Counter()

        for idx in tqdm(
            self.sampler, desc="counting dialect..."):
            dialect = data_source.get_dialect(idx)
            frame_len = data_source.get_frame_len(idx)
            self.dialect_to_samples[dialect].append((idx, frame_len))
            dialect_counter[dialect] += 1

        # Automatically calculate the proportion (in integer ratio) based on the number of samples
        min_count = min(dialect_counter.values())
        self.dialect_ratios = {k: max(1, v // min_count) for k, v in dialect_counter.items()}

        print(f"[AutoRatioSampler] Detected dialect ratios: {self.dialect_ratios}")

        # Samples within each dialect are sorted by the number of frames.
        for dialect in self.dialect_to_samples:
            self.dialect_to_samples[dialect].sort(key=lambda x: x[1])  # sort by frame len

        self.batches = self._build_batches()

    def _build_batches(self):
        dialect_iters = {d: iter(v) for d, v in self.dialect_to_samples.items()}

        # Construct an interleaved sequence template, for example ['a','b','c','a','b','a']
        sampling_pattern = []
        for dialect, count in self.dialect_ratios.items():
            sampling_pattern.extend([dialect] * count)


        # eg: ['a','b','c','a','b','a']
        batches = []
        batch = []
        batch_frames = 0
        max_flag = 0
        while True:
            added = False
            random.shuffle(sampling_pattern)
            for dialect in sampling_pattern:
                try:
                    idx, frame_len = next(dialect_iters[dialect])
                    if ((batch_frames + frame_len <= self.frame_threshold) and
                        (self.max_samples == 0 or len(batch) < self.max_samples)):
                        batch.append(idx)
                        batch_frames += frame_len
                        added = True
                    else:
                        # begin a new batch
                        if batch:
                            batches.append(batch)
                        if frame_len <= self.frame_threshold:
                            batch = [idx]
                            #flag
                            max_flag = max(max_flag,batch_frames)
                            batch_frames = frame_len
                            added = True
                        else:
                            #flag
                            max_flag = max(max_flag, batch_frames)
                            batch = []
                            batch_frames = 0
                except StopIteration:
                    continue

            if not added:
                break
        if not self.drop_residual and batch:
            batches.append(batch)
        print(f'create dynamic batches with dialect ratio and {self.frame_threshold} per gpu!')
        del self.dialect_to_samples
        del dialect_iters
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)



# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    arrow_name = None
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")
    print(dataset_name)
    if dataset_type == "CustomDataset":
        
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}"))
        
        if audio_type == "raw":
            print(f"{rel_data_path}/raw.arrow")
            # try:
            #     train_dataset = load_from_disk(f"{rel_data_path}/raw")
            # except:  # noqa: E722
            if os.path.exists(f"{rel_data_path}/raw.arrow"):
                train_dataset = Dataset_.from_file(f"{rel_data_path}/{arrow_name}.arrow")
            else:
                train_dataset = Dataset_.from_file(f"{rel_data_path}/{arrow_name}.arrow")
            # train_dataset1 = Dataset_.from_file(f"{rel_data_path}/zh.arrow")

            # train_dataset = concatenate_datasets([train_dataset, train_dataset1])
            print(len(train_dataset))
            print(train_dataset[20])

            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/{arrow_name}_duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        # with open(f"{rel_data_path}/zh_duration.json", "r", encoding="utf-8") as f:
        #     data_dict1 = json.load(f)
        durations = data_dict["duration"]
        # durations1 = data_dict1["duration"]
        # durations = durations + durations1


        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )
    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])
    ori_text = [item["ori_text"] for item in batch]
    dialect = [item["dialect"] for item in batch]
    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
        ori_text=ori_text,
        dialect = dialect,
    )
