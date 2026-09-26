import gc
import io
import numpy as np
import os
import psutil
import tarfile
import torch
import torch.nn.functional as F
import webdataset as wds

from transformers.image_utils import load_image
from torch.utils.data import Dataset
from tqdm import tqdm

import io
import tarfile
import random
import threading
from queue import Queue

import numpy as np
import torch
import torch.nn.functional as F

IMG_EXT = "img_embedding.pth"
TXT_EXT = "txt_embedding.pth"
LABEL_EXT = "label.npy"


def to_float(x, normalize=True):
    x = x.float()
    if normalize:
        x = F.normalize(x, dim=-1, eps=1e-8)
    return x


class StreamingTarDataset(torch.utils.data.IterableDataset):
    def __init__(self, tar_paths, label_transform, modality="full", seed=0,
                 key_filter_fn=None, shuffle=True, shuffle_buffer_size=10000):
        self.tar_paths = list(tar_paths)
        self.label_transform = label_transform
        self.modality = modality
        self.seed = seed
        self.key_filter_fn = key_filter_fn
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch


    def _needed_exts(self):
        exts = {LABEL_EXT}
        if self.modality in ("full", "img"):
            exts.add(IMG_EXT)
        if self.modality in ("full", "txt"):
            exts.add(TXT_EXT)
        return exts

    def _decode(self, key, blobs):
        try:
            parts = [key]
            if self.modality in ("full", "img"):
                parts.append(to_float(torch.load(
                    io.BytesIO(blobs[IMG_EXT]),
                    map_location="cpu", weights_only=True)))
            if self.modality in ("full", "txt"):
                parts.append(to_float(torch.load(
                    io.BytesIO(blobs[TXT_EXT]),
                    map_location="cpu", weights_only=True)))
            label = self.label_transform(
                torch.from_numpy(
                    np.load(io.BytesIO(blobs[LABEL_EXT]))
                ).clone())
            parts.append(label)
            return tuple(parts)
        except KeyError as e:
            print(f"Warning: skipping sample '{key}' - missing {e}")
            return None


    def _iter_shard_raw(self, tar_path):
        needed = self._needed_exts()
        groups = {}
        skip_keys = set()

        with tarfile.open(tar_path, "r:") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                fname = member.name
                if ".img" in fname:
                    key, ext = fname.split(".img")[0], IMG_EXT
                elif ".txt" in fname:
                    key, ext = fname.split(".txt")[0], TXT_EXT
                elif ".label" in fname:
                    key, ext = fname.split(".label")[0], LABEL_EXT
                else:
                    continue

                if key in skip_keys:
                    continue
                if self.key_filter_fn and not self.key_filter_fn(key):
                    skip_keys.add(key)
                    groups.pop(key, None)
                    continue
                if ext not in needed:
                    continue

                if key not in groups:
                    groups[key] = {}
                groups[key][ext] = tf.extractfile(member).read()

                if needed.issubset(groups[key]):
                    yield key, groups.pop(key)


    @staticmethod
    def _shuffle_buffer(stream, buf_size, rng):
        buf = []
        for item in stream:
            buf.append(item)
            if len(buf) >= buf_size:
                idx = rng.randint(0, len(buf) - 1)
                yield buf[idx]
                buf[idx] = buf[-1]
                buf.pop()
        rng.shuffle(buf)
        yield from buf


    @staticmethod
    def _prefetch(iterable, maxsize=128):
        q = Queue(maxsize=maxsize)
        sentinel = object()
        error = [None]

        def producer():
            try:
                for item in iterable:
                    q.put(item)
            except Exception as e:
                error[0] = e
            finally:
                q.put(sentinel)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
        t.join()
        if error[0] is not None:
            raise error[0]


    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        paths = list(self.tar_paths)
        if self.shuffle:
            rng.shuffle(paths)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        def raw_stream():
            for p in paths:
                yield from self._iter_shard_raw(p)

        prefetched = self._prefetch(raw_stream())

        def decoded():
            for key, blobs in prefetched:
                sample = self._decode(key, blobs)
                if sample is not None:
                    yield sample

        if self.shuffle:
            yield from self._shuffle_buffer(
                decoded(), self.shuffle_buffer_size, rng)
        else:
            yield from decoded()


def make_dataset(tar_paths, label_transform, modality="full", seed=0,
                 key_filter_fn=None, shuffle=True, shuffle_buffer_size=5000):
    return StreamingTarDataset(
        tar_paths, label_transform, modality=modality, seed=seed,
        key_filter_fn=key_filter_fn, shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
    )

class QwenChatDataset(Dataset):
    def __init__(self, image_paths, messages):
        self.image_paths = image_paths
        self.messages = messages

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return self.image_paths[idx], self.messages[idx]

class ImagesDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = load_image(path)
        return path, image

class CaptionsDataset(Dataset):
    def __init__(self, captions_df):
        self.image_paths = captions_df["image_path"].tolist()
        self.captions = captions_df["captions"].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        caption = self.captions[idx]
        return path, caption

class FeaturesShardDataset(Dataset):
    def __init__(self, shard_paths):
        self.shard_paths = shard_paths

    def __len__(self):
        return len(self.shard_paths)

    def __getitem__(self, idx):
        shard_path = self.shard_paths[idx]
        features_dict = torch.load(shard_path)
        return features_dict

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tar_paths,
        label_transform,
        modality="full",
        filter_fn=None,
        disable_pbars=False,
        check_mem=False,
    ):
        self.samples = []
        self.proc = psutil.Process(os.getpid())
        self.check_mem = check_mem

        counter = 0
        for tar_path in tqdm(
            tar_paths, desc="Loading shards into memory", disable=disable_pbars
        ):
            if self.check_mem:
                torch.cuda.empty_cache()
                print(f"Counter={counter}, RSS={self.proc.memory_info().rss/1024**3:.1f}GB "
                    f"GPU={torch.cuda.memory_allocated()/1024**3:.1f}GB", flush=True)
            self._load_shard(tar_path, label_transform, modality, filter_fn)
            gc.collect()
            counter += 1

        print(f"Loaded {len(self.samples)} samples into memory")

    def _load_shard(self, tar_path, label_transform, modality, filter_fn):

        groups = {}
        with tarfile.open(tar_path, "r") as tf:
                    for member in tf:
                        if not member.isfile():
                            continue
                        fname = os.path.basename(member.name)
                        dot_pos = fname.find(".")
                        if dot_pos < 0:
                            continue
                        if ".img" in fname:
                            key = member.name.split(".img")[0]
                            ext = IMG_EXT
                        elif ".txt" in fname:
                            key = member.name.split(".txt")[0]
                            ext = TXT_EXT
                        elif ".label" in fname:
                            key = member.name.split(".label")[0]
                            ext = LABEL_EXT
                        else:
                            raise ValueError(f"Unexpected file in shard: {member.name}")

                        if key not in groups:
                            groups[key] = {}
                        groups[key][ext] = tf.extractfile(member).read()

        for key, blobs in groups.items():
            try:
                if modality in ("full", "img"):
                    img_emb = to_float(torch.load(
                        io.BytesIO(blobs[IMG_EXT]),
                        map_location="cpu", weights_only=True,
                    ))
                if modality in ("full", "txt"):
                    txt_emb = to_float(torch.load(
                        io.BytesIO(blobs[TXT_EXT]),
                        map_location="cpu", weights_only=True,
                    ))
                label_arr = np.load(io.BytesIO(blobs[LABEL_EXT]))
                label_raw = torch.from_numpy(label_arr).clone()
            except KeyError as e:
                print(f"Warning: skipping sample '{key}' - missing {e}")
                continue

            label = label_transform(label_raw)

            if modality == "full":
                sample = (key, img_emb, txt_emb, label)
            elif modality == "img":
                sample = (key, img_emb, label)
            else:
                sample = (key, txt_emb, label)

            if filter_fn is not None and not filter_fn(sample):
                continue

            self.samples.append(sample)

        del groups

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
