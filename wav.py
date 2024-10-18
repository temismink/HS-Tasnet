import hashlib
import math
import json
import os
from pathlib import Path
import tqdm

import musdb
import julius
import torch as th
from torch import distributed
import torchaudio as ta
from torch.nn import functional as F

def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

class Wavset:
    def __init__(
            self,
            root, metadata, sources,
            segment=None, shift=None, normalize=True,
            samplerate=44100, channels=2, ext=EXT):
        """
        Waveset (or mp3 set for that matter). Can be used to train
        with arbitrary sources. Each track should be one folder inside of path.
        The folder should contain files named {source}.{ext}.

        Args:
            root (Path or str): root folder for the dataset.
            metadata (dict): output from build_metadata.
            sources (list[str]): list of source names.
            segment (None or float): segment length in seconds. If None, returns entire tracks.
            shift (None or float): stride in seconds bewteen samples.
            normalize (bool): normalizes input audio, **based on the metadata content**,
                i.e. the entire track is normalized, not individual extracts.
            samplerate (int): target sample rate. if the file sample rate
                is different, it will be resampled on the fly.
            channels (int): target nb of channels. if different, will be
                changed onthe fly.
            ext (str): extension for audio files (default is .wav).

        samplerate and channels are converted on the fly.
        """
        self.root = Path(root)
        self.metadata = OrderedDict(metadata)
        self.segment = segment
        self.shift = shift or segment
        self.normalize = normalize
        self.sources = sources
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext
        self.num_examples = []
        for name, meta in self.metadata.items():
            track_duration = meta['length'] / meta['samplerate']
            if segment is None or track_duration < segment:
                examples = 1
            else:
                examples = int(math.ceil((track_duration - self.segment) / self.shift) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def get_file(self, name, source):
        return self.root / name / f"{source}{self.ext}"

    def __getitem__(self, index):
        for name, examples in zip(self.metadata, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            meta = self.metadata[name]
            num_frames = -1
            offset = 0
            if self.segment is not None:
                offset = int(meta['samplerate'] * self.shift * index)
                num_frames = int(math.ceil(meta['samplerate'] * self.segment))
            wavs = []
            for source in self.sources:
                file = self.get_file(name, source)
                wav, _ = ta.load(str(file), frame_offset=offset, num_frames=num_frames)
                wav = convert_audio_channels(wav, self.channels)
                wavs.append(wav)

            example = th.stack(wavs)
            example = julius.resample_frac(example, meta['samplerate'], self.samplerate)
            if self.normalize:
                example = (example - meta['mean']) / meta['std']
            if self.segment:
                length = int(self.segment * self.samplerate)
                example = example[..., :length]
                example = F.pad(example, (0, length - example.shape[-1]))
            return example


def get_wav_datasets(args, name='wav'):
    """Extract the wav datasets from the XP arguments."""
    path = getattr(args, name)
    sig = hashlib.sha1(str(path).encode()).hexdigest()[:8]
    metadata_file = Path(args.metadata) / ('wav_' + sig + ".json")
    train_path = Path(path) / "train"
    valid_path = Path(path) / "valid"
    if not metadata_file.is_file() and distrib.rank == 0:
        metadata_file.parent.mkdir(exist_ok=True, parents=True)
        train = build_metadata(train_path, args.sources)
        valid = build_metadata(valid_path, args.sources)
        json.dump([train, valid], open(metadata_file, "w"))
    if distrib.world_size > 1:
        distributed.barrier()
    train, valid = json.load(open(metadata_file))
    if args.full_cv:
        kw_cv = {}
    else:
        kw_cv = {'segment': args.segment, 'shift': args.shift}
    train_set = Wavset(train_path, train, args.sources,
                       segment=args.segment, shift=args.shift,
                       samplerate=args.samplerate, channels=args.channels,
                       normalize=args.normalize)
    valid_set = Wavset(valid_path, valid, [MIXTURE] + list(args.sources),
                       samplerate=args.samplerate, channels=args.channels,
                       normalize=args.normalize, **kw_cv)
    return train_set, valid_set