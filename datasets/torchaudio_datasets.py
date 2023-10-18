import torchaudio
import torch
import sentencepiece as spm
from torch.utils.data import Dataset
import os
import numpy as np

class SpeechCommandDatasetWrapper(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root, url: str = "speech_commands_v0.02", folder_in_archive: str = "SpeechCommands", download: bool = False, subset = None,
                 classes_to_include = None) -> None:
        super().__init__(root, url, folder_in_archive, download, subset)
        if classes_to_include is None:
            classes_to_include = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                                  'backward', 'no', 'visual', 'marvin', 'cat', 'sheila', 'bed', 'tree', 'forward',
                                  'yes', 'stop', 'wow', 'on', 'down', 'dog', 'happy', 'learn', 'go', 'right',
                                  'bird', 'left', 'follow', 'off', 'up', 'house']
        self._walker = [w for w in self._walker if w.split('/')[-2] in classes_to_include]
        self.class_to_idx = {c:i for i, c in enumerate(classes_to_include)}
        self.targets = [self.class_to_idx[w.split('/')[-2]] for w in self._walker]
    def __getitem__(self, n: int):
        outputs = super().__getitem__(n)
        wav = outputs[0]
        if wav.shape[-1] < 16_000:
            wav = torch.nn.functional.pad(wav, (0, 16_000 - wav.shape[-1]))
        label = self.class_to_idx[outputs[2]]
        return wav, label

class LibrispeechDatasetWrapper(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, root, url: str = 'train-clean-100', folder_in_archive: str = 'LibriSpeech', download: bool = False) -> None:
        super().__init__(root, url, folder_in_archive, download)
    
    def __getitem__(self, n: int):
        e = super().__getitem__(n)
        wav = e[0]
        transcript = e[2]
        return wav, transcript

class LibrispeechFilelistDataset(Dataset):
    _ext_txt = ".txt"
    _ext_audio = ".wav"
    
    def __init__(self, root, split='train', transform=None, sp_file='ls_train-sp2-v650.model') -> None:
        super().__init__()
        with open(os.path.join(root, f'{split}_paths.txt'), 'r') as f:
            self._walker = [l.strip().split('/')[-1].split('.')[0] for l in f.readlines()]
        self.wav_dir = os.path.join(root, split, 'wav')
        self.txt_dir = os.path.join(root, split, 'txt')
        print(f'loading sentencepiece model from {sp_file}')
        self.tokenizer = spm.SentencePieceProcessor(os.path.join(root, sp_file))
        self.transform = transform
        # self._walker = sorted(str(p.stem) for p in Path(self.wav_dir).glob("*" + self._ext_audio))
    
    def __getitem__(self, n):
        fileid = self._walker[n]
        wavpath = os.path.join(self.wav_dir, f"{fileid}{self._ext_audio}")
        txtpath = os.path.join(self.txt_dir, f"{fileid}{self._ext_txt}")
        if not (os.path.exists(txtpath) and os.path.exists(wavpath)):
            return self.__getitem__(np.random.randint(0, len(self._walker)))
        with open(txtpath, 'r') as f:
            txt = f.read().strip()
        txt = self.tokenizer.encode(txt, out_type=int)
        wav, _ = torchaudio.load(wavpath)
        if self.transform is not None:
            wav = self.transform(wav)
        return wav, txt
    
    def __len__(self):
        return len(self._walker)