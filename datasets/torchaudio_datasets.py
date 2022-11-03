import torchaudio
import torch

class SpeechCommandDatasetWrapper(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root, url: str = "speech_commands_v0.02", folder_in_archive: str = "SpeechCommands", download: bool = False, subset = None) -> None:
        super().__init__(root, url, folder_in_archive, download, subset)
        classes_to_include = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
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