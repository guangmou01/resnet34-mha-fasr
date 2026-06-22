# PATH: frontend/fbank.py

import torch
import torchaudio
import librosa

class PreEmphasis(torch.nn.Module):
    """
    Pre-emphasis filter used in DNN training, adapted from:
    https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py

    """
    def __init__(self,
                 coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter',
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(1)
        input = torch.nn.functional.pad(input, (1, 0), mode='reflect')
        return torch.nn.functional.conv1d(input, self.flipped_filter).squeeze(1)


def safe_load_audio(file_path: str,
                    target_sr: int = 8000):
    """
    Safely load an audio file and resample it.

    :param file_path: Path to the audio file.
    :param target_sr: Target sampling rate.
                      If None, keep the original sampling rate.

    :return waveform: Mono waveform with shape [1, T], dtype float32.
    :return sr: Sampling rate of the returned waveform.
    """
    try:
        # torchaudio.load
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.to(torch.float32)
    except Exception:
        # librosa.load
        data, sr = librosa.load(file_path, sr=None, mono=False)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data)
        waveform = waveform.to(torch.float32)

    # Merge channels to mono
    waveform = waveform.mean(dim=0, keepdim=True)

    # Resample the waveform
    if target_sr is not None and sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr
        )
        waveform = resampler(waveform)
        sr = target_sr

    return waveform, sr

def get_fbank(file_path: str) -> torch.Tensor:

    # 1. Load + resample the audio:
    waveform, sr = safe_load_audio(file_path=file_path,
                                   target_sr=8000)

    # 2. Pre-emphasis filter:
    pre_emphasis = PreEmphasis(0.97).to(waveform.device)
    waveform = pre_emphasis(waveform)

    # 3. Compute the fbank by torchaudio:
    fbank_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=8000,
        n_fft=512,
        win_length=200,     # 25ms window length
        hop_length=80,      # 10ms frame shift
        f_min=20,
        f_max=3900,
        window_fn=torch.hamming_window,
        n_mels=64
    )
    fbank = fbank_transform(waveform) + 1e-6
    fbank = fbank.log()

    # 4. Cepstral/spectrum mean and variance normalization (CMVN), see:
    # Viikki, O., & Laurila, K. (1998).
    # Cepstral domain segmental feature vector normalization for noise robust speech recognition.
    # Speech Communication, 25(1), 133–147.
    # https://doi.org/10.1016/S0167-6393(98)00033-8
    mean = fbank.mean(dim=-1, keepdim=True)
    std = fbank.std(dim=-1, keepdim=True)
    fbank = (fbank - mean) / (std + 1e-6)
    fbank = fbank.transpose(1, 2)

    return fbank

