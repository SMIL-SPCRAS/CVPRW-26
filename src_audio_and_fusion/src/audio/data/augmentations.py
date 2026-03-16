import math
import torch
import torch.nn.functional as F
import torchaudio

def rand_uniform(a: float, b: float) -> float:
    return float(a + (b - a) * torch.rand(1).item())


def apply_gain(x: torch.Tensor, db_min: float, db_max: float) -> torch.Tensor:
    db = rand_uniform(db_min, db_max)
    gain = 10.0 ** (db / 20.0)
    return x * gain


def add_noise_snr(x: torch.Tensor, snr_db_min: float, snr_db_max: float) -> torch.Tensor:
    # white noise
    snr_db = rand_uniform(snr_db_min, snr_db_max)
    sig_power = x.pow(2).mean().clamp_min(1e-8)
    noise = torch.randn_like(x)
    noise_power = noise.pow(2).mean().clamp_min(1e-8)

    snr = 10.0 ** (snr_db / 10.0)
    scale = torch.sqrt(sig_power / (snr * noise_power))
    return x + noise * scale


def simple_lowpass_fft(x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    # keep_ratio in (0,1], keeps low freqs
    # x: [T]
    X = torch.fft.rfft(x)
    n = X.shape[0]
    k = max(1, int(n * keep_ratio))
    mask = torch.zeros_like(X)
    mask[:k] = 1
    y = torch.fft.irfft(X * mask, n=x.shape[0])
    return y


def simple_bandlimit(x: torch.Tensor) -> torch.Tensor:
    # randomly do lowpass or "telephone" band
    r = torch.rand(1).item()
    if r < 0.5:
        # lowpass keep 15-45% spectrum
        keep = rand_uniform(0.15, 0.45)
        return simple_lowpass_fft(x, keep)
    else:
        # telephone-ish: bandpass via lowpass(high) - lowpass(low)
        keep_hi = rand_uniform(0.18, 0.5)
        keep_lo = rand_uniform(0.03, 0.12)
        hi = simple_lowpass_fft(x, keep_hi)
        lo = simple_lowpass_fft(x, keep_lo)
        return (hi - lo)


def simple_reverb(x: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    # very lightweight synthetic RIR: exponentially decaying noise
    # good enough as augmentation without external libs
    t = x.shape[0]
    rir_len = int(rand_uniform(0.05, 0.20) * sr)  # 50-200 ms
    rir_len = max(16, min(rir_len, t))
    decay = rand_uniform(2.0, 8.0)
    n = torch.randn(rir_len, device=x.device, dtype=x.dtype)
    env = torch.exp(-torch.linspace(0, decay, rir_len, device=x.device, dtype=x.dtype))
    rir = n * env
    rir = rir / (rir.abs().sum().clamp_min(1e-6))

    y = F.conv1d(x[None, None, :], rir[None, None, :], padding=rir_len - 1)[0, 0, :t]
    mix = rand_uniform(0.1, 0.4)  # wet mix
    out = (1.0 - mix) * x + mix * y
    return out


class SpecAugment:
    """
    Simple SpecAugment: freq mask + time mask on log-mel.
    Expects spec [M, T] (float).
    """
    def __init__(
        self,
        p: float = 0.8,
        time_mask_param: int = 40,
        freq_mask_param: int = 12,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        self.p = float(p)
        self.time_mask_param = int(time_mask_param)
        self.freq_mask_param = int(freq_mask_param)
        self.num_time_masks = int(num_time_masks)
        self.num_freq_masks = int(num_freq_masks)

        self._tm = torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask_param)
        self._fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_param)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return spec

        # torchaudio masking expects [*, freq, time]
        x = spec.unsqueeze(0)  # [1, M, T]
        for _ in range(self.num_freq_masks):
            x = self._fm(x)
        for _ in range(self.num_time_masks):
            x = self._tm(x)
        return x.squeeze(0)
