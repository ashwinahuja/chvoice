import torchaudio as ta
import torch as T

class TorchDSP:

    def __init__(self, n_fft, hop_len=None):
        self.n_fft = n_fft
        self.hop_len = n_fft//2 if hop_len is None else hop_len
        self.stft = ta.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=self.hop_len,
            win_length=n_fft,
            power=None
        )
        self.amplitude_to_db = ta.transforms.AmplitudeToDB()

    def _DB_to_amplitude(self, x):
        return T.pow(T.pow(10.0, 0.1 * x), 1.)

    def sig_to_db_phase(self, sig):
        x = self.stft(sig)
        mag, phase = ta.functional.magphase(x)
        db = self.amplitude_to_db(mag)
        return db, phase

    def db_phase_to_sig(self, db, phase):
        mag = self._DB_to_amplitude(db)
        real = mag * T.cos(phase)
        imag = mag * T.sin(phase)
        xstft = T.stack((real, imag), dim=-1)
        sig = xstft.istft(n_fft=self.n_fft, hop_length=self.hop_len)
        return sig