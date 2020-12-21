import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
import torchvision
import numpy as np
from torchsummary import torchsummary
from IPython.display import Audio
import matplotlib.pyplot as plt
import random
import os
import pyaudio
from glob import glob
import math

class DSP:
    def __init__(self, n_fft=254, hop_len=None):
        """ signal processing utils using torchaudio
        """
        self.n_fft = n_fft
        self.hop_len = n_fft//2 if hop_len is None else hop_len
        self.stft = ta.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=self.hop_len,
            win_length=n_fft,
            power=None
        )
        self.amplitude_to_db = ta.transforms.AmplitudeToDB()
        self.db_to_amplitude = lambda x: T.pow(T.pow(10.0, 0.1 * x), 1.)

    def sig_to_db_phase(self, sig):
        """ get dB and phase spectrograms of signal
            example usage:
                >>> sig, sr = torchaudio.load('sound.wav')
                >>> db, phase = chvoice.sig_to_db_phase(sig)
        """
        # represent input signal in time-frequency domain
        stft = self.stft(sig)
        # magnitude = amount of power/volume for each phase = frequency
        mag, phase = ta.functional.magphase(stft)
        # put magnitudes on log scale
        db = self.amplitude_to_db(mag)

        return db, phase

    def db_phase_to_sig(self, db, phase):
        """ get wav signal from db and phase spectrograms.
            example usage:
                >>> sig, sr = torchaudio.load('sound.wav')
                >>> db, phase = chvoice.sig_to_db_phase(sig)
                    ... do stuff to db ...
                >>> recovered_sig = chvoice.spec_to_sig(db, phase)
        """
        # go from log scale back to linear
        mag = self.db_to_amplitude(db)
        # recover full fourier transform of signal
        real = mag * T.cos(phase)
        imaginary = mag * T.sin(phase)
        complex = T.stack((real, imaginary), dim=-1)
        # inverse fourier transform to get signal
        sig = complex.istft(
            n_fft=self.n_fft,
            hop_length=self.hop_len
        )

        return sig


DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(DEVICE)

#train_dataset = ta.datasets.LIBRISPEECH("./", url="dev-clean-100", download=True)


class ReverbEcho:
    """
    CREDIT :: https://github.com/facebookresearch/denoiser/blob/e27bf5cdcda6e6ffc3a332763411d864210f94c8/denoiser/augment.py

    Hacky Reverb but runs on GPU without slowing down training.
    This reverb adds a succession of attenuated echos of the input
    signal to itself. Intuitively, the delay of the first echo will happen
    after roughly 2x the radius of the room and is controlled by `first_delay`.
    Then RevEcho keeps adding echos with the same delay and further attenuation
    until the amplitude ratio between the last and first echo is 1e-3.
    The attenuation factor and the number of echos to adds is controlled
    by RT60 (measured in seconds). RT60 is the average time to get to -60dB
    (remember volume is measured over the squared amplitude so this matches
    the 1e-3 ratio).
    At each call to RevEcho, `first_delay`, `initial` and `RT60` are
    sampled from their range. Then, to prevent this reverb from being too regular,
    the delay time is resampled uniformly within `first_delay +- 10%`,
    as controlled by the `jitter` parameter. Finally, for a denser reverb,
    multiple trains of echos are added with different jitter noises.
    Args:
        - initial: amplitude of the first echo as a fraction
            of the input signal. For each sample, actually sampled from
            `[0, initial]`. Larger values means louder reverb. Physically,
            this would depend on the absorption of the room walls.
        - rt60: range of values to sample the RT60 in seconds, i.e.
            after RT60 seconds, the echo amplitude is 1e-3 of the first echo.
            The default values follow the recommendations of
            https://arxiv.org/ftp/arxiv/papers/2001/2001.08662.pdf, Section 2.4.
            Physically this would also be related to the absorption of the
            room walls and there is likely a relation between `RT60` and
            `initial`, which we ignore here.
        - first_delay: range of values to sample the first echo delay in seconds.
            The default values are equivalent to sampling a room of 3 to 10 meters.
        - repeat: how many train of echos with differents jitters to add.
            Higher values means a denser reverb.
        - jitter: jitter used to make each repetition of the reverb echo train
            slightly different. For instance a jitter of 0.1 means
            the delay between two echos will be in the range `first_delay +- 10%`,
            with the jittering noise being resampled after each single echo.
        - keep_clean: fraction of the reverb of the clean speech to add back
            to the ground truT. 0 = dereverberation, 1 = no dereverberation.
        - sample_rate: sample rate of the input signals.
    """

    def __init__(self, initial=0.2, rt60=(0.1, 1.0), first_delay=(0.01, 0.03),
                 repeat=31, jitter=0.1, keep_clean=0.1, sample_rate=16000):
        super().__init__()
        self.initial = initial
        self.rt60 = rt60
        self.first_delay = first_delay
        self.repeat = repeat
        self.jitter = jitter
        self.keep_clean = keep_clean
        self.sample_rate = sample_rate

    def reverb(self, sig):
        """
        Return the reverb for a single sig.
        """

        initial = random.random() * self.initial
        first_delay = random.uniform(*self.first_delay)
        rt60 = random.uniform(*self.rt60)

        length = sig.shape[-1]
        reverb = T.zeros_like(sig)
        
        for _ in range(self.repeat):
            frac = 1  # what fraction of the first echo amplitude is still here
            echo = initial * sig
            while frac > 1e-3:
                # First jitter noise for the delay
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                delay = min(
                    1 + int(jitter * first_delay * self.sample_rate),
                    length)
                # Delay the echo in time by padding with zero on the left
                echo = F.pad(echo[:, :, :-delay], (delay, 0))
                reverb += echo

                # Second jitter noise for the attenuation
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                # we want, with `d` the attenuation, d**(rt60 / first_ms) = 1e-3
                # i.e. log10(d) = -3 * first_ms / rt60, so that
                attenuation = 10**(-3 * jitter * first_delay / rt60)
                echo *= attenuation
                frac *= attenuation

        return reverb

class Noiser:

    def __init__(self, noise_dir, noise_lvl=0.4, probs=[], nsr_scalar=[]):
        """ add noise to signal, where noise is randomly 
            selected wav from noise_dir.
            noise_lvl = ratio of noise to signal (additive)
        """
        rs = ta.transforms.Resample()
        self.nsr = noise_lvl
        self.noises = []
        self.probs = []
        self.nsr_scalar = []
        self.reverber = ReverbEcho()
        noise_dirs = os.listdir(noise_dir)
        noise_dirs.sort()
        for index, directory in enumerate(noise_dirs):
          directory = "./noises/" + str(directory)
          print(directory)
          print(probs[index])
          fns = os.listdir(directory)
          for fn in fns:
            noise = ta.load(os.path.join(directory, fn))
            rs = ta.transforms.Resample(noise[1], 16000)(noise[0])
            self.noises.append(rs.squeeze(0))
            self.probs.append(probs[index])
            self.nsr_scalar.append(nsr_scalar[index])
        
    def add_noise(self, sig):
        """ sig is expected to be one-dimensional (N,)
            snr = maximum signal to noise ratio 
        """
        # stochastic amount of noise
        noise_lvl = np.random.random() * self.nsr

        # 10% of the time just add white noise
        r = np.random.random()
        if r > (1-WHITE_NOISE_PROB):
            white_noise = T.normal(
                mean=0., 
                std=0.025, size=sig.shape
            )
            return sig + white_noise*self.nsr

        elif r > (1- WHITE_NOISE_PROB - NO_NOISE_PROB):
          return sig

        elif r > (1 - WHITE_NOISE_PROB - NO_NOISE_PROB - REVERB_PROB):
          sig_reverb = self.reverber.reverb(sig.unsqueeze(0).unsqueeze(0))
          sig_reverb = sig_reverb.squeeze(0).squeeze(0)
          return sig_reverb

        # otherwise add a noise from the archive
        j = random.randint(1, 3)
        for i in range(0, j):
          noise_index = random.choices(range(len(self.noises)), weights=self.probs)[0]
          noise = self.noises[noise_index]
          nsr = self.nsr * self.nsr_scalar[noise_index] *  (1/j) * (.5 + (T.rand(size=(1,))[0] / 2))
          if len(noise) < len(sig):
              n_repeats = int(np.ceil(len(sig) / len(noise)))
              noise = T.repeat_interleave(noise, n_repeats)

          sig = sig + noise[:len(sig)]*nsr

        return sig


# denormalize = torchvision.transforms.Compose([ 
#     torchvision.transforms.Normalize(mean=0., std=1/18),
#     torchvision.transforms.Normalize(mean=32., std=1.)
# ])

# normalize = torchvision.transforms.Normalize(mean=-32., std=18)

def preprocess(X, dsp, noiser):
    clean = []
    noisy = []
    wavs = [d[0] for d in X]

    for wav in wavs:

        db, phase = dsp.sig_to_db_phase(wav)
        if db.size(2) < 128:
            continue

        # make clean chunks of audio
        chunks = db.unfold(2, 128, 128).squeeze(0).movedim(1,0)
        chunks = normalize(chunks)
        clean.append(chunks)

        # make corresponding noisy chunks of audio
        aug_wav = noiser.add_noise(wav.squeeze(0))
        db, phase = dsp.sig_to_db_phase(aug_wav.unsqueeze(0))
        chunks = db.unfold(2, 128, 128).squeeze(0).movedim(1,0)
        chunks = normalize(chunks)
        noisy.append(chunks)

    clean = T.vstack(clean).unsqueeze(1)
    noisy = T.vstack(noisy).unsqueeze(1)
    
    return clean, noisy

### CREDIT : https://github.com/milesial/Pytorch-UNet

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = T.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


model = UNet(1, 1)
model.load_state_dict(T.load("./two_epochs_enhance.torch", map_location=T.device(DEVICE)))

sr = 16000
dsp = DSP(254)
sig = dsp.db_phase_to_sig(T.ones((128,128)), T.ones((128,128)))
frame_len = len(sig)  # 16129

p = pyaudio.PyAudio()

stream_in = p.open(
	format=pyaudio.paInt16,
	channels=1,
	rate=sr,
    frames_per_buffer=frame_len,
	input=True
)

stream_out = p.open(
	format=pyaudio.paInt16, 
	channels=1, 
	rate=sr, 
	output=True
)

mean = 0
std = 0

# buf = np.empty((FRAMES_PER_INFER, CHUNK), dtype=float)

# for i in range(FRAMES_PER_INFER):
#     data = np.fromstring(stream_in.read(CHUNK, exception_on_overflow = False),dtype=np.int16)
#     chunks[i] = data

while(True):
    data = np.fromstring(stream_in.read(frame_len, exception_on_overflow=False), dtype=np.int16)
    # chunks = chunks[1:]
    # chunks = np.vstack([chunks, data])
    # c2 = T.FloatTensor(chunks.flatten()).unsqueeze(0)
    sig = T.from_numpy(data).float().unsqueeze(0).to(DEVICE)

    # print(c2)
    db, phase = dsp.sig_to_db_phase(sig)
    mean = T.mean(db)
    std = T.std(db)
    db = ((db - mean)/std).unsqueeze(1)
                 
    with T.no_grad():
        proc = ((model(db) * std) + mean).squeeze(1)

    # print(proc.shape, phase.shape)
    # db_out = T.cat([c for c in proc.squeeze(1)], dim=1)
    # phase_clipped = phase[0,:,:db_out.size(1)]

    sig_out = dsp.db_phase_to_sig(proc, phase[0,:,:]).numpy()
    # print(sig_out)
    # print(len(sig))
    bytes_out = sig_out.astype(np.int16).tostring()
    # print(data)
    stream_out.write(bytes_out)
    

# close the stream gracefully
stream_in.stop_stream()
stream_out.stop_stream()
stream_in.close()
stream_out.close()
p.terminate()

