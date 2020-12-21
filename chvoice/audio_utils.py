import torch as T
import torch.nn.functional as F
import torchaudio as ta
import numpy as np
import random
from glob import glob


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

    def __init__(self, initial=0.1, rt60=(0.1, 1.0), first_delay=(0.01, 0.03),
                 repeat=25, jitter=0.1, keep_clean=0.1, sample_rate=16000):
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

        for _ in range(int(self.repeat * np.random.random())):
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
                attenuation = 10 ** (-3 * jitter * first_delay / rt60)
                echo *= attenuation
                frac *= attenuation

        return reverb


class Noiser:

    def __init__(self, wav_noise_config=None, additional_noise_config=None):
        """ add noise to signal, where noise is randomly
            selected wav from noise_dir.
            noise_lvl = ratio of noise to signal (additive)
        """
        if not wav_noise_config:
            self.wav_noise_config = {
                './noises/keyboard-mono': {'prob': 0.4, 'mul': 1.},
                './noises/coughing-mono': {'prob': 0.1, 'mul': .2},
                './noises/clock-tick-mono': {'prob': 0.1, 'mul': .6},
                './noises/click-mono': {'prob': 0.2, 'mul': .8},
                './noises/wind-mono': {'prob': 0.2, 'mul': .4}
            }
        else:
            self.wav_noise_config = wav_noise_config
        if not additional_noise_config:
            self.additional_noise_config = {
                'white_noise': {'prob': 0.25, 'mul': 1.},
                'no_noise': {'prob': 0.2},
                'reverb': {'prob': 0.2, 'density': 25}
            }
        else:
            self.additional_noise_config = additional_noise_config

        self.dirs = list(self.wav_noise_config.keys())
        self.probs = [self.wav_noise_config[k]['prob'] for k in self.dirs]

        self.reverb = ReverbEcho(repeat=self.additional_noise_config['reverb']['density'])

        for _dir in self.wav_noise_config.keys():
            self.wav_noise_config[_dir]['noises'] = []
            fns = glob(f'{_dir}/*.wav', recursive=True)
            for fn in fns:
                noise, sr = ta.load(fn)
                noise = ta.transforms.Resample(sr, 16000)(noise)
                self.wav_noise_config[_dir]['noises'].append(noise.squeeze(0))


    def add_noise(self, sig):
        """ sig is expected to be one-dimensional (N,)
            snr = maximum signal to noise ratio
        """
        r = np.random.random()

        if r > (1 - self.additional_noise_config['white_noise']['prob']):
            white_noise = T.normal(
                mean=0.,
                std=0.02, size=sig.shape
            )
            sig = sig + white_noise * self.additional_noise_config['white_noise']['mul']

        elif r > (1 - self.additional_noise_config['white_noise']['prob'] -
                  self.additional_noise_config['no_noise']['prob']):
            return sig

        elif r > (1 - self.additional_noise_config['white_noise']['prob'] -
                  self.additional_noise_config['no_noise']['prob'] -
                  self.additional_noise_config['reverb']['prob']):
            sig = self.reverb.reverb(sig.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        #  add a random number of noises from the archive
        for i in range(0, random.randint(0, 2)):

            noise_type = random.choices(self.dirs, weights=self.probs, k=1)[0]

            noise = random.choice(self.wav_noise_config[noise_type]['noises'])

            multiplier = (.6 + (T.rand(size=(1,))[0] / 2)) * self.wav_noise_config[noise_type]['mul']

            if len(noise) < len(sig):
                n_repeats = int(np.ceil(len(sig) / len(noise)))
                noise = T.repeat_interleave(noise, n_repeats)

            sig = sig + noise[:len(sig)] * multiplier

        return sig
