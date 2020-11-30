from os import listdir
from os.path import join
import numpy as np
import librosa
from chvoice.audio_processing import sig_to_chunks, sig_to_spec


class StaticDataGenerator:

    def __init__(self, clean_dir, noise_dir, sample_rate=22050):
        """ generate batches of spectrograms from pairs of .wav
            files in specified directories
        :param clean_dir: path to clean .wav files
        :param noise_dir: path to noisy .wav files
        """
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        clean_fns = listdir(clean_dir)
        noise_fns = listdir(noise_dir)
        self.fns = list(set(clean_fns).intersection(noise_fns))
        self.num_samples = len(self.fns)
        self.ix = 0  # count to last sample seen
        self.sr = sample_rate

    def batch(self, batch_size=16, sec_per_sample=2):
        """ returns batch (noisy, clean) spectrograms, where each sample
            is sec_per_sample seconds long.
        """
        samples_clean = []
        samples_noise = []

        while len(samples_clean) < batch_size:
            # wrap back to start if we've seen all samples
            if self.ix >= self.num_samples: self.ix = 0

            path_clean = join(self.clean_dir, self.fns[self.ix])
            path_noise = join(self.noise_dir, self.fns[self.ix])
            sig_clean, _ = librosa.load(path_clean, sr=self.sr)
            sig_noise, _ = librosa.load(path_noise, sr=self.sr)

            try:
                chunks_clean = sig_to_chunks(sig_clean, sec_per_sample, self.sr)
                chunks_noise = sig_to_chunks(sig_noise, sec_per_sample, self.sr)
                samples_clean.extend(chunks_clean)
                samples_noise.extend(chunks_noise)
                self.ix += 1
            except AssertionError:
                del self.fns[self.ix]
                self.num_samples -= 1

        # truncate in case we added too many
        samples_clean = samples_clean[:batch_size]
        samples_noise = samples_noise[:batch_size]

        # convert to spectrograms
        samples_clean = [sig_to_spec(x)[0] for x in samples_clean]
        samples_noise = [sig_to_spec(x)[0] for x in samples_noise]

        X = np.stack(samples_noise)
        Y = np.stack(samples_clean)

        return X, Y
