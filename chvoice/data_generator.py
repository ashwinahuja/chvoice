from os import listdir
from os.path import join
import numpy as np
import librosa
from chvoice.audio_processing import sig_to_chunks, sig_to_spec


class StaticDataGenerator:

    def __init__(self, clean_dir, noise_dir, batch_size=16, sample_rate=22050, n_fft=510, sec_per_sample=1.485):
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
        self.n_fft = n_fft
        self.batch_size = batch_size
        self.secs = sec_per_sample

    def __len__(self):
        return self.num_samples

    def __getitem__(self):
        """ returns batch (noisy, clean) spectrograms, where each sample
            is sec_per_sample seconds long.
        """
        samples_clean = []
        samples_noise = []

        while len(samples_clean) < self.batch_size:
            # wrap back to start if we've seen all samples
            if self.ix >= self.num_samples: self.ix = 0

            path_clean = join(self.clean_dir, self.fns[self.ix])
            path_noise = join(self.noise_dir, self.fns[self.ix])
            sig_clean, _ = librosa.load(path_clean, sr=self.sr)
            sig_noise, _ = librosa.load(path_noise, sr=self.sr)

            try:
                chunks_clean = sig_to_chunks(sig_clean, self.secs, self.sr)
                chunks_noise = sig_to_chunks(sig_noise, self.secs, self.sr)
                samples_clean.extend(chunks_clean)
                samples_noise.extend(chunks_noise)
                self.ix += 1
            except AssertionError:
                del self.fns[self.ix]
                self.num_samples -= 1

        # truncate in case we added too many
        samples_clean = samples_clean[:self.batch_size]
        samples_noise = samples_noise[:self.batch_size]

        # convert to spectrograms
        samples_clean = [sig_to_spec(x, n_fft=self.n_fft)[0] for x in samples_clean]
        samples_noise = [sig_to_spec(x, n_fft=self.n_fft)[0] for x in samples_noise]

        X = np.stack(samples_noise)
        Y = np.stack(samples_clean)

        return X, Y
