import numpy as np
import librosa


def mix(sig1, sig2, sig2_power=0.2):
    """ mix two wav signals together.
        example usage:
            >>> sig1, sr = librosa.load('sound.wav')
            >>> sig2, sr = librosa.load('noise.wav')
            >>> sig = chvoice.mix(sig1, sig2)
    """
    if len(sig2) < len(sig1):
        n_repeats = np.ceil(len(sig1) / len(sig2))
        sig2 = np.repeat(sig2, n_repeats)

    sig = sig1 + sig2[:len(sig1)] * sig2_power

    return sig


def sig_to_spec(sig, n_fft=512, hop_length=128):
    """ get dB and phase spectrograms of signal
        example usage:
            >>> sig, sr = librosa.load('sound.wav')
            >>> db, phase = chvoice.sig_to_spec(sig)
    """

    # represent input signal in time-frequency domain
    stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)

    # magnitude = amount of power/volume for each phase = frequency
    stft_magnitude, stft_phase = librosa.magphase(stft)

    # put magnitudes on log scale
    stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, np.max)

    return stft_magnitude_db, stft_phase


def spec_to_sig(spec_db, spec_phase, hop_length=128):
    """ get wav signal from db and phase spectrograms.
        example usage:
            >>> sig, sr = librosa.load('sound.wav')
            >>> db, phase = chvoice.sig_to_spec(sig)
            >>> recovered_sig = chvoice.spec_to_sig(db, phase)
    """

    # go from log scale back to linear
    stft_magnitude = librosa.db_to_amplitude(spec_db)

    # recover full fourier transform of signal
    stft = stft_magnitude * spec_phase

    # inverse fourier transform to get signal
    sig = librosa.istft(stft, hop_length=hop_length)

    return sig
