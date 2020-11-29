import numpy as np
import librosa


def sig_to_spec(sig):
    """ get dB and phase spectrograms of signal
        expect sig sample rate to be 22050.
        example usage:
            >>> sig, sr = librosa.load('sound.wav')
            >>> db, phase = chvoice.sig_to_spec(sig)

        TODO: add sample_rate param and calculate
              n_fft, hop_length dynamically
    """

    # represent input signal in time-frequency domain
    stft = librosa.stft(sig, n_fft=512, hop_length=128)

    # magnitude = amount of power/volume for each phase = frequency
    stft_magnitude, stft_phase = librosa.magphase(stft)

    # 'normalize' each frequency bin relative to the loudest one
    stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, np.max)

    return stft_magnitude_db, stft_phase


def spec_to_sig(spec_db, spec_phase):
    """ get wav signal from amplitude (spec_db) and
        phase spectrograms.
        example usage:
            >>> sig, sr = librosa.load('sound.wav')
            >>> db, phase = chvoice.sig_to_spec(sig)
            >>> recovered_sig = chvoice.spec_to_sig(db, phase)

        TODO: dynamic hop_length based on sample_rate
    """

    # 'denormalize' magnitude spectrogram
    stft_magnitude = librosa.db_to_amplitude(spec_db, ref=1.0)

    # recover full fourier transform of signal
    stft = stft_magnitude * spec_phase

    # inverse fourier transform to get signal
    sig = librosa.istft(stft, hop_length=128)

    return sig
