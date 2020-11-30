import numpy as np
import matplotlib.pyplot as plt
import librosa.display as lplt


def plot_spec_db(spec_db, sample_rate=22050, hop_length=128):
    """ plot a magnitude (dB) spectrogram
        example usage:
            >>> sig, sr = librosa.load('sound.wav')
            >>> db, phase = chvoice.sig_to_spec(sig)
            >>> chvoice.plot_spec_phase(db)
    """

    plt.figure(figsize=(10, 5))

    lplt.specshow(
        spec_db,
        x_axis='time',
        y_axis='linear',
        sr=sample_rate,
        hop_length=hop_length
    )

    plt.colorbar()

    title = f'hop_length={hop_length}'
    title += f',  fft_bins={spec_db.shape[0]}'
    title += f',  timesteps={spec_db.shape[1]}'
    title += f' (2D shape: {spec_db.shape})'
    plt.title(title)

    plt.show()
    return


def plot_spec_phase(spec_phase, sample_rate=22050, hop_length=128):
    """ plot a phase spectrogram
        example usage:
            >>> sig, sr = librosa.load('sound.wav')
            >>> db, phase = chvoice.sig_to_spec(sig)
            >>> chvoice.plot_spec_phase(phase)
    """

    plt.figure(figsize=(10, 5))

    lplt.specshow(
        np.angle(spec_phase),
        x_axis='time',
        y_axis='linear',
        sr=sample_rate,
        hop_length=hop_length
    )

    plt.colorbar()

    title = f'hop_length={hop_length}'
    title += f',  fft_bins={spec_phase.shape[0]}'
    title += f',  timesteps={spec_phase.shape[1]}'
    title += f' (2D shape: {spec_phase.shape})'
    plt.title(title)

    plt.show()
    return
