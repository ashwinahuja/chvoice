### [L46 Project] Real-Time Voice Denoising


#### YouTube Speech Enhancement Flask Application

#### Experimental Notebooks
Contains experimental notebooks for each of the experiments. These contain definition, training and some evaluation, though evaluation in general involved importing the model created into the Evaluation Notebook.
- Normal Model (Equal Probs, No Reverb, Normal Weights, Normal Weights with Knowledge Distillation)
- Smaller Model (Equal Probs, No Reverb, Normal Weights)

Also contains:
- Minimal Notebook which has examples of audio samples before and after voice denoising
- Evaluation Notebook which was used to evaluate models, getting latency and RMS error
- Models folder which contains the loss histories, state dict and complete model (stored with pickle)

#### Real-Time Python Denoising Application
To run the real-time denoising application, run:
```bash
python3 app.py
```

You require the following:
- portaudio (brew install portaudio)
- pyaudio (pip3 install pyaudio)
- PyTorch (pip3 install torch)
- torchaudio (pip3 install torchaudio)

#### Flask App

Likewise, a web interface to our model can be served by running the following from `youtube-app/`:

```bash
python3 app.py
```

Requirements are the same as above


#### Python DSP Library
To install the DSP library, run the following:
```bash
pip install git+https://github.com/indrasweb/chvoice
```

#### Repository Structure
```

.
├── README.md
├── chvoice   # utils for training (DSP and pre-processing)
├── mac-app   # non-working AudioUnitV3 attempt
├── noises    # noise wavs for training
├── noises.zip  # noise wavs for training
├── notebooks   # self-contained ipynb for training models
├── real-time-python-app   # take input from mic and pipe through model
├── setup.py
└── youtube-app  # download youtube video with audio processed by our model

```
