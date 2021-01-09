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

#### Python DSP Library
To install the DSP library, run the following:
```bash
pip install git+https://github.com/indrasweb/chvoice
```

#### Repository Structure
```
.
├── chvoice                   python lib functions
│   ├── audio_processing.py   turn audio into spectrograms/images and visa-versa
│   └── plotting.py           plot spectrograms of audio
└── web                       javascript stuff for chrome extension (attempted)
    └── README.md
```
