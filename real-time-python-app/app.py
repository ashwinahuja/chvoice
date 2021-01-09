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

class PrunableConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.taylor_estimates = None
        self._recent_activations = None
        self._pruning_hook = None

    def forward(self, x):
        output = super().forward(x)
        self._recent_activations = output.clone()
        return output

    def set_pruning_hooks(self):
        self._pruning_hook = self.register_backward_hook(self._calculate_taylor_estimate)

    def _calculate_taylor_estimate(self, _, grad_input, grad_output):
        # skip dim 1 as it is kernel size
        estimates = self._recent_activations.mul_(grad_output[0])
        estimates = estimates.mean(dim=(0, 2, 3))        

        # normalization
        self.taylor_estimates = T.abs(estimates) / T.sqrt(T.sum(estimates * estimates))
        del estimates, self._recent_activations
        self._recent_activations = None

    def prune_feature_map(self, map_index):
        is_cuda = self.weight.is_cuda

        indices = Variable(T.LongTensor([i for i in range(self.out_channels) if i != map_index]))
        indices = indices.cuda() if is_cuda else indices

        self.weight = nn.Parameter(self.weight.index_select(0, indices).data)
        self.bias = nn.Parameter(self.bias.index_select(0, indices).data)
        self.out_channels -= 1

    def drop_input_channel(self, index):
        is_cuda = self.weight.is_cuda

        indices = Variable(torch.LongTensor([i for i in range(self.in_channels) if i != index]))
        indices = indices.cuda() if is_cuda else indices

        self.weight = nn.Parameter(self.weight.index_select(1, indices).data)
        self.in_channels -= 1

class PrunableBatchNorm2d(nn.BatchNorm2d):
  def __init__(self, channels):
      super(PrunableBatchNorm2d, self).__init__(channels)

  def drop_input_channel(self, index):
        if self.affine:
            is_cuda = self.weight.is_cuda
            indices = Variable(torch.LongTensor([i for i in range(self.num_features) if i != index]))
            indices = indices.cuda() if is_cuda else indices

            self.weight = nn.Parameter(self.weight.index_select(0, indices).data)
            self.bias = nn.Parameter(self.bias.index_select(0, indices).data)
            self.running_mean = self.running_mean.index_select(0, indices.data)
            self.running_var = self.running_var.index_select(0, indices.data)

        self.num_features -= 1

  def forward(self, x):
      x = super().forward(x)
      return x
      
class p_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_channels=None):
        super(p_double_conv, self).__init__()
        
        if not mid_channels:
            mid_channels = out_ch

        self.quant1 = T.quantization.QuantStub()
        self.dequant1 = T.quantization.DeQuantStub()

        self.conv = nn.Sequential(
            PrunableConv2d(in_ch, out_ch, 3, padding=1),
            PrunableBatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            PrunableConv2d(mid_channels, out_ch, 3, padding=1),
            PrunableBatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        


    def forward(self, x):
        x = self.quant1(x)
        x = self.conv(x)
        x = self.dequant1(x)
        return x


class p_inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(p_inconv, self).__init__()
        self.conv = p_double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class p_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(p_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            p_double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class p_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(p_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = p_double_conv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch , in_ch // 2, kernel_size=2, stride=2)
            self.conv = p_double_conv(in_ch, out_ch)
        

      

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = T.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class p_outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(p_outconv, self).__init__()
        self.conv = PrunableConv2d(in_ch, out_ch, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class PruneUNet(nn.Module):
    def __init__(self, n_channels, n_classes, channel_depth=64):
        super(PruneUNet, self).__init__()
        self.inc = p_inconv(n_channels, channel_depth)
        self.down1 = p_down(channel_depth, (channel_depth*2))
        self.down2 = p_down((channel_depth*2), (channel_depth*4))
        self.down3 = p_down((channel_depth*4), (channel_depth*8))
        self.down4 = p_down((channel_depth*8), (channel_depth*8))
        self.up1 = p_up(channel_depth*16, (channel_depth*4))
        self.up2 = p_up(channel_depth*8, (channel_depth*2))
        self.up3 = p_up(channel_depth*4, channel_depth)
        self.up4 = p_up(channel_depth*2, channel_depth)
        self.outc = p_outconv(channel_depth, n_classes)


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
        x = self.outc(x)
        return x

    def set_pruning(self):
        prunable_modules = [module for module in self.modules()
                             if getattr(module, "prune_feature_map", False)
                             and module.out_channels > 1]

        # # Print getting all conv layers
        # for i, m in enumerate(self.modules()):
        #     if getattr(m, "prune_feature_map", False) and m.out_channels > 1:
        #         print(i, "->", m)

        for p in prunable_modules:
            p.set_pruning_hooks()
    
    def prune(self, verbose=False):
        # Get all layers excluding larger blocks
        module_list = [module for module in self.modules() 
                        if not isinstance(module, 
                            (nn.Sequential, 
                            p_double_conv, 
                            p_inconv, 
                            p_down, 
                            p_up, 
                            p_outconv,
                            PruneUNet)
                            )
                        ]
        
        # Also checks if layer has been pruned to 1 channel remaining
        taylor_estimates_by_module = [(module.taylor_estimates, idx) for idx, module in enumerate(module_list) if getattr(module, "prune_feature_map", False) and module.out_channels > 1]
        taylor_estimates_by_feature_map = [(estimate, f_map_idx, module_idx) for estimates_by_f_map, module_idx in taylor_estimates_by_module for f_map_idx, estimate in enumerate(estimates_by_f_map)]

        min_estimate, min_f_map_idx, min_module_idx = min(taylor_estimates_by_feature_map, key=operator.itemgetter(0))

        p_conv = module_list[min_module_idx]
        p_conv.prune_feature_map(min_f_map_idx)
        
        if verbose:
            print("Pruned conv layer number {}, {}".format(min_module_idx, p_conv))

        # Find next conv layer to drop input channel
        is_last_conv = len(module_list)-1 == min_module_idx
        if not is_last_conv:
            p_batchnorm = module_list[min_module_idx+1]
            p_batchnorm.drop_input_channel(min_f_map_idx)
            
            next_conv_idx = min_module_idx + 2
            while next_conv_idx < len(module_list):
                if isinstance(module_list[next_conv_idx], PrunableConv2d):
                    module_list[next_conv_idx].drop_input_channel(min_f_map_idx)
                    if verbose:
                        print("Found next conv layer at number {}, {}"
                            .format(next_conv_idx, module_list[next_conv_idx]))
                    break
                next_conv_idx += 1
            
            # Hardcoded way of dealing with up sampled layers

            # x4 as output of down3 -> drop input channel of up1
            if min_module_idx == 24:
                p_up_conv = module_list[35]
                p_up_conv.drop_input_channel(min_f_map_idx)
            # x3 as output of down2 -> drop input channel of up2
            elif min_module_idx == 17:
                p_up_conv = module_list[42]
                p_up_conv.drop_input_channel(min_f_map_idx)
            # x2 as output of down1 -> drop input channel of up3
            elif min_module_idx == 10:
                p_up_conv = module_list[49]
                p_up_conv.drop_input_channel(min_f_map_idx)
            # x1 as output of inc -> drop input channel of up4
            elif min_module_idx == 3:
                p_up_conv = module_list[56]
                p_up_conv.drop_input_channel(min_f_map_idx)



model = PruneUNet(n_channels=1, n_classes=1, channel_depth=32).to(DEVICE)
model.load_state_dict(T.load('../notebooks/Models/model-distilled-noreverb-normal-model.pkl', map_location=T.device('cpu')))

sr = 16000
dsp = DSP(254)
sig = dsp.db_phase_to_sig(T.ones((128,128)), T.ones((128,128)))
frame_len = math.ceil(len(sig)/4)  # 16129

p = pyaudio.PyAudio()



stream_out = p.open(
	format=pyaudio.paInt16, 
	channels=1, 
	rate=sr, 
	output=True
)


stream_in = p.open(
	format=pyaudio.paInt16,
	channels=1,
	rate=sr,
    frames_per_buffer=frame_len,
	input=True
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
        p = model(db)
        p *= std
        p += mean
        proc = p.squeeze(1)
        #proc = ((model(db) * std) + mean).squeeze(1)

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

