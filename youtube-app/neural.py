import operator

import chvoice
import librosa
import soundfile as sf
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
from torch.autograd import Variable

DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')


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


### CREDIT : https://github.com/milesial/Pytorch-UNet
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
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
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
        self.down1 = p_down(channel_depth, (channel_depth * 2))
        self.down2 = p_down((channel_depth * 2), (channel_depth * 4))
        self.down3 = p_down((channel_depth * 4), (channel_depth * 8))
        self.down4 = p_down((channel_depth * 8), (channel_depth * 8))
        self.up1 = p_up(channel_depth * 16, (channel_depth * 4))
        self.up2 = p_up(channel_depth * 8, (channel_depth * 2))
        self.up3 = p_up(channel_depth * 4, channel_depth)
        self.up4 = p_up(channel_depth * 2, channel_depth)
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
        taylor_estimates_by_module = [(module.taylor_estimates, idx) for idx, module in enumerate(module_list) if
                                      getattr(module, "prune_feature_map", False) and module.out_channels > 1]
        taylor_estimates_by_feature_map = [(estimate, f_map_idx, module_idx) for estimates_by_f_map, module_idx in
                                           taylor_estimates_by_module for f_map_idx, estimate in
                                           enumerate(estimates_by_f_map)]

        min_estimate, min_f_map_idx, min_module_idx = min(taylor_estimates_by_feature_map, key=operator.itemgetter(0))

        p_conv = module_list[min_module_idx]
        p_conv.prune_feature_map(min_f_map_idx)

        if verbose:
            print("Pruned conv layer number {}, {}".format(min_module_idx, p_conv))

        # Find next conv layer to drop input channel
        is_last_conv = len(module_list) - 1 == min_module_idx
        if not is_last_conv:
            p_batchnorm = module_list[min_module_idx + 1]
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


class Neural:

    def __init__(self, model_weights_path):
        self.model = PruneUNet(1, 1).to(DEVICE)
        self.model.load_state_dict(T.load(model_weights_path, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.dsp = DSP(254)

    def proc(self, wav_path):

        noisy_sig, _ = librosa.load(wav_path, sr=16000)
        noisy_sig = T.from_numpy(noisy_sig)
        db, phase = self.dsp.sig_to_db_phase(noisy_sig)
        chunks = db.unsqueeze(0).unfold(2, 128, 128).squeeze(0).movedim(1, 0)
        mean = T.mean(chunks)
        std = T.std(chunks)
        chunks = (chunks - mean) / std
        chunks = chunks.unsqueeze(1).to(DEVICE)
        proc = T.empty_like(chunks)

        with T.no_grad():
            for idx in range(0, len(chunks), 64):
                proc[idx:idx + 64] = self.model(chunks[idx:idx + 64])

        proc = (proc * std) + mean
        db_out = T.cat([c for c in proc.squeeze(1)], dim=1)
        phase_clipped = phase[:, :db_out.size(1)]
        sig = self.dsp.db_phase_to_sig(db_out, phase_clipped)
        sf.write(wav_path, sig, 16000)
