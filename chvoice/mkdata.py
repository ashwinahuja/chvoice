import h5py
import numpy as np
from tqdm import tqdm
from chvoice.data_generator import StaticDataGenerator


clean_dir = '/Users/harrysonghurst/Downloads/clean_trainset_56spk_wav'
noise_dir = '/Users/harrysonghurst/Downloads/noisy_trainset_56spk_wav'

N = 32768
s = 0
batch_size = 512

f = h5py.File('ds.hdf5', mode='w')
ds_noisy = f.create_dataset('noisy', (N, 256, 256), dtype=np.float32, compression='gzip', compression_opts=9)
ds_clean = f.create_dataset('clean', (N, 256, 256), dtype=np.float32, compression='gzip', compression_opts=9)

data = StaticDataGenerator(clean_dir, noise_dir, batch_size=batch_size)

pbar = tqdm(range(0, N, batch_size))
for s in pbar:

    noisy, clean = data.batch()
    ds_noisy[s:s+batch_size] = noisy
    ds_clean[s:s+batch_size] = clean

    pbar.set_description(
        f'seen {data.ix}/{data.num_samples} files | '
        f'{s+batch_size}/{N} samples created | '
        f'({(s+batch_size)/data.ix:.2f} samples per file avg)'
    )

f.close()