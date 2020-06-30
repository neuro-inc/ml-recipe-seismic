#
# Data generator for uResNet34 2D-segmentation model
#

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.const import carotage_types, norm_dict_path, slices_dir, nsamples, dt, model_input_size
import numpy as np
import cv2
import pickle
import threading
import matplotlib.pyplot as plt
from itertools import islice, chain

from keras import backend as K
from pathlib import Path


def null_transform(*args, **kwargs):
    return args


def primary_transform(x, m, y, size, norm, aug, blur):
    if norm:
        x = (x - norm[0][0]) / norm[0][1]
        for i, n in enumerate(norm[1:]):
            y[..., i] = (y[..., i] - n[0]) / n[1]
    x = cv2.resize(x, dsize=size[::-1], interpolation=cv2.INTER_CUBIC)[..., np.newaxis]
    m = cv2.resize(m, dsize=size[::-1], interpolation=cv2.INTER_NEAREST)
    y = cv2.resize(y, dsize=size[::-1], interpolation=cv2.INTER_CUBIC)
    if blur:
        m = cv2.GaussianBlur(m.astype(np.uint16) * 255, (15, 15), 0)
        m = m / 255.0
    if len(m.shape) == 2:
        m = m[..., np.newaxis]
    if len(y.shape) == 2:
        y = y[..., np.newaxis]
    if aug and (np.random.rand() > 0.5):
        x = x[:, ::-1]
        m = m[:, ::-1]
        y = y[:, ::-1]
    return x, m, y


class SliceIterator(object):
    """This class iterates over seismic slices for training or inference"""

    def __init__(self, slice_list, carotage_types, image_size, transform_fun=null_transform, norm=None,
                 aug=False, batch_size=8, shuffle=True, seed=None, verbose=False, infinite_loop=True,
                 output_ids=False, blur=False, gen_id=''):
        self.sllice_list = list(slice_list)
        self.carotage_types = carotage_types
        self.image_size = tuple(image_size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform_fun = transform_fun
        self.norm = norm
        self.blur = blur
        self.aug = aug
        self.output_names = output_ids
        self.gen_id = gen_id
        self.seed = seed
        self.verbose = verbose
        self.infinite_loop = infinite_loop

        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(len(self.sllice_list))

    def next(self):
        with self.lock:
            slice_ids, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size,) + self.image_size + (1,), dtype=K.floatx())
        batch_m = np.zeros((current_batch_size,) + self.image_size + (len(self.carotage_types),), dtype=K.floatx())
        batch_y = np.zeros_like(batch_m, dtype=K.floatx())
        names = []
        for b, slice_id in enumerate(slice_ids):
            slice_fname = self.sllice_list[slice_id]
            with open(slice_fname, 'rb') as f:
                slice_data = pickle.load(f)
            x = slice_data['seismic']
            m = np.stack([slice_data['projections'][c]['mask'] for c in self.carotage_types], axis=-1).astype(K.floatx())
            y = np.stack([slice_data['projections'][c]['target'] for c in self.carotage_types], axis=-1)
            x, m, y = self.transform_fun(x, m, y, size=self.image_size, norm=self.norm, aug=self.aug, blur=self.blur)

            if K.image_data_format() == 'channels_first':
                x = np.moveaxis(x, -1, 0)
                m = np.moveaxis(m, -1, 0)
                y = np.moveaxis(y, -1, 0)
            batch_x[b] = x
            batch_m[b] = m
            batch_y[b] = y
            names.append(slice_fname.stem)

        out = [batch_x, batch_m], batch_y
        if self.output_names:
            out += (names,)
        return out

    def _flow_index(self, n):
        # Ensure self.batch_index is 0.
        self.batch_index = 0
        ids = list(range(n))
        while 1:
            if self.seed is None:
                random_seed = None
            else:
                random_seed = self.seed + self.total_batches_seen
            if self.batch_index == 0:
                if not self.infinite_loop and (self.total_batches_seen > 0):
                    break
                if self.verbose:
                    print("\n************** New epoch. Generator", self.gen_id, "*******************\n")
                if self.shuffle:
                    np.random.RandomState(random_seed).shuffle(ids)

            current_index = (self.batch_index * self.batch_size) % n
            if n > current_index + self.batch_size:
                current_batch_size = self.batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (ids[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


def dump_normalization_values(slices_dir, path=norm_dict_path, overwrite=False):
    """Compute and save to disk normalization values"""
    if path.exists() and not overwrite:
        return
    # image_size = (1024, 810)
    image_size = model_input_size
    slice_list = list(Path(slices_dir).glob('*.pkl'))
    gen = SliceIterator(slice_list, carotage_types, image_size, transform_fun=primary_transform,
                        norm=None, batch_size=16, shuffle=False, infinite_loop=False)
    norm_dict = {c: {'mean': [], 'std': []} for c in ['seismic'] + carotage_types}
    for b, ((x, m), y) in enumerate(gen):
        print('batch', b + 1)
        norm_dict['seismic']['mean'].append(np.median(x))
        norm_dict['seismic']['std'].append(np.std(x))
        for i, c in enumerate(carotage_types):
            norm_dict[c]['mean'].append(np.median(y[..., i][m[..., i].astype(bool)]))
            norm_dict[c]['std'].append(np.std(y[..., i][m[..., i].astype(bool)]))
    for k in norm_dict:
        norm_dict[k]['mean'] = np.median(norm_dict[k]['mean'])
        norm_dict[k]['std'] = np.median(norm_dict[k]['std'])
    with open(path, 'wb') as f:
        pickle.dump(norm_dict, f)


if __name__ == '__main__':

    c_types = ['Gamma_Ray']
    with open(norm_dict_path, 'rb') as f:
        norm_dict = pickle.load(f)
    norm = [(norm_dict[c]['mean'], norm_dict[c]['std']) for c in ['seismic'] + c_types]
    slice_list = slices_dir.glob('*pkl')
    gen = SliceIterator(slice_list, c_types, model_input_size, transform_fun=primary_transform, norm=norm, aug=True,
                        blur=True, batch_size=8, shuffle=True, seed=None, verbose=False, output_ids=True, gen_id='')

    x_m, y, ids = zip(*islice(gen, 2))
    x, m = zip(*x_m)
    x = np.concatenate(x)
    m = np.concatenate(m)
    y = np.concatenate(y)
    ids = list(chain(*ids))

    c = 0
    c_type = c_types[c]
    for mask, target, id in zip(m, y, ids):
        extent = [0, mask.shape[1] - 1, nsamples * dt, 0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 10), sharey=True)
        ax1.imshow(mask[..., c], cmap='Greys_r', extent=extent)
        ax1.set_title('Mask')
        ax1.set_ylabel('ms', fontsize=10)
        ax2.imshow(target[..., c], cmap='Spectral_r', extent=extent)
        ax2.set_title(f'{id}, {c_type}')
        plt.tight_layout()
        plt.show()
