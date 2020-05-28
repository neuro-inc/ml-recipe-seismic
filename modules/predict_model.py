# Run this file to perform inference from a trained model.

import warnings
warnings.simplefilter(action='ignore')

from keras import backend as K
from model1 import uResNet34
from data_gen1 import SliceIterator, primary_transform
from const import (
    norm_dict_path,
    log_dir,
    model_dir,
    dumps_dir,
    single_test_slices,
    single_train_slices,
    well_width,
    data_dir
)
import numpy as np
import cv2
from pathlib import Path
import pickle
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# Global parameters: model type, image size, trained weights
model_class = uResNet34
image_size = (1024, 512)
model_weights = model_dir / 'uResNet34.GK.sz1024x512.mxd.05-0.63.hdf5'
im_dir = Path('../images/GK_.mxd')
im_dir.mkdir(exist_ok=True)

c_types = ['GK']
carotage = 'GK'
slice_list = single_test_slices
with open(norm_dict_path, 'rb') as f:
    norm_dict = pickle.load(f)
norm = [(norm_dict[c]['mean'], norm_dict[c]['std']) for c in ['seismic'] + c_types]


if __name__ == '__main__':
    K.clear_session()

    # Read model structure and weights
    model = model_class(input_size=image_size, weights=model_weights, n_carotage=len(c_types))
    gen = SliceIterator(slice_list, c_types, image_size, transform_fun=primary_transform, norm=norm, aug=False,
                        batch_size=10, shuffle=False, seed=None, verbose=False, output_ids=True, infinite_loop=False)
    x_m, y, ids = zip(*gen)
    x, m = zip(*x_m)
    x = np.concatenate(x)
    m = np.concatenate(m)
    y = np.concatenate(y)
    ids = list(chain(*ids))

    # Generate predictions
    pred = model.predict([x, m], batch_size=4)

    # Prepare data for plotting
    c_index = c_types.index(carotage)
    data = []
    for seismic, mask, y_true, p, id in zip(x, m, y, pred, ids):
        designation_size = (697 - 62 + 1, 1024) if 'xline' in id else (810 - 1 + 1, 1024)
        y_pred = p[..., :len(c_types)]

        seism = cv2.resize(seismic[..., 0], dsize=designation_size, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask[..., c_index], dsize=designation_size, interpolation=cv2.INTER_NEAREST)
        y_true = cv2.resize(y_true[..., c_index], dsize=designation_size, interpolation=cv2.INTER_NEAREST)
        y_pred = cv2.resize(y_pred[..., c_index], dsize=designation_size, interpolation=cv2.INTER_NEAREST)

        corr = np.corrcoef(y_true[mask.astype(bool)], y_pred[mask.astype(bool)])[0, 1]
        data.append({'seism': seism, 'mask': mask, 'y_true': y_true, 'y_pred': y_pred, 'corr': corr, 'id': id})

    # Plot images of seismic slices
    for d in data:
        color_norm = colors.Normalize(vmin=d['y_pred'].min(), vmax=d['y_pred'].max())
        extent = [0, d['seism'].shape[1] - 1, d['seism'].shape[0] * 2 - 1, 0]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 14), sharey=False)
        ax1.imshow(d['seism'], cmap='seismic', extent=extent)
        ax1.set_title('Seismic')
        ax1.set_ylabel('ms', fontsize='10')

        ax2.imshow(d['y_true'], cmap='Greys_r', extent=extent, norm=color_norm)
        ax2.set_title('Actual carotage')

        ax3.imshow(d['mask'], extent=extent, cmap='Greys_r')
        ax3.set_title('Mask')

        ax4.imshow(d['y_pred'], cmap='Greys_r', extent=extent, norm=color_norm)
        ax4.set_title(f'Prediced carotage. Correlation={d["corr"]:.2f}')

        plt.tight_layout()
        plt.suptitle(f'{{}} {{}}, {{}} {{}}. {carotage} carotage'.format(*d['id'].split('_')), fontsize=18, y=0.99)
        plt.savefig(im_dir / f'{d["id"]}.{carotage}.png')
        plt.close()
