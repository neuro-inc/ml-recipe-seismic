#
# Evaluate trained models
#
import warnings
warnings.simplefilter(action='ignore')

from keras import backend as K
from src.model1 import uResNet34
from src.train import cv_dataset, norm_dict
from src.gen1 import SliceIterator, primary_transform
from src.const import (
    model_dir, model_input_size,
    wells, ilines, xlines, nsamples, dt
)
import numpy as np
import cv2
from itertools import chain
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path


model_class = uResNet34

# model weights
weights = {
    'Gamma_Ray': [
        'uResNet34.Gamma_Ray.sz480x512.smtd_0.14-0.78.hdf5',
        'uResNet34.Gamma_Ray.sz480x512.smtd_1.11-0.37.hdf5',
        'uResNet34.Gamma_Ray.sz480x512.smtd_2.07-0.65.hdf5',
        'uResNet34.Gamma_Ray.sz480x512.smtd_3.42-0.67.hdf5'
    ],
}


def predict_on_fold(slice_list: List[Path], carotage: str, model_weights: Path, verbose: bool = False) -> dict:
    """predict model for a single fold
    return: dict[slice/well]{'seism', 'mask', 'y_true', 'y_pred', 'corr'}"""

    norm = [(norm_dict[c]['mean'], norm_dict[c]['std']) for c in ['seismic', carotage]]

    K.clear_session()
    model = model_class(input_size=model_input_size, weights=model_weights, n_carotage=1)
    gen = SliceIterator(slice_list, [carotage], model_input_size, transform_fun=primary_transform, norm=norm, aug=False,
                        batch_size=10, shuffle=False, seed=None, verbose=False, output_ids=True, infinite_loop=False)
    x_m, y, ids = zip(*gen)
    x, m = zip(*x_m)
    x = np.concatenate(x)
    m = np.concatenate(m)
    y = np.concatenate(y)
    ids = list(chain(*ids))

    pred = model.predict([x, m], batch_size=1)

    data = {}
    for seismic, mask, y_true, p, i_d in zip(x, m, y, pred, ids):
        designation_size = (max(ilines) - min(ilines) + 1, nsamples) if 'xline' in i_d \
            else (max(xlines) - min(xlines) + 1, nsamples)
        y_pred = p[..., :1]

        seism = cv2.resize(seismic[..., 0], dsize=designation_size, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask[..., 0], dsize=designation_size, interpolation=cv2.INTER_NEAREST)
        y_true = cv2.resize(y_true[..., 0], dsize=designation_size, interpolation=cv2.INTER_NEAREST)
        y_pred = cv2.resize(y_pred[..., 0], dsize=designation_size, interpolation=cv2.INTER_NEAREST)

        corr = np.corrcoef(y_true[mask.astype(bool)], y_pred[mask.astype(bool)])[0, 1]
        data[i_d] = {'seism': seism, 'mask': mask, 'y_true': y_true, 'y_pred': y_pred, 'corr': corr}
    if verbose:
        # provisional correlation based on single pixels. not used in final evaluation
        print(f'corr={np.mean([d["corr"] for d in data if ~np.isnan(d["corr"])])}')
    return data


def eval_fold(data: dict, carotage: str) -> dict:
    """return: dict[slice/well]: {'t', 'true_carotage', 'pred_carotage', 'corr'}"""

    mean, std = norm_dict[carotage]['mean'], norm_dict[carotage]['std']
    viz_data = {}
    for i_d, d in data.items():
        mask = d['mask'].astype(int)
        y_true = d['y_true']
        y_pred = d['y_pred']

        true_carotage = (y_true * mask).sum(1) / mask.sum(1) * std + mean
        pred_carotage = (y_pred * mask).sum(1) / mask.sum(1) * std + mean
        t = np.arange(nsamples, dtype=float) * dt
        t[mask.sum(1) == 0] = np.nan
        corr = np.corrcoef(true_carotage[~np.isnan(true_carotage)], pred_carotage[~np.isnan(pred_carotage)])[0, 1]
        viz_data[i_d] = {'t': t, 'true_carotage': true_carotage, 'pred_carotage': pred_carotage, 'corr': corr}
    return viz_data


def process_all_folds(weights: dict, carotage: str) -> dict:
    """process all folds for a given carotage type
    return: dict[slice/well]: {'t', 'true_carotage', 'pred_carotage', 'corr'}"""

    folds = range(len(cv_dataset))
    all_viz = {}
    for fold in folds:
        model_weights = model_dir / weights[carotage][fold]
        slice_list = cv_dataset[fold]['test']
        data = predict_on_fold(slice_list, carotage, model_weights, verbose=False)
        viz_data = eval_fold(data, carotage)
        print('fold:', fold, 'corr:', np.mean([v['corr'] for v in viz_data.values()]))
        all_viz.update(viz_data)
    return all_viz


def average_prediction(eval_dict: dict) -> dict:
    """average predictions for every well across slices of the same type (i.e. ilines/xlines)
    return: dict[well][slice_type]: {'t', 'true_average', 'pred_average', 'corr'}
    """

    mean_eval_dict = {}
    for w in wells:
        mean_eval_dict[w] = {}
        t = next(v['t'] for k, v in eval_dict.items() if w in k)
        true_carotage = next(v['true_carotage'] for k, v in eval_dict.items() if w in k)
        pred_carotage = [v['pred_carotage'] for k, v in eval_dict.items() if w in k]
        pred_carotage = np.stack(pred_carotage).mean(0)
        corr = np.corrcoef(true_carotage[~np.isnan(true_carotage)], pred_carotage[~np.isnan(pred_carotage)])[0, 1]
        print(f'well {w}, corr={corr:0.2f}')
        mean_eval_dict[w]['t'] = t
        mean_eval_dict[w]['true_carotage'] = true_carotage
        mean_eval_dict[w]['pred_carotage'] = pred_carotage
        mean_eval_dict[w]['corr'] = corr
    return mean_eval_dict


if __name__ == '__main__':

    carotage = 'Gamma_Ray'
    eval_dict = process_all_folds(weights, carotage)
    mean_eval_dict = average_prediction(eval_dict)

    fontsize = 12
    slice_type = 'xline'
    fig, ax = plt.subplots(1, 4, figsize=(14, 14), sharey=True)

    for i, (w, ax_) in enumerate(zip(wells, ax.flatten())):
        t = mean_eval_dict[w][slice_type]['t']
        true_carotage = mean_eval_dict[w][slice_type]['true_carotage']
        pred_carotage = mean_eval_dict[w][slice_type]['pred_carotage']
        corr = mean_eval_dict[w][slice_type]['corr']
        ax_.plot(true_carotage, t, pred_carotage, t)
        ax_.set_title(r'{} {}'.format(w, slice_type), fontsize=fontsize)
        if i == 0:
            ax_.set_ylabel('ms', fontsize=fontsize)
            ax_.legend(['Actual carotage', 'Predicted'], fontsize=fontsize)
        ax_.text(0.65, 0.02, f'corr={corr:0.2f}', transform=ax_.transAxes, fontsize=fontsize)
        ax_.grid(True)
    ax_.invert_yaxis()

    plt.tight_layout()
    plt.show()
