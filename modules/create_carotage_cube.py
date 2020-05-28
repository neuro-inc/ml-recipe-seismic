import segyio
from const import data_dir, seg_path, norm_dict_path, model_dir
from model1 import uResNet34

from keras import backend as K
from model1 import uResNet34

import numpy as np
import cv2
import pandas as pd
import pickle
from pathlib import Path
from shutil import copyfile


zero_file = data_dir / 'zero_file.sgy'
seismic_cut = data_dir / 'seismic_cut.sgy'
cube_path = data_dir / 'cubes_'
cube_path.mkdir(exist_ok=True)


# get seismic cube parameters
with segyio.open(seg_path) as segyfile:
    segyfile.mmap()
    ilines = segyfile.ilines
    xlines = segyfile.xlines
    samples = segyfile.samples


models = {
    'GK': 'uResNet34.GK.sz1024x512.mxd.05-0.63.hdf5',
    'NKTD': 'uResNet34.NKTD.sz1024x512.sgl.06-0.63.hdf5',
}
modelsize = (1024, 512)


def create_seismic_cut():
    # create_zero_cube
    spec = segyio.spec()

    spec.sorting = 2
    spec.format = 1
    spec.samples = samples[:1024]
    spec.ilines = ilines
    spec.xlines = xlines

    trace = np.zeros(len(spec.samples), dtype=np.float32)

    with segyio.open(seg_path) as segyfile:
        header = segyfile.header
        with segyio.create(zero_file, spec) as f:

            tr = 0
            for il in spec.ilines:
                for xl in spec.xlines:
                    f.header[tr] = {
                        segyio.su.offset: 1,
                        segyio.su.iline: il,
                        segyio.su.xline: xl,
                        segyio.TraceField.TRACE_SAMPLE_COUNT: len(spec.samples),
                        segyio.TraceField.SourceX: header[tr][segyio.TraceField.SourceX],
                        segyio.TraceField.SourceY: header[tr][segyio.TraceField.SourceY]
                    }
                    f.trace[tr] = trace
                    tr += 1

            f.bin.update(
                tsort=segyio.TraceSortingFormat.INLINE_SORTING
            )

    # create seismic cut
    copyfile(zero_file, seismic_cut)
    with segyio.open(seg_path) as segyfile:
        segyfile.mmap()
        with segyio.open(seismic_cut, mode='r+') as f:
            f.mmap()
            for iline in ilines:
                seismic_slice = segyfile.iline[iline]
                f.iline[iline] = seismic_slice[..., : len(spec.samples)]
            f.flush()


def preprocess(x, dsize, mean, std):
    # in:  n x h0 x w0
    # ret: n x h1 x w1 x 1
    x = np.moveaxis(x, 0, -1)
    x = (x - mean) / std
    x = cv2.resize(x, dsize=dsize[::-1], interpolation=cv2.INTER_CUBIC)
    x = np.moveaxis(x, -1, 0)[..., np.newaxis]
    return x


def postprocess(x, dsize, mean, std):
    # in:  n x h1 x w1
    # ret: n x h0 x w0
    x = np.moveaxis(x, 0, -1)
    x = cv2.resize(x, dsize=dsize[::-1], interpolation=cv2.INTER_NEAREST)
    x = x * std + mean
    x = np.moveaxis(x, -1, 0)
    return x


if __name__ == '__main__':
    originalsize = (1024, 810)  # inline
    carotage = 'GK'

    batch_size = 16
    with open(norm_dict_path, 'rb') as f:
        norm_dict = pickle.load(f)
    seismic_mean, seismic_std = norm_dict['seismic']['mean'], norm_dict['seismic']['std']

    # build model
    mean, std = norm_dict[carotage]['mean'], norm_dict[carotage]['std']
    model_weights = model_dir / models[carotage]
    model_class = uResNet34

    cube_file = cube_path / (carotage + '.sgy')
    copyfile(zero_file, cube_file)

    K.clear_session()
    model = model_class(input_size=modelsize, weights=model_weights, n_carotage=1)

    with segyio.open(seismic_cut, "r") as segyfile:
        segyfile.mmap()
        with segyio.open(cube_file, mode='r+') as f:
            f.mmap()
            for i in range(0, len(ilines), batch_size):
                iline_batch = ilines[i: min(i + batch_size, len(ilines))]
                print(iline_batch)
                seismic = np.asarray([np.copy(x) for x in segyfile.iline[iline_batch.min(): iline_batch.max() + 1]])
                seismic = np.moveaxis(seismic, 2, 1)
                seismic = preprocess(seismic, modelsize, seismic_mean, seismic_std)
                mask = np.zeros_like(seismic, dtype=np.float32)

                pred = model.predict([seismic, mask], batch_size=8)[..., 0]
                pred = postprocess(pred, originalsize, mean, std)

                pred = np.moveaxis(pred, 1, 2)
                for idx, iline_batch_index in enumerate(iline_batch):
                    f.iline[iline_batch_index] = pred[idx]
