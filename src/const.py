#
# Global constants used throughout the code
#

from pathlib import Path
import segyio
import numpy as np
import pandas as pd
import os


# raw data paths
ROOT_PATH = Path(__file__).absolute().parent.parent
storage_root = Path(os.environ.get('DATA_PATH', ROOT_PATH))

data_dir = storage_root / 'data'
seg_path = data_dir / 'Seismic_data.sgy'
# las_dir = data_dir / 'las/raw'        # raw carotage data
las_dir = data_dir / 'las/smoothed'     # smoothed carotage data

# preprocessed data paths
log_dir = data_dir / 'processed_las/smoothed'
log_dir.mkdir(exist_ok=True, parents=True)
slices_dir = data_dir / 'slices/smoothed'
slices_dir.mkdir(exist_ok=True, parents=True)

# model paths
model_log_dir = storage_root / 'train_log'
model_log_dir.mkdir(exist_ok=True, parents=True)
dumps_dir = storage_root / 'dumps'
dumps_dir.mkdir(exist_ok=True, parents=True)
model_dir = storage_root / 'models/smoothed'
model_dir.mkdir(exist_ok=True, parents=True)

model_input_size = (480, 512)

# carotage types used throughout the project
carotage_types = ['Density', 'Sonic', 'Gamma_Ray', 'Porosity', 'P_Impedance', 'P_Impedance_rel', 'Vp']

# well list
wells = ['F02-1', 'F03-2', 'F03-4', 'F06-1']

# crossvalidation split
crossval_dict = [
    {'train': ['F03-2', 'F03-4', 'F06-1'], 'test': ['F02-1']},
    {'train': ['F02-1', 'F03-4', 'F06-1'], 'test': ['F03-2']},
    {'train': ['F02-1', 'F03-2', 'F06-1'], 'test': ['F03-4']},
    {'train': ['F02-1', 'F03-2', 'F03-4'], 'test': ['F06-1']},
]

# data normalisation dictionary
norm_dict_path = data_dir / 'norm_dict_smoothed.pkl'

# max distance from a seismic slice to a well to be projected
max_distance = 180  # meters. Not currently used
slice_range = 10    # slices

# projection width in seismic traces
well_width = 30

# seismic cube parameters
# NOTICE: the seismic data is in `pre_stack` mode, hence we address unstructured data
with segyio.open(seg_path, strict=False) as segyfile:
    segyfile.mmap()
    sourceX = segyfile.attributes(segyio.TraceField.SourceX)[:] / 10
    sourceY = segyfile.attributes(segyio.TraceField.SourceY)[:] / 10
    INLINE_3D = segyfile.attributes(segyio.TraceField.INLINE_3D)[:]
    CROSSLINE_3D = segyfile.attributes(segyio.TraceField.CROSSLINE_3D)[:]
    raw_cube = segyfile.trace.raw[:]                # raw seismic data, [ntraces x nsamples]
trace_coords = np.column_stack([sourceX, sourceY])  # trace lateral coordinates
ilines = sorted(np.unique(INLINE_3D))               # iline list
xlines = sorted(np.unique(CROSSLINE_3D))            # crossline list
nsamples = raw_cube.shape[-1]                       # number of trace samples
dt = 4                                              # delta-time, the sample rate, ms

# well head coordinates
wellheads = pd.read_csv(data_dir / 'wellheads.csv', index_col='WellName')

# iline/crossline coordinates
slice_coord_path = data_dir / 'slice_coord_dict.pkl'
