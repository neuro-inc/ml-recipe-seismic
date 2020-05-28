# This file contains global constants used throughout the code, mostly paths, filenames, and train/test splits.

from pathlib import Path
import segyio
import numpy as np
import pandas as pd


# пути  к исходным данным
data_dir = Path('../data/F3_Demo_2016_training_v6')
seg_path = data_dir / 'Rawdata/Seismic_data.sgy'
custom_dir = data_dir / 'custom'
custom_dir.mkdir(exist_ok=True, parents=True)
raw_log_dir = custom_dir / 'las/smoothed'

# пути к генерируемым данным
log_dir = custom_dir / 'processed_las/smoothed'
log_dir.mkdir(exist_ok=True, parents=True)
slices_dir = custom_dir / 'slices/smoothed'
slices_dir.mkdir(exist_ok=True)

# модели
model_log_dir = Path('../train_log')
model_log_dir.mkdir(exist_ok=True)
dumps_dir = Path('../dumps')
dumps_dir.mkdir(exist_ok=True)
model_dir = Path('../models')
model_dir.mkdir(exist_ok=True)

# типы каротажей
carotage_types = ['Density', 'Sonic', 'Gamma_Ray', 'Porosity', 'P_Impedance', 'P_Impedance_rel', 'Vp']

# скважины
wells = ['F02-1', 'F03-2', 'F03-4', 'F06-1']

# разбиение на фолды для кроссвалидации
crossval_dict = [
    {'train': ['F03-2', 'F03-4', 'F06-1'], 'test': ['F02-1']},
    {'train': ['F02-1', 'F03-4', 'F06-1'], 'test': ['F03-2']},
    {'train': ['F02-1', 'F03-2', 'F06-1'], 'test': ['F03-4']},
    {'train': ['F02-1', 'F03-2', 'F03-4'], 'test': ['F06-1']},
]

# нормализация данных
norm_dict_path = custom_dir / 'norm_dict_smoothed.pkl'

# макс расстояние до сейсмического среза для проекции
max_distance = 180  # meters

# ширина проекции в трассах
# well_width = 22     # traces
well_width = 30     # traces

# параметры куба
with segyio.open(seg_path, strict=False) as segyfile:
    segyfile.mmap()
    sourceX = segyfile.attributes(segyio.TraceField.SourceX)[:] / 10
    sourceY = segyfile.attributes(segyio.TraceField.SourceY)[:] / 10
    INLINE_3D = segyfile.attributes(segyio.TraceField.INLINE_3D)[:]
    CROSSLINE_3D = segyfile.attributes(segyio.TraceField.CROSSLINE_3D)[:]
    raw_cube = segyfile.trace.raw[:]
trace_coords = np.column_stack([sourceX, sourceY])
ilines = sorted(np.unique(INLINE_3D))
xlines = sorted(np.unique(CROSSLINE_3D))
nsamples = raw_cube.shape[-1]
dt = 4

# датафрейм с забоями скважин
wellheads = pd.read_csv(custom_dir / 'wellheads.csv', index_col='WellName')

# словарь с горизонтальными координами сейсмических срезов
slice_coord_path = custom_dir / 'slice_coord_dict.pkl'
