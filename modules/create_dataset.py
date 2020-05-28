from const import wells, carotage_types, raw_log_dir, log_dir, slices_dir, wellheads,\
    raw_cube, sourceX, sourceY, INLINE_3D, CROSSLINE_3D, nsamples, dt, ilines, xlines, \
    max_distance, well_width, slice_coord_path, norm_dict_path
from utils import projection
from data_gen1 import dump_normalization_values
from data_types import Point
import numpy as np
import segyio
import pandas as pd
import pickle
import re


def create_slice_coord_dict(path):
    """Create a dictionary with slice coordinates"""

    # fill in holes
    d = dict()
    for c in xlines:
        idx = CROSSLINE_3D == c
        lines = INLINE_3D[idx]
        x = sourceX[idx]
        y = sourceY[idx]

        ax = np.empty(651)
        ax[:] = np.nan
        ay = np.empty(651)
        ay[:] = np.nan
        for l, xx, yy in zip(lines, x, y):
            ax[l - 100] = xx
            ay[l - 100] = yy
        if len(lines) < 651:
            stepx = (-x.max() + x.min()) / (lines.max() - lines.min())
            stepy = (y.max() - y.min()) / (lines.max() - lines.min())
            for i in range(len(ax)):
                if np.isnan(ax[i]):
                    ax[i] = ax[i - 1] + stepx
                    ay[i] = ay[i - 1] + stepy

        d.update({(c, i + 100): (xx, yy) for i, xx, yy in zip(range(len(ax)), ax, ay)})

    # create coord dict
    slice_coord_dict = {'iline': {}, 'xline': {}}
    for iline in ilines:
        slice_coord_dict['iline'][iline] = np.array([d[(xline, iline)] for xline in xlines])
    for xline in xlines:
        slice_coord_dict['xline'][xline] = np.array([d[(xline, iline)] for iline in ilines])

    with open(path, 'wb') as f:
        pickle.dump(slice_coord_dict, f)


def get_slice_coord_dict():
    with open(slice_coord_path, 'rb') as f:
        slice_coord_dict = pickle.load(f)
    return slice_coord_dict


slice_coord_dict = get_slice_coord_dict()


def create_zero_cube(seg_path, zero_file_path, samples, ilines, xlines):
    spec = segyio.spec()
    spec.sorting = 2
    spec.format = 1
    spec.samples = samples
    spec.ilines = ilines
    spec.xlines = xlines

    trace = np.zeros(len(spec.samples), dtype=np.float32)

    with segyio.open(seg_path) as segyfile:
        header = segyfile.header
        with segyio.create(zero_file_path, spec) as f:

            tr = 0
            for il in spec.ilines:
                for xl in spec.xlines:
                    f.header[tr] = {
                        segyio.su.offset : 1,
                        segyio.su.iline  : il,
                        segyio.su.xline  : xl,
                        segyio.TraceField.TRACE_SAMPLE_COUNT : len(spec.samples),
                        segyio.TraceField.SourceX: header[tr][segyio.TraceField.SourceX],
                        segyio.TraceField.SourceY: header[tr][segyio.TraceField.SourceY]
                    }
                    f.trace[tr] = trace
                    tr += 1

            f.bin.update(
                tsort=segyio.TraceSortingFormat.INLINE_SORTING
            )


# фильтрация каротажа
def filter_carotage(well_log, carotage_types, max_depth, max_sigma=2):
    """Filter carotage values for outliers."""
    window = 599
    diff = well_log[carotage_types] - well_log[carotage_types].rolling(window, center=True).median().\
        fillna(method='ffill').fillna(method='bfill')
    sigma = diff.std()

    well_log_filtered = well_log.copy()
    mask = diff.abs() > (max_sigma * sigma)
    mask[well_log['tvd'] >= max_depth] = False
    well_log_filtered[mask] = np.nan
    return well_log_filtered


# полная генерация логов
def generate_logs(well_list, log_dir, min_depth=0, max_depth=nsamples * dt):
    """Generate csv logs"""
    for well_name in well_list:
        print(well_name)
        las_df = pd.read_csv(raw_log_dir / f'{well_name}.log.csv',
                             delimiter='\t', na_values='1.000000e+30')

        idx = np.logical_and(las_df['Inline'].values[0] == INLINE_3D, las_df['Crossline'].values[0] == CROSSLINE_3D)
        las_df['x'] = sourceX[idx][0]
        las_df['y'] = sourceY[idx][0]

        fun = lambda s: re.match('(.+)\(.+\)', s)[1] if re.match('(.+)\(.+\)', s) else s
        las_df = las_df.rename(columns=fun)
        las_df = las_df.rename(columns={'Time': 't'})

        las_df = las_df.loc[(min_depth <= las_df['t']) & (las_df['t'] <= max_depth)]

        las_df.to_csv(log_dir / (well_name + '.csv'), index=False)


def gen_tgt_mask(slice_coords, vertical_grid, las_df, carotage_types, well_width):
    """
    Generates target and mask for a well projection on a vertical seismic slice

    slice_coords: seismic traces coords of a slice, sorted, meters
    v_grid: vertical grid of a seismic slice, milliseconds
    las_df: las data frame
    carotage_types: ...
    trace_width: width of the target channel

    returns: target and mask, HxWxC numpy arrays
    """
    horizontal_grid = np.sqrt(np.square(slice_coords - slice_coords[0]).sum(axis=1))
    assert all(np.diff(horizontal_grid) > 0)
    assert all(np.diff(vertical_grid) > 0)
    width2 = well_width // 2

    pt1 = Point(*slice_coords[0])
    pt2 = Point(*slice_coords[-1])
    # horizontal projection on the slice in the original coordinates
    well_projection_xy = np.array([projection(pt1, pt2, Point(*p)) for p in las_df[['x', 'y']].values])
    # horizontal projection on the slice in the slice coordinates
    well_projection_1d = np.sqrt(np.square(well_projection_xy - slice_coords[0]).sum(axis=1))

    target = np.zeros((len(vertical_grid), len(horizontal_grid), len(carotage_types)), dtype=np.float32)
    mask = np.zeros_like(target, dtype=bool)

    idx = np.digitize(well_projection_1d, horizontal_grid)
    idx[idx == 0] += 1
    idx[idx == len(horizontal_grid)] -= 1
    a1 = np.abs(well_projection_1d - horizontal_grid[idx])
    a2 = np.abs(well_projection_1d - horizontal_grid[idx - 1])
    idx[a2 < a1] -= 1
    las_df['h_index'] = idx

    idx = np.digitize(las_df['t'], vertical_grid)
    idx[idx == 0] += 1
    idx[idx == len(vertical_grid)] -= 1
    a1 = np.abs(las_df['t'] - vertical_grid[idx])
    a2 = np.abs(las_df['t'] - vertical_grid[idx - 1])
    idx[a2 < a1] -= 1
    las_df['v_index'] = idx

    gp = las_df.groupby(['h_index', 'v_index']).mean().reset_index().sort_values('t')
    # iy_ = tuple(np.repeat(gp.v_index.values[..., None], well_width, axis=1))
    # ix_ = tuple([range(max(0, i - (well_width // 2)),
    #                    min(len(horizontal_grid), i - (well_width // 2) + well_width))
    #              for i in gp.h_index])
    iy_ = tuple(np.repeat(gp.v_index.values[..., None], well_width, axis=1))
    ix_ = []
    for i in gp.h_index:
        x1 = i - (well_width // 2)
        x1 = max(0, x1)
        x1 = min(len(horizontal_grid) - well_width, x1)
        ix_.append(range(x1, x1 + well_width))

    mask[iy_, ix_] = ~np.isnan(gp[carotage_types].values[:, None, :])
    target[iy_, ix_] = gp[carotage_types].values[:, None, :]
    target[~mask] = 0

    return target, mask


def project_wells_onto_slice(slice_num, slice_type, well_list, carotage_types, well_width, verbose=False):
    """Finds projections of wells onto a given seismic slice"""
    slice_coords = slice_coord_dict[slice_type][slice_num]
    vertical_grid = np.arange(nsamples) * dt
    target, mask = None, None

    for well_name in well_list:
        if verbose:
            print(' ', well_name)
        # las_df = read_las(well_name, carotage_types)
        # dev_df, kb = read_dev(well_name)
        # layer_0_corr = get_layer0_correction(well_name)
        # las_df = extend_las(las_df, dev_df, layer_0_corr)
        las_df = pd.read_csv(log_dir / (well_name + '.csv'))

        t, m = gen_tgt_mask(slice_coords, vertical_grid, las_df, carotage_types, well_width)
        if target is None:
            target = t.copy()
            mask = m.copy()
        else:
            target[m] = t[m]
            mask = np.logical_or(mask, m)

    return target, mask


def find_proxi_wells(slice_coords, wells, max_distance):
    """Find nearest wells for a given coordinate on a slice"""
    pt1 = Point(*slice_coords[0])
    pt2 = Point(*slice_coords[-1])
    proxi_wells = []
    for well_name in wells:
        ptw = Point(*wellheads.loc[well_name, ['X-Coord', 'Y-Coord']].values)
        ptp = projection(pt1, pt2, ptw)
        dist = np.sqrt(np.square(np.array(ptw) - np.array(ptp)).sum())
        if dist <= max_distance:
            proxi_wells.append(well_name)
    return proxi_wells


def find_nearest_slice(wells):
    """Find nearest slice for a given set of wells"""
    iline_coords = []
    for iline in ilines:
        coords = slice_coord_dict['iline'][iline]
        iline_coords.append((Point(*coords[0]), Point(*coords[-1])))
    xline_coords = []
    for xline in xlines:
        coords = slice_coord_dict['xline'][xline]
        xline_coords.append((Point(*coords[0]), Point(*coords[-1])))
    slice_well_list = []
    for well_name in wells:
        ptw = Point(*wellheads.loc[well_name, ['X-Coord', 'Y-Coord']].values)
        proj = [projection(pt1, pt2, ptw) for pt1, pt2 in iline_coords]
        dist = np.sqrt(np.square(np.array(ptw) - np.array(proj)).sum(axis=1))
        iline = ilines[np.argmin(dist)]
        proj = [projection(pt1, pt2, ptw) for pt1, pt2 in xline_coords]
        dist = np.sqrt(np.square(np.array(ptw) - np.array(proj)).sum(axis=1))
        xline = xlines[np.argmin(dist)]
        slice_well_list.append(('iline', iline, well_name))
        slice_well_list.append(('xline', xline, well_name))
    return slice_well_list


def slice_crossline(crossline):
    idx = CROSSLINE_3D == crossline
    assert len(idx) > 0, 'crossline out of range'
    assert all(np.diff(INLINE_3D[idx]) > 0)
    a = np.zeros((nsamples, max(ilines) - min(ilines) + 1), dtype=raw_cube.dtype)
    a[:, INLINE_3D[idx].min() - min(ilines): INLINE_3D[idx].min() - min(ilines) + raw_cube[idx].shape[0]] = \
        raw_cube[idx].T
    return a


def slice_inline(inline):
    idx = INLINE_3D == inline
    assert len(idx) > 0, 'inline out of range'
    assert all(np.diff(CROSSLINE_3D[idx]) > 0)
    a = np.zeros((nsamples, max(xlines) - min(xlines) + 1), dtype=raw_cube.dtype)
    a[:, CROSSLINE_3D[idx].min() - min(xlines): CROSSLINE_3D[idx].min() - min(xlines) + raw_cube[idx].shape[0]] = \
        raw_cube[idx].T
    return a


# генерация единицы данных первого типа (проекция многих скважин на один срез)
def get_slice_data(slice_num, slice_type, wells_list, max_distance, carotage_types, well_width):
    """Read and prepare slice data"""
    if slice_type == 'iline':
        seismic_slice = slice_inline(slice_num)
    elif slice_type == 'xline':
        seismic_slice = slice_crossline(slice_num)
    else:
        raise ValueError('wrong slice type')
    slice_coords = slice_coord_dict[slice_type][slice_num]

    proxi_wells = find_proxi_wells(slice_coords, wells_list, max_distance)
    if len(proxi_wells) == 0:
        print(f'{slice_type} {slice_num} has no proxi wells at {max_distance}m')
        return None

    target, mask = project_wells_onto_slice(slice_num, slice_type, proxi_wells, carotage_types, well_width)
    projections = {carotage_type: {'target': target[..., i], 'mask': mask[..., i]}
                   for i, carotage_type in enumerate(carotage_types)}
    slice_data = {'seismic': seismic_slice,
                  'projections': projections,
                  'proxi_wells': proxi_wells}
    return slice_data


# генерация единицы данных второго типа (проекция одной скважины на один срез)
def get_slice_data_single_well(slice_num, slice_type, well_name, carotage_types, well_width):
    """Read and prepare slice data with a single well (second type of dataset preparation)"""
    if slice_type == 'iline':
        seismic_slice = slice_inline(slice_num)
    elif slice_type == 'xline':
        seismic_slice = slice_crossline(slice_num)
    else:
        raise ValueError('wrong slice type')
    target, mask = project_wells_onto_slice(slice_num, slice_type, [well_name], carotage_types, well_width)
    projections = {carotage_type: {'target': target[..., i], 'mask': mask[..., i]}
                   for i, carotage_type in enumerate(carotage_types)}
    slice_data = {'seismic': seismic_slice,
                  'projections': projections,
                  'proxi_wells': [well_name]}
    return slice_data


# полная генерация датасета 1-го типа (проекция многих скважин на один срез)
def process_all_cube(well_list, slice_list, max_distance, carotage_types, well_width, slice_dir):
    """Process the entire seismic cube, dumping the resulting slice data"""
    for slice_type, slice_num in slice_list:
        print(f'{slice_type} {slice_num}')
        slice_data = get_slice_data(slice_num, slice_type, well_list, max_distance, carotage_types, well_width)
        if slice_data:
            with open(slice_dir / f'{slice_type}_{slice_num}.pkl', 'wb') as f:
                pickle.dump(slice_data, f)


# полная генерация датасета 2-го типа (проекция одной скважины на один срез)
def process_single_wells(slice_well_list, carotage_types, well_width, slice_dir):
    """Process the entire seismic cube in the one well per slice variant, dumping the resulting slice data"""
    for slice_type, slice_num, well_name in slice_well_list:
        print(f'{slice_type} {slice_num}, {well_name}')
        slice_data = get_slice_data_single_well(slice_num, slice_type, well_name, carotage_types, well_width)
        if slice_data:
            with open(slice_dir / f'{slice_type}_{slice_num}_{well_name}.pkl', 'wb') as f:
                pickle.dump(slice_data, f)


def create_slice_well_list(wells, slice_range=10):
    slice_well_list = []
    for well_name in wells:
        las_df = pd.read_csv(log_dir / (well_name + '.csv'))
        inline, clossline = las_df[['Inline', 'Crossline']].values[0]
        for slice_type, line in zip(['iline', 'xline'], [inline, clossline]):
            for s in range(line - slice_range, line + slice_range + 1):
                slice_well_list.append((slice_type, s, well_name))
    return slice_well_list


if __name__ == '__main__':
    generate_logs(wells, log_dir)

    slice_well_list = create_slice_well_list(wells)
    process_single_wells(slice_well_list, carotage_types, well_width, slices_dir)
    dump_normalization_values(slices_dir, path=norm_dict_path, overwrite=True)
