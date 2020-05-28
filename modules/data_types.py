import segyio
import numpy as np
from collections import namedtuple
Point = namedtuple('Point', 'x y')

class Segy:
    '''This class reads and stores .sgy seismic data in the form of xlines and ilines.'''

    def __init__(self, segyfn):
        '''Constructor; accepts file name as input'''
        self._segyfile = segyio.open(segyfn)
        self._segyfile.mmap()
        self.xlines = self._segyfile.xlines
        self.ilines = self._segyfile.ilines
        self.xlines_index = {xline: i for i, xline in enumerate(self.xlines)}

    def crop(self, iline_start, iline_stop, xline_start, xline_stop, samples):
        """
        :return: np.array[ilines x xlines x samples]
        """
        assert 0 <= min(samples)
        assert max(samples) < self._segyfile.bin[segyio.BinField.Samples]
        assert iline_start in self.ilines
        assert (iline_start <= iline_stop) and (iline_stop in self.ilines)
        assert xline_start in self.xlines
        assert (xline_start <= xline_stop) and (xline_stop in self.xlines)
        crop = np.array([t[self.xlines_index[xline_start]: self.xlines_index[xline_stop + 1], samples]
                         for t in self._segyfile.iline[iline_start: iline_stop + 1]])
        return crop

    def trace_point(self, trace_index, sample_index):
        return self._segyfile.trace[int(trace_index)][int(sample_index)]
