import numpy as np
from distance import squared_distance

def voronoi_partition(centers, data, dist=squared_distance):
    npoints = data.shape[0]
    ncells = centers.shape[0]
    cells = []
    cell_idx = np.zeros(npoints, dtype=int)
    cell_center_dist = dist(data, centers[0, :])
    for k in range(1, ncells):
        center = centers[k, :]
        center_dist = dist(data, center)
        cell_idx = np.where(cell_center_dist <= center_dist, cell_idx, k)
        np.minimum(cell_center_dist, center_dist, out=cell_center_dist)
    cells = [np.where(cell_idx == k)[0] for k in range(ncells)]
    cells = tuple([list(cell) for cell in cells if len(cell) > 0])
    return Partition(npoints=npoints, ncells=len(cells), cells=cells)


def rand_voronoi_partition(ncenters, data, dist=squared_distance):
    indices = np.random.permutation(data.shape[0])[:ncenters]
    centers = data[indices, :]
    return voronoi_partition(centers, data, dist)


def max_affine_partition(data, maf):
    nhyperplanes = maf.shape[0]
    idx = np.argmax(data.dot(maf.T), axis=1)
    cells = []
    for k in range(nhyperplanes):
        cells.append(np.where(idx == k)[0])
    cells = [c for c in cells if len(c) > 0]
    return Partition(npoints=data.shape[0], ncells=len(cells), cells=tuple(cells))

class Partition(object):
    __slots__ = ['npoints', 'ncells', 'cells', 'extra']

    def __init__(self, npoints, ncells, cells):
        self.npoints = npoints
        self.ncells = ncells
        assert ncells <= npoints
        self.cells = cells
        self.extra = {}

    def cell_sizes(self):
        sizes = []
        for cell in self.cells:
            sizes.append(len(cell))
        return tuple(sizes)

    def cell_indices(self):
        idx = np.empty(self.npoints, dtype=int)
        for i, cell in enumerate(self.cells):
            idx[cell] = i
        return idx

    def assert_consistency(self):
        assert self.ncells == len(self.cells)
        elems = []
        for k in range(self.ncells):
            cell_k = self.cells[k]
            assert 0 < len(cell_k)
            elems += list(cell_k)
        assert self.npoints == len(elems)
        assert list(range(self.npoints)) == sorted(elems)

    def __eq__(self, other):
        if not isinstance(other, Partition):
            return NotImplementedError('Cannot compare Partition to {}!'.format(type(other)))

        if self.npoints != other.npoints or self.ncells != other.ncells:
            return False
        for cell, other_cell in zip(sorted([tuple(c) for c in self.cells]),
                                    sorted([tuple(c) for c in other.cells])):
            if len(cell) != len(other_cell):
                return False
            if tuple(cell) != tuple(other_cell):
                return False
        return True
