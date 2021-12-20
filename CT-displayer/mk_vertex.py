'''
FileName: mk_vertex.py
Author: Chuncheng
Version: V0.0
Purpose: Make vertex and indices for CT-Data

# The cube looks like
#    v4------v5
#   /|      /|
#  v0------v1|
#  | |     | |
#  | v7----|-v6
#  |/      |/
#  v3------v2
'''

# %%
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydicom as dicom
import SimpleITK as sitk
import pandas as pd

from tqdm.auto import tqdm

# %%
# GL_C4F_N3F_V3F

# The position of the 8 vertices
cube_vertices = np.array([
    [-1, 1, 1],
    [1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],

    [-1, 1, 1],
    [1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1]
], dtype=np.float32)

# The normals of the 8 vertices,
# now they are the same as the positions.
# ? I do not know if I have to normalize the vectors
cube_normals = np.array([
    [-1, 1, 1],
    [1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],

    -1 * np.array([-1, 1, 1]),
    -1 * np.array([1, 1, 1]),
    -1 * np.array([1, -1, 1]),
    -1 * np.array([-1, -1, 1]),
    -1 * np.array([-1, 1, -1]),
    -1 * np.array([1, 1, -1]),
    -1 * np.array([1, -1, -1]),
    -1 * np.array([-1, -1, -1])
], dtype=np.float32) / np.sqrt(3)

# The colors of the 8 vertices.
cube_colors = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 1],

    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 1]
], dtype=np.float32)

# The indices of the 8 vertices
cube_indices = np.array([
    [0, 3, 2, 1],  # front
    [4, 5, 6, 7],  # back
    [0, 1, 5, 4],  # up
    [2, 3, 7, 6],  # bottom
    [0, 4, 7, 3],  # left
    [1, 2, 6, 5],  # right

    8 + np.array([0, 1, 2, 3]),  # front
    8 + np.array([4, 7, 6, 5]),  # back
    8 + np.array([0, 4, 5, 1]),  # up
    8 + np.array([2, 6, 7, 3]),  # bottom
    8 + np.array([0, 3, 7, 4]),  # left
    8 + np.array([1, 5, 6, 2]),  # right
], dtype=np.int32)


# %%
folder = os.path.join(
    __file__, r'../../CT-data/01001/s1-post')

# folder = os.path.join(
#     __file__, r'../../CT-data/01002/s2-pre')

dcm_files = [os.path.join(folder, e) for e in os.listdir(folder)]
dcm_files = sorted(dcm_files, key=lambda x: int(
    os.path.basename(x).split('.')[0]))
print('Found {} files, with extents of {}'.format(
    len(dcm_files), set([e.split('.')[-1] for e in dcm_files])))

img_list = [sitk.GetArrayFromImage(sitk.ReadImage(e))
            for e in tqdm(dcm_files, 'Reaing .dcm files')]
img_array = np.concatenate(img_list, axis=0)
print('Generate img_array with shape of {}'.format(img_array.shape))

# %%
# Min, Max = 1500, 2000
Min, Max = 1000, 2000  # 1000, np.max(img_array)
cMin, cMax = 100, 2200

# %%
xrange, yrange, zrange = np.where((img_array > Min) * (img_array < Max))
print(xrange.shape, yrange.shape, zrange.shape)

# %%
vertices_list = []
indices_list = []
idx = 0

for j in tqdm(range(len(xrange))):
    x, y, z = xrange[j], yrange[j], zrange[j]
    xyz = np.array([x - img_array.shape[0] / 2,
                    y - img_array.shape[1] / 2,
                    z - img_array.shape[2] / 2], dtype=np.float32)
    ver = cube_vertices * 0.5 + xyz
    nor = cube_normals
    ind = cube_indices + idx

    # value = np.mean(img_array[x-3:x+3, y-3:y+3, z-3:z+3])
    if np.count_nonzero(img_array[x-1:x+1, y-1:y+1, z-1:z+1] < Min) > 7:
        continue

    value = img_array[x, y, z]
    c = (value - cMin) / (cMax - cMin)
    if c > 1:
        c = 1
    if c < 0:
        c = 0
    col = cube_colors.copy()
    col[:, -1] = col[:, -1] * c

    vertices_list.append(np.concatenate([col, nor, ver], axis=1))
    indices_list.append(ind)

    idx += cube_vertices.shape[0]

vertices = np.array(vertices_list)
indices = np.array(indices_list)

vertices.shape, indices.shape
# %%
if __name__ == '__main__':
    for j in [int(e) for e in np.linspace(10, int(img_array.shape[1])-10, 5)]:
        fig = px.imshow(img_array[:, j], title=f'The {j}th slice')
        fig.show()

# %%
