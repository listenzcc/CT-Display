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
cube_vertices = np.array([
    [-1, 1, 1],
    [1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1]
], dtype=np.float32)

cube_colors = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]

], dtype=np.float32)

cube_indices = np.array([
    [0, 3, 2, 1],  # front
    [4, 5, 6, 7],  # back
    [0, 1, 5, 4],  # up
    [2, 3, 7, 6],  # bottom
    [0, 4, 7, 3],  # left
    [1, 2, 6, 5],  # right
], dtype=np.int32)

# %%
# %%
folder = os.path.join(
    __file__, r'../../CT-data/01001/ZHU CHUAN WEN_K0312074-pos')

# folder = os.path.join(
#     __file__, r'../../CT-data/01002/ZHANG SHU XIU_D677583-pre')

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
Min, Max = 800, 1200  # 1000, np.max(img_array)

# %%
xrange, yrange, zrange = np.where((img_array > Min) * (img_array < Max))
print(xrange.shape, yrange.shape, zrange.shape)

# %%
vertices_list = []
colors_list = []
indices_list = []
idx = 0

for j in tqdm(range(len(xrange))):
    x, y, z = xrange[j], yrange[j], zrange[j]
    xyz = np.array([x - img_array.shape[0]/2, y - img_array.shape[1] /
                   2, z - img_array.shape[2]/2], dtype=np.float32)
    ver = cube_vertices * 0.5 + xyz

    ind = cube_indices + idx

    c = np.min((1, (img_array[x][y][z] - Min) /
               (np.min((Min * 2, Max)) - Min)))
    col = cube_colors * c

    vertices_list.append(ver)
    colors_list.append(col)
    indices_list.append(ind)

    idx += 8

vertices = np.array(vertices_list)
colors = np.array(colors_list)
indices = np.array(indices_list)

vertices = np.concatenate([colors, vertices], axis=-1)
vertices.shape, indices.shape
# %%
if __name__ == '__main__':
    for j in [int(e) for e in np.linspace(0, int(img_array.shape[0])-1, 5)]:
        fig = px.imshow(img_array[j], title=f'The {j}th slice')
        fig.show()

# %%
