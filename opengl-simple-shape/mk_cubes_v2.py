'''
FileName: mk_cubes_v2.py
Author: Chuncheng
Version: V0.0
Purpose: Make vertex and indices for cubes

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
import numpy as np
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
    [0, 0, 0, 0.8],
    [0, 0, 1, 0.8],
    [0, 1, 0, 0.8],
    [0, 1, 1, 0.8],
    [1, 0, 0, 0.8],
    [1, 0, 1, 0.8],
    [1, 1, 0, 0.8],
    [1, 1, 1, 0.8],

    [0, 0, 0, 0.8],
    [0, 0, 1, 0.8],
    [0, 1, 0, 0.8],
    [0, 1, 1, 0.8],
    [1, 0, 0, 0.8],
    [1, 0, 1, 0.8],
    [1, 1, 0, 0.8],
    [1, 1, 1, 0.8]
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
xs, ys, zs = 5, 5, 5
# %%

vertices_list = []
indices_list = []
idx = 0

for x in tqdm(range(xs)):
    x_vertices_list = []
    x_indices_list = []

    for y in range(ys):
        y_vertices_list = []
        y_indices_list = []

        for z in range(zs):
            xyz = np.array([x, y, z], dtype=np.float32)

            # GL_C4F_N3F_V3F
            ver = cube_vertices * 0.1 + xyz * 1
            nor = cube_normals
            col = cube_colors
            ind = cube_indices + idx

            # vertices_list.append(ver)
            # indices_list.append(ind)
            # print(col.shape, nor.shape, ver.shape)
            y_vertices_list.append(np.concatenate([col, nor, ver], axis=1))
            y_indices_list.append(ind)

            idx += cube_vertices.shape[0]

        x_vertices_list.append(y_vertices_list)
        x_indices_list.append(y_indices_list)

    vertices_list.append(x_vertices_list)
    indices_list.append(x_indices_list)

vertices = np.array(vertices_list)
indices = np.array(indices_list)

vertices.shape, indices.shape


# %%
