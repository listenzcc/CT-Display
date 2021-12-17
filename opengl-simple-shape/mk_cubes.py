'''
FileName: mk_cubes.py
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

cube_indices = np.array([
    [0, 3, 2, 1],  # front
    [4, 5, 6, 7],  # back
    [0, 1, 5, 4],  # up
    [2, 3, 7, 6],  # bottom
    [0, 4, 7, 3],  # left
    [1, 2, 6, 5],  # right
], dtype=np.int32)

# %%
xs, ys, zs = 5, 5, 1
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
            ver = cube_vertices * 0.3 + xyz * 1
            ind = cube_indices + idx
            # vertices_list.append(ver)
            # indices_list.append(ind)
            y_vertices_list.append(ver)
            y_indices_list.append(ind)
            idx += 8

        x_vertices_list.append(y_vertices_list)
        x_indices_list.append(y_indices_list)

    vertices_list.append(x_vertices_list)
    indices_list.append(x_indices_list)

vertices = np.array(vertices_list)
indices = np.array(indices_list)

vertices.shape, indices.shape


# %%
# vers = np.repeat(
#     [np.repeat([np.repeat([cube_vertices * 0.5], zs, axis=0)], ys, axis=0)], xs, axis=0)

# inds = np.repeat(
#     [np.repeat([np.repeat([cube_indices], zs, axis=0)], ys, axis=0)], xs, axis=0)

# print(vers.shape, inds.shape)

# # %%
# grid = np.meshgrid(range(ys), range(xs), range(zs), 8)
# print(len(grid), [e.shape for e in grid])

# # %%
# vers[:, :, :, :, 0] += grid[1]
# vers[:, :, :, :, 1] += grid[0]
# vers[:, :, :, :, 2] += grid[2]

# # %%
# idx = 0
# for x in tqdm(range(xs)):
#     for y in range(ys):
#         for z in range(zs):
#             # vers[x][y][z] += np.array([x, y, z], dtype=np.float32)
#             inds[x][y][z] += idx
#             idx += 24
#             pass


# # %%
# a = np.array(range(np.prod([xs, ys, zs])),
#              dtype=np.int64).reshape((xs, ys, zs)) * 24

# for j in tqdm(range(6)):
#     for k in range(4):
#         inds[:, :, :, j, k] += a


# # %%
# vertices = vers
# indices = inds

# # %%
