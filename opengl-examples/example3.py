# -*- coding: utf-8 -*-
# 六面体数据
# ------------------------------------------------------
#    v4----- v5
#   /|      /|
#  v0------v1|
#  | |     | |
#  | v7----|-v6
#  |/      |/
#  v3------v2

# 顶点集


from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
import numpy as np

glutInit()
glutCreateWindow(b'Example 3')

vertices = np.array([
    -0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, -0.5, 0.5,
    -0.5, -0.5, 0.5,
    -0.5, 0.5, -0.5,
    0.5, 0.5, -0.5,
    0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5
], dtype=np.float32)

# 索引集

indices = np.array([
    0, 1, 2, 3,  # v0-v1-v2-v3 (front)
    4, 5, 1, 0,  # v4-v5-v1-v0 (top)
    3, 2, 6, 7,  # v3-v2-v6-v7 (bottom)
    5, 4, 7, 6,  # v5-v4-v7-v6 (back)
    1, 5, 6, 2,  # v1-v5-v6-v2 (right)
    4, 0, 3, 7  # v4-v0-v3-v7 (left)
], dtype=np.int)

vbo_vertices = vbo.VBO(vertices)
vbo_vertices.bind()
glInterleavedArrays(GL_V3F, 0, None)

vbo_indices = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)
vbo_indices.bind()


def draw():
    glDrawElements(GL_QUADS, int(vbo_indices .size/4), GL_UNSIGNED_INT, None)
    glutSwapBuffers()                    # 切换缓冲区，以显示绘制内容


if __name__ == '__main__':
    glutDisplayFunc(draw)
    glutMainLoop()
