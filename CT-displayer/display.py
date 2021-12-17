'''
FileName: Display.py
Author: Chuncheng
Version: V0.0
Purpose: Display CT-data
'''

# %%
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

from myOpenGLWorld import World, logger

from mk_vertex import vertices, indices
import numpy as np


# %%
world = World()

vbo_vertices = vbo.VBO(vertices)
vbo_vertices.bind()

glInterleavedArrays(GL_C3F_V3F, 0, None)

vbo_indices = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)
vbo_indices.bind()

# %%


def draw():
    IS_PERSPECTIVE = world.IS_PERSPECTIVE
    VIEW = world.VIEW
    EYE = world.EYE
    LOOK_AT = world.LOOK_AT
    EYE_UP = world.EYE_UP
    SCALE_K = world.SCALE_K
    WIN_W = world.WIN_W
    WIN_H = world.WIN_H

    # Clear latest display buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Projection mode
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    if WIN_W > WIN_H:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0] * WIN_W / WIN_H,
                      VIEW[1] * WIN_W / WIN_H,
                      VIEW[2], VIEW[3], VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0] * WIN_W / WIN_H,
                    VIEW[1] * WIN_W / WIN_H,
                    VIEW[2], VIEW[3], VIEW[4], VIEW[5])
    else:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0], VIEW[1],
                      VIEW[2] * WIN_H / WIN_W,
                      VIEW[3] * WIN_H / WIN_W,
                      VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0], VIEW[1],
                    VIEW[2] * WIN_H / WIN_W,
                    VIEW[3] * WIN_H / WIN_W,
                    VIEW[4], VIEW[5])

    # View mode
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    gluLookAt(
        EYE[0] * SCALE_K[0], EYE[1] * SCALE_K[1], EYE[2] * SCALE_K[2],
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )

    glViewport(0, 0, WIN_W, WIN_H)

    # ---- VBO object -----------------------------------------------------
    glColor4f(1.0, 1.0, 1.0, 0.8)
    glDrawElements(GL_QUADS, int(vbo_indices .size/4), GL_UNSIGNED_INT, None)

    # ---- Axes -----------------------------------------------------------
    glBegin(GL_LINES)
    glColor4f(1.0, 0.0, 0.0, 1.0)
    glVertex3f(-0.8, 0.0, 0.0)
    glVertex3f(0.8, 0.0, 0.0)

    glColor4f(0.0, 1.0, 0.0, 1.0)
    glVertex3f(0.0, -0.8, 0.0)
    glVertex3f(0.0, 0.8, 0.0)

    glColor4f(0.0, 0.0, 1.0, 1.0)
    glVertex3f(0.0, 0.0, -0.8)
    glVertex3f(0.0, 0.0, 0.8)
    glEnd()

    # ---- Triangle 1 -----------------------------------------------------------
    glBegin(GL_TRIANGLES)
    glColor4f(1.0, 0.0, 0.0, 1.0)
    glVertex3f(-0.5, -0.366, -0.5)

    glColor4f(0.0, 1.0, 0.0, 1.0)
    glVertex3f(0.5, -0.366, -0.5)

    glColor4f(0.0, 0.0, 1.0, 1.0)
    glVertex3f(0.0, 0.5, -0.5)
    glEnd()

    # ---- Triangle 2 -----------------------------------------------------------
    glBegin(GL_TRIANGLES)
    glColor4f(1.0, 0.0, 0.0, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)

    glColor4f(0.0, 1.0, 0.0, 0.5)
    glVertex3f(0.5, 0.5, 0.5)

    glColor4f(0.0, 0.0, 1.0, 0.5)
    glVertex3f(0.0, -0.366, 0.5)
    glEnd()

    # ---- Flush the display-----------------------------------------------------------
    glutSwapBuffers()


if __name__ == "__main__":
    glutDisplayFunc(draw)

    glutMainLoop()
