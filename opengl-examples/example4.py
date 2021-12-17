# -*- coding: utf-8 -*-
# -------------------------------------------
# Combine VBO and GL-objects
# -------------------------------------------
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
import numpy as np

# Create window to enable the glGenBuffers() function
glutInit()
glutCreateWindow(b'Example 3')

# VBO Setting
vertices = np.array([
    -0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, -0.5, 0.5,
    -0.5, -0.5, 0.5,
    -0.5, 0.5, -0.5,
    0.5, 0.5, -0.5,
    0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5
], dtype=np.float32) * 0.5

indices = np.array([
    0, 1, 2, 3,  # v0-v1-v2-v3 (front)
    4, 5, 1, 0,  # v4-v5-v1-v0 (top)
    3, 2, 6, 7,  # v3-v2-v6-v7 (bottom)
    5, 4, 7, 6,  # v5-v4-v7-v6 (back)
    1, 5, 6, 2,  # v1-v5-v6-v2 (right)
    4, 0, 3, 7  # v4-v0-v3-v7 (left)
], dtype=np.int32)

vbo_vertices = vbo.VBO(vertices)
vbo_vertices.bind()
glInterleavedArrays(GL_V3F, 0, None)

vbo_indices = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)
vbo_indices.bind()

# Display Setting
IS_PERSPECTIVE = True

EYE = np.array([0.0, 0.0, 2.0])
VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])
LOOK_AT = np.array([0.0, 0.0, 0.0])
EYE_UP = np.array([0.0, 1.0, 0.0])
SCALE_K = np.array([1.0, 1.0, 1.0])

MOUSE_X, MOUSE_Y = 0, 0

WIN_W, WIN_H = 800, 600
LEFT_IS_DOWNED = False


def getposture():
    global EYE, LOOK_AT
    dist = np.sqrt(np.power((EYE-LOOK_AT), 2).sum())
    if dist > 0:
        phi = np.arcsin((EYE[1]-LOOK_AT[1])/dist)
        theta = np.arcsin((EYE[0]-LOOK_AT[0])/(dist*np.cos(phi)))
    else:
        phi = 0.0
        theta = 0.0
    return dist, phi, theta


DIST, PHI, THETA = getposture()


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glDepthMask(GL_TRUE)

    # glEnable(GL_CULL_FACE)
    glDisable(GL_MULTISAMPLE)


def draw():
    global IS_PERSPECTIVE, VIEW
    global EYE, LOOK_AT, EYE_UP
    global SCALE_K
    global WIN_W, WIN_H

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
    glColor4f(1.0, 1.0, 1.0, 0.3)
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


def reshape(width, height):
    global WIN_W, WIN_H
    WIN_W, WIN_H = width, height
    glutPostRedisplay()


def mouseclick(button, state, x, y):
    global SCALE_K
    global LEFT_IS_DOWNED
    global MOUSE_X, MOUSE_Y

    MOUSE_X, MOUSE_Y = x, y
    if button == GLUT_LEFT_BUTTON:
        LEFT_IS_DOWNED = state == GLUT_DOWN
    elif button == 3:
        SCALE_K *= 0.9
        print('Zoom In: {}'.format(SCALE_K))
        glutPostRedisplay()

    elif button == 4:
        SCALE_K *= 1.1
        print('Zoom Out: {}'.format(SCALE_K))
        glutPostRedisplay()


def mousemotion(x, y):
    global LEFT_IS_DOWNED
    global EYE, EYE_UP
    global MOUSE_X, MOUSE_Y
    global DIST, PHI, THETA
    global WIN_W, WIN_H

    if LEFT_IS_DOWNED:
        dx = MOUSE_X - x
        dy = y - MOUSE_Y
        MOUSE_X, MOUSE_Y = x, y

        PHI += 2*np.pi*dy/WIN_H
        PHI %= 2*np.pi
        THETA += 2*np.pi*dx/WIN_W
        THETA %= 2*np.pi

        r = DIST*np.cos(PHI)
        EYE[1] = DIST*np.sin(PHI)
        EYE[0] = r*np.sin(THETA)
        EYE[2] = r*np.cos(THETA)

        if 0.5*np.pi < PHI < 1.5*np.pi:
            EYE_UP[1] = -1.0
        else:
            EYE_UP[1] = 1.0

        glutPostRedisplay()


def keydown(key, x, y):
    global DIST, PHI, THETA
    global EYE, LOOK_AT, EYE_UP
    global IS_PERSPECTIVE, VIEW

    if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
        # Move along the x-, y-, z-axis
        if key == b'x':
            LOOK_AT[0] -= 0.01

        elif key == b'X':
            LOOK_AT[0] += 0.01

        elif key == b'y':
            LOOK_AT[1] -= 0.01

        elif key == b'Y':
            LOOK_AT[1] += 0.01

        elif key == b'z':
            LOOK_AT[2] -= 0.01

        elif key == b'Z':
            LOOK_AT[2] += 0.01

        DIST, PHI, THETA = getposture()

        glutPostRedisplay()

    elif key == b'\r':
        # Enter: Zoom In
        EYE = LOOK_AT + (EYE - LOOK_AT) * 0.9
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()

    elif key == b'\x08':
        # Backspace: Zoom Out
        EYE = LOOK_AT + (EYE - LOOK_AT) * 1.1
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()

    elif key == b' ':
        # Space: Toggle perspective mode
        # It will also disable zoom function.
        IS_PERSPECTIVE = not IS_PERSPECTIVE
        glutPostRedisplay()


if __name__ == "__main__":
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
    glutInitDisplayMode(displayMode)
    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 200)

    init()
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouseclick)
    glutMotionFunc(mousemotion)
    glutKeyboardFunc(keydown)

    glutMainLoop()
