'''
FileName: myOpenGLWorld.py
Author: Chuncheng
Version: V0.0
Purpose: Provide the easy-to-use OpenGL World with amazing controller
'''

# %%

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
import numpy as np
import logging

# %%
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] {%(pathname)s:%(lineno)5d} %(levelname)6s - %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger('Cubes')
logger.setLevel(logging.DEBUG)

# %%


class World(object):
    '''
    Create the World of OpenGL window
    '''

    def __init__(self, title='Some OpenGL World'):
        # Create window object for everything
        glutInit()
        glutCreateWindow(title.encode())
        # Init blender, camera and bind controllers
        self.init_window()
        self.init_blender()
        self.init_camera()
        self.bind_controllers()
        logger.info('World is initialized: {}.'.format(title))
        pass

    def init_camera(self):
        # Init the camera parameters
        self.IS_PERSPECTIVE = True

        self.EYE = np.array([0.0, 0.0, 2.0])
        self.VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 800.0])
        self.LOOK_AT = np.array([0.0, 0.0, 0.0])
        self.EYE_UP = np.array([0.0, 1.0, 0.0])

        self.DIST, self.PHI, self.THETA = self._get_camera_posture()
        logger.debug('Initialized camera parameters.')
        return 0

    def init_blender(self):
        # Init the blender parameters
        glClearColor(0, 0.1, 0.26, 1.0)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)

        # glEnable(GL_CULL_FACE)
        glDisable(GL_MULTISAMPLE)
        logger.debug('Initialized blender parameters.')
        return 0

    def init_window(self):
        # Init the window parameters
        self.WIN_W, self.WIN_H = 800, 600
        self.WIN_X, self.WIN_Y = 300, 200

        displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
        glutInitDisplayMode(displayMode)
        glutInitWindowSize(self.WIN_W, self.WIN_H)
        glutInitWindowPosition(self.WIN_X, self.WIN_Y)
        logger.debug('Initialized window.')
        return 0

    def bind_controllers(self):
        # Init the controller parameters and bind the functions
        self.SCALE_K = np.array([1.0, 1.0, 1.0])
        self.MOUSE_X, self.MOUSE_Y = 0, 0
        self.LEFT_IS_DOWNED = False

        glutReshapeFunc(self._reshape)
        logger.debug('Bound controller: Reshap Window.')

        glutMouseFunc(self._mouseevent)
        logger.debug('Bound controller: Mouse Click.')

        glutMotionFunc(self._mousemotion)
        logger.debug('Bound controller: Mouse Motion.')

        glutKeyboardFunc(self._keydown)
        logger.debug('Bound controller: Key Down.')

        return 0

    def _get_camera_posture(self):
        # Inner function to get the the posture of the camera
        EYE = self.EYE
        LOOK_AT = self.LOOK_AT
        dist = np.sqrt(np.power((EYE-LOOK_AT), 2).sum())
        if dist > 0:
            phi = np.arcsin((EYE[1]-LOOK_AT[1])/dist)
            theta = np.arcsin((EYE[0]-LOOK_AT[0])/(dist*np.cos(phi)))
        else:
            phi = 0.0
            theta = 0.0
        logger.debug(
            'Controller: Got posture: {}, {}, {}.'.format(dist, phi, theta))
        return dist, phi, theta

    def _reshape(self, width, height):
        # Inner function to handle the window reshape event
        self.WIN_W, self.WIN_H = width, height
        glutPostRedisplay()

    def _mouseevent(self, button, state, x, y):
        # Inner function to handle the mouse click and scroll event
        self.MOUSE_X, self.MOUSE_Y = x, y
        if button == GLUT_LEFT_BUTTON:
            self.LEFT_IS_DOWNED = state == GLUT_DOWN
            logger.debug('Controller: Mouse left button changed: {}'.format(
                self.LEFT_IS_DOWNED))

        elif button == 3:
            self.SCALE_K *= 0.9
            logger.debug(
                'Controller: Changed SCALE_K (x 0.9): {}'.format(self.SCALE_K))
            glutPostRedisplay()

        elif button == 4:
            self.SCALE_K *= 1.1
            logger.debug(
                'Controller: Changed SCALE_K (x 1.1): {}'.format(self.SCALE_K))
            glutPostRedisplay()

    def _mousemotion(self, x, y):
        # Inner function to handle the mouse motion event
        if self.LEFT_IS_DOWNED:
            dx = self.MOUSE_X - x
            dy = y - self.MOUSE_Y
            self.MOUSE_X, self.MOUSE_Y = x, y

            self.PHI += 2*np.pi*dy/self.WIN_H
            self.PHI %= 2*np.pi
            self.THETA += 2*np.pi*dx/self.WIN_W
            self.THETA %= 2*np.pi

            r = self.DIST*np.cos(self.PHI)
            self.EYE[1] = self.DIST*np.sin(self.PHI)
            self.EYE[0] = r*np.sin(self.THETA)
            self.EYE[2] = r*np.cos(self.THETA)

            if 0.5*np.pi < self.PHI < 1.5*np.pi:
                self.EYE_UP[1] = -1.0
            else:
                self.EYE_UP[1] = 1.0

            glutPostRedisplay()
            logger.debug('Controller: Moved mouse.')

    def _keydown(self, key, x, y):
        # Inner function to handle the key down event
        if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
            # Move along the x-, y-, z-axis
            old_LOOK_AT = self.LOOK_AT.copy()
            if key == b'x':
                self.LOOK_AT[0] -= 0.01

            elif key == b'X':
                self.LOOK_AT[0] += 0.01

            elif key == b'y':
                self.LOOK_AT[1] -= 0.01

            elif key == b'Y':
                self.LOOK_AT[1] += 0.01

            elif key == b'z':
                self.LOOK_AT[2] -= 0.01

            elif key == b'Z':
                self.LOOK_AT[2] += 0.01

            self.DIST, self.PHI, self.THETA = self._get_camera_posture()
            glutPostRedisplay()
            logger.debug(
                'Controller: Changed LOOK_AT: {} -> {}'.format(old_LOOK_AT, self.LOOK_AT))

        elif key == b'=':
            # Enter: Zoom In
            self.EYE = self.LOOK_AT + (self.EYE - self.LOOK_AT) * 0.9
            self.DIST, self.PHI, self.THETA = self._get_camera_posture()
            glutPostRedisplay()
            logger.debug('Controller: Zoomed In: {}'.format(self.EYE))

        elif key == b'-':
            # Backspace: Zoom Out
            self.EYE = self.LOOK_AT + (self.EYE - self.LOOK_AT) * 1.1
            self.DIST, self.PHI, self.THETA = self._get_camera_posture()
            glutPostRedisplay()
            logger.debug('Controller: Zoomed Out: {}'.format(self.EYE))

        elif key == b' ':
            # Space: Toggle perspective mode
            # It will also disable zoom function.
            self.IS_PERSPECTIVE = not self.IS_PERSPECTIVE
            glutPostRedisplay()
            logger.debug('Controller: Change IS_PERSPECTIVE: {}'.format(
                self.IS_PERSPECTIVE))
