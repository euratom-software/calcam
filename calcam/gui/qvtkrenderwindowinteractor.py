"""
A simple VTK widget for PyQt or PySide.
See http://www.trolltech.com for Qt documentation,
http://www.riverbankcomputing.co.uk for PyQt, and
http://pyside.github.io for PySide.

This class is based on the vtkGenericRenderWindowInteractor and is
therefore fairly powerful.  It should also play nicely with the
vtk3DWidget code.

Created by Prabhu Ramachandran, May 2002
Based on David Gobbi's QVTKRenderWidget.py

Changes by Gerard Vermeulen Feb. 2003
 Win32 support.

Changes by Gerard Vermeulen, May 2003
 Bug fixes and better integration with the Qt framework.

Changes by Phil Thompson, Nov. 2006
 Ported to PyQt v4.
 Added support for wheel events.

Changes by Phil Thompson, Oct. 2007
 Bug fixes.

Changes by Phil Thompson, Mar. 2008
 Added cursor support.

Changes by Rodrigo Mologni, Sep. 2013 (Credit to Daniele Esposti)
 Bug fix to PySide: Converts PyCObject to void pointer.

Changes by Greg Schussman, Aug. 2014
 The keyPressEvent function now passes keysym instead of None.

Changes by Alex Tsui, Apr. 2015
 Port from PyQt4 to PyQt5.

Changes by Fabian Wenzel, Jan. 2016
 Support for Python3

Changes by Scott Silburn, Jan. 2019
 Change default base class to QGLWidget; falls back to QWidget if not available.
 Also get PyQt version from calcam qt wrapper.

"""
import vtk
from .qt_wrapper import qt_ver
# Check whether a specific PyQt implementation was chosen
try:
    import vtk.qt
    PyQtImpl = vtk.qt.PyQtImpl
except ImportError:
    PyQtImpl = None

if PyQtImpl is None:
    # Check what qt version we're using
    if qt_ver == 6:
        PyQtImpl = "PyQt6"
    elif qt_ver == 5:
        PyQtImpl = "PyQt5"
    elif qt_ver == 4:
        PyQtImpl = "PyQt4"
    else:
        raise ImportError("Unknown PyQt implementation")

if PyQtImpl == "PyQt6":

    try:
        from PyQt6.QtOpenGLWidgets import QOpenGLWidget as QGLWidget
        QVTKRWIBase = "QGLWidget"
    except:
        QVTKRWIBase = "QWidget"

    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtWidgets import QSizePolicy
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtCore import QTimer
    from PyQt6.QtCore import QSize
    from PyQt6.QtCore import QEvent

    Qt.NoButton = Qt.MouseButton.NoButton
    Qt.LeftButton = Qt.MouseButton.LeftButton
    Qt.RightButton = Qt.MouseButton.RightButton
    Qt.MidButton = Qt.MouseButton.MiddleButton
    Qt.NoModifier = Qt.KeyboardModifier.NoModifier
    Qt.ControlModifier = Qt.KeyboardModifier.ControlModifier
    Qt.ShiftModifier = Qt.KeyboardModifier.ShiftModifier
    Qt.AltModifier = Qt.KeyboardModifier.AltModifier
    Qt.WA_OpaquePaintEvent = Qt.WidgetAttribute.WA_OpaquePaintEvent
    Qt.WA_PaintOnScreen = Qt.WidgetAttribute.WA_OpaquePaintEvent
    Qt.WheelFocus = Qt.FocusPolicy.WheelFocus
    Qt.WaitCursor = Qt.CursorShape.WaitCursor
    Qt.MSWindowsOwnDC = Qt.WindowType.MSWindowsOwnDC
    MouseButtonDblClick = Qt.MouseEventFlag.MouseEventCreatedDoubleClick

elif PyQtImpl == "PyQt5":

    try:
        from PyQt5.QtOpenGL import QGLWidget
        QVTKRWIBase = "QGLWidget"
    except:
        QVTKRWIBase = "QWidget"

    from PyQt5.QtWidgets import QWidget
    from PyQt5.QtWidgets import QSizePolicy
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from PyQt5.QtCore import QTimer
    from PyQt5.QtCore import QSize
    from PyQt5.QtCore import QEvent
    MouseButtonDblClick = QEvent.MouseButtonDblClick

elif PyQtImpl == "PyQt4":

    try:
        from PyQt4.QtOpenGL import QGLWidget
        QVTKRWIBase = "QGLWidget"
    except:
        QVTKRWIBase = "QWidget"

    from PyQt4.QtGui import QWidget
    from PyQt4.QtGui import QSizePolicy
    from PyQt4.QtGui import QApplication
    from PyQt4.QtCore import Qt
    from PyQt4.QtCore import QTimer
    from PyQt4.QtCore import QSize
    from PyQt4.QtCore import QEvent

    MouseButtonDblClick = QEvent.MouseButtonDblClick
else:
    raise ImportError("Unknown PyQt implementation " + repr(PyQtImpl))

# Define types for base class, based on string
if QVTKRWIBase == "QWidget":
    QVTKRWIBaseClass = QWidget
elif QVTKRWIBase == "QGLWidget":
    QVTKRWIBaseClass = QGLWidget
else:
    raise ImportError("Unknown base class for QVTKRenderWindowInteractor " + QVTKRWIBase)

class QVTKRenderWindowInteractor(QVTKRWIBaseClass):

    """ A QVTKRenderWindowInteractor for Python and Qt.  Uses a
    vtkGenericRenderWindowInteractor to handle the interactions.  Use
    GetRenderWindow() to get the vtkRenderWindow.  Create with the
    keyword stereo=1 in order to generate a stereo-capable window.

    The user interface is summarized in vtkInteractorStyle.h:

    - Keypress j / Keypress t: toggle between joystick (position
    sensitive) and trackball (motion sensitive) styles. In joystick
    style, motion occurs continuously as long as a mouse button is
    pressed. In trackball style, motion occurs when the mouse button
    is pressed and the mouse pointer moves.

    - Keypress c / Keypress o: toggle between camera and object
    (actor) modes. In camera mode, mouse events affect the camera
    position and focal point. In object mode, mouse events affect
    the actor that is under the mouse pointer.

    - Button 1: rotate the camera around its focal point (if camera
    mode) or rotate the actor around its origin (if actor mode). The
    rotation is in the direction defined from the center of the
    renderer's viewport towards the mouse position. In joystick mode,
    the magnitude of the rotation is determined by the distance the
    mouse is from the center of the render window.

    - Button 2: pan the camera (if camera mode) or translate the actor
    (if object mode). In joystick mode, the direction of pan or
    translation is from the center of the viewport towards the mouse
    position. In trackball mode, the direction of motion is the
    direction the mouse moves. (Note: with 2-button mice, pan is
    defined as <Shift>-Button 1.)

    - Button 3: zoom the camera (if camera mode) or scale the actor
    (if object mode). Zoom in/increase scale if the mouse position is
    in the top half of the viewport; zoom out/decrease scale if the
    mouse position is in the bottom half. In joystick mode, the amount
    of zoom is controlled by the distance of the mouse pointer from
    the horizontal centerline of the window.

    - Keypress 3: toggle the render window into and out of stereo
    mode.  By default, red-blue stereo pairs are created. Some systems
    support Crystal Eyes LCD stereo glasses; you have to invoke
    SetStereoTypeToCrystalEyes() on the rendering window.  Note: to
    use stereo you also need to pass a stereo=1 keyword argument to
    the constructor.

    - Keypress e: exit the application.

    - Keypress f: fly to the picked point

    - Keypress p: perform a pick operation. The render window interactor
    has an internal instance of vtkCellPicker that it uses to pick.

    - Keypress r: reset the camera view along the current view
    direction. Centers the actors and moves the camera so that all actors
    are visible.

    - Keypress s: modify the representation of all actors so that they
    are surfaces.

    - Keypress u: invoke the user-defined function. Typically, this
    keypress will bring up an interactor that you can type commands in.

    - Keypress w: modify the representation of all actors so that they
    are wireframe.
    """

    # Map between VTK and Qt cursors.
    if qt_ver > 5:
        obj = Qt.CursorShape
    else:
        obj = Qt

    _CURSOR_MAP = {
        0:  obj.ArrowCursor,          # VTK_CURSOR_DEFAULT
        1:  obj.ArrowCursor,          # VTK_CURSOR_ARROW
        2:  obj.SizeBDiagCursor,      # VTK_CURSOR_SIZENE
        3:  obj.SizeFDiagCursor,      # VTK_CURSOR_SIZENWSE
        4:  obj.SizeBDiagCursor,      # VTK_CURSOR_SIZESW
        5:  obj.SizeFDiagCursor,      # VTK_CURSOR_SIZESE
        6:  obj.SizeVerCursor,        # VTK_CURSOR_SIZENS
        7:  obj.SizeHorCursor,        # VTK_CURSOR_SIZEWE
        8:  obj.SizeAllCursor,        # VTK_CURSOR_SIZEALL
        9:  obj.PointingHandCursor,   # VTK_CURSOR_HAND
        10: obj.CrossCursor,          # VTK_CURSOR_CROSSHAIR
    }

    def __init__(self, parent=None, **kw):
        # the current button
        self._ActiveButton = Qt.NoButton

        # private attributes
        self.__saveX = 0
        self.__saveY = 0
        self.__saveModifiers = Qt.NoModifier
        self.__saveButtons = Qt.NoButton
        self.__wheelDelta = 0

        # do special handling of some keywords:
        # stereo, rw

        try:
            stereo = bool(kw['stereo'])
        except KeyError:
            stereo = False

        try:
            rw = kw['rw']
        except KeyError:
            rw = None

        # create base qt-level widget
        if QVTKRWIBase == "QWidget":
            if "wflags" in kw:
                wflags = kw['wflags']

            else:
                wflags = 0

            QWidget.__init__(self, parent, wflags | Qt.MSWindowsOwnDC)
        elif QVTKRWIBase == "QGLWidget":
            QGLWidget.__init__(self, parent)

        if rw: # user-supplied render window
            self._RenderWindow = rw
        else:
            self._RenderWindow = vtk.vtkRenderWindow()

        WId = self.winId()

        # Python2
        if type(WId).__name__ == 'PyCObject':
            from ctypes import pythonapi, c_void_p, py_object

            pythonapi.PyCObject_AsVoidPtr.restype  = c_void_p
            pythonapi.PyCObject_AsVoidPtr.argtypes = [py_object]

            WId = pythonapi.PyCObject_AsVoidPtr(WId)

        # Python3
        elif type(WId).__name__ == 'PyCapsule':
            from ctypes import pythonapi, c_void_p, py_object, c_char_p

            pythonapi.PyCapsule_GetName.restype = c_char_p
            pythonapi.PyCapsule_GetName.argtypes = [py_object]

            name = pythonapi.PyCapsule_GetName(WId)

            pythonapi.PyCapsule_GetPointer.restype  = c_void_p
            pythonapi.PyCapsule_GetPointer.argtypes = [py_object, c_char_p]

            WId = pythonapi.PyCapsule_GetPointer(WId, name)

        self._RenderWindow.SetWindowInfo(str(int(WId)))

        if stereo: # stereo mode
            self._RenderWindow.StereoCapableWindowOn()
            self._RenderWindow.SetStereoTypeToCrystalEyes()

        try:
            self._Iren = kw['iren']
        except KeyError:
            self._Iren = vtk.vtkGenericRenderWindowInteractor()
            self._Iren.SetRenderWindow(self._RenderWindow)

        # do all the necessary qt setup
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_PaintOnScreen)
        self.setMouseTracking(True) # get all mouse events
        self.setFocusPolicy(Qt.WheelFocus)
        if qt_ver < 6:
            self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        else:
            self.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))

        self._Timer = QTimer(self)
        self._Timer.timeout.connect(self.TimerEvent)

        self._Iren.AddObserver('CreateTimerEvent', self.CreateTimer)
        self._Iren.AddObserver('DestroyTimerEvent', self.DestroyTimer)
        self._Iren.GetRenderWindow().AddObserver('CursorChangedEvent',
                                                 self.CursorChangedEvent)

        #Create a hidden child widget and connect its destroyed signal to its
        #parent ``Finalize`` slot. The hidden children will be destroyed before
        #its parent thus allowing cleanup of VTK elements.
        self._hidden = QWidget(self)
        self._hidden.hide()
        self._hidden.destroyed.connect(self.Finalize)

    def __getattr__(self, attr):
        """Makes the object behave like a vtkGenericRenderWindowInteractor"""
        if attr == '__vtk__':
            return lambda t=self._Iren: t
        elif hasattr(self._Iren, attr):
            return getattr(self._Iren, attr)
        else:
            raise AttributeError(self.__class__.__name__ +
                  " has no attribute named " + attr)

    def Finalize(self):
        '''
        Call internal cleanup method on VTK objects
        '''
        self._RenderWindow.Finalize()

    def CreateTimer(self, obj, evt):
        self._Timer.start(10)

    def DestroyTimer(self, obj, evt):
        self._Timer.stop()
        return 1

    def TimerEvent(self):
        self._Iren.TimerEvent()

    def CursorChangedEvent(self, obj, evt):
        """Called when the CursorChangedEvent fires on the render window."""
        # This indirection is needed since when the event fires, the current
        # cursor is not yet set so we defer this by which time the current
        # cursor should have been set.
        QTimer.singleShot(0, self.ShowCursor)

    def HideCursor(self):
        """Hides the cursor."""
        self.setCursor(Qt.BlankCursor)

    def ShowCursor(self):
        """Shows the cursor."""
        vtk_cursor = self._Iren.GetRenderWindow().GetCurrentCursor()
        qt_cursor = self._CURSOR_MAP.get(vtk_cursor, Qt.ArrowCursor)
        self.setCursor(qt_cursor)

    def closeEvent(self, evt):
        self.Finalize()

    def sizeHint(self):
        return QSize(400, 400)

    def paintEngine(self):
        return None

    def paintEvent(self, ev):
        self._Iren.Render()

    def resizeEvent(self, ev):
        w = self.width()
        h = self.height()
        vtk.vtkRenderWindow.SetSize(self._RenderWindow, w, h)
        self._Iren.SetSize(w, h)
        self._Iren.ConfigureEvent()
        self.update()

    def _GetCtrlShift(self, ev):
        ctrl = shift = False

        if hasattr(ev, 'modifiers'):
            if ev.modifiers() & Qt.ShiftModifier:
                shift = True
            if ev.modifiers() & Qt.ControlModifier:
                ctrl = True
        else:
            if self.__saveModifiers & Qt.ShiftModifier:
                shift = True
            if self.__saveModifiers & Qt.ControlModifier:
                ctrl = True

        return ctrl, shift

    def enterEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        self._Iren.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                            ctrl, shift, chr(0), 0, None)
        self._Iren.EnterEvent()

    def leaveEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        self._Iren.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                            ctrl, shift, chr(0), 0, None)
        self._Iren.LeaveEvent()

    def mousePressEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        repeat = 0
        if ev.type() == MouseButtonDblClick:
            repeat = 1

        if qt_ver < 6:
            x = ev.x()
            y = ev.y()
        else:
            pos = ev.position()
            x = int(pos.x())
            y = int(pos.y())

        self._Iren.SetEventInformationFlipY(x, y,
                                            ctrl, shift, chr(0), repeat, None)

        self._ActiveButton = ev.button()

        if self._ActiveButton == Qt.LeftButton:
            self._Iren.LeftButtonPressEvent()
        elif self._ActiveButton == Qt.RightButton:
            self._Iren.RightButtonPressEvent()
        elif self._ActiveButton == Qt.MidButton:
            self._Iren.MiddleButtonPressEvent()

    def mouseReleaseEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        if qt_ver < 6:
            x = ev.x()
            y = ev.y()
        else:
            pos = ev.position()
            x = int(pos.x())
            y = int(pos.y())

        self._Iren.SetEventInformationFlipY(x, y,
                                            ctrl, shift, chr(0), 0, None)

        if self._ActiveButton == Qt.LeftButton:
            self._Iren.LeftButtonReleaseEvent()
        elif self._ActiveButton == Qt.RightButton:
            self._Iren.RightButtonReleaseEvent()
        elif self._ActiveButton == Qt.MidButton:
            self._Iren.MiddleButtonReleaseEvent()

    def mouseMoveEvent(self, ev):
        self.__saveModifiers = ev.modifiers()
        self.__saveButtons = ev.buttons()

        if qt_ver < 6:
            x = ev.x()
            y = ev.y()
        else:
            pos = ev.position()
            x = int(pos.x())
            y = int(pos.y())

        self.__saveX = x
        self.__saveY = y

        ctrl, shift = self._GetCtrlShift(ev)
        self._Iren.SetEventInformationFlipY(x, y,
                                            ctrl, shift, chr(0), 0, None)
        self._Iren.MouseMoveEvent()

    def keyPressEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        if ev.key() < 256:
            key = str(ev.text())
        else:
            key = chr(0)

        keySym = _qt_key_to_key_sym(ev.key())
        if keySym is not None and shift and len(keySym) == 1 and keySym.isalpha():
            keySym = keySym.upper()

        self._Iren.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                            ctrl, shift, key, 0, keySym)
        self._Iren.KeyPressEvent()
        self._Iren.CharEvent()

    def keyReleaseEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        if ev.key() < 256:
            key = chr(ev.key())
        else:
            key = chr(0)

        self._Iren.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                            ctrl, shift, key, 0, None)
        self._Iren.KeyReleaseEvent()

    def wheelEvent(self, ev):
        if hasattr(ev, 'delta'):
            self.__wheelDelta += ev.delta()
        else:
            self.__wheelDelta += ev.angleDelta().y()

        if self.__wheelDelta >= 120:
            self._Iren.MouseWheelForwardEvent()
            self.__wheelDelta = 0
        elif self.__wheelDelta <= -120:
            self._Iren.MouseWheelBackwardEvent()
            self.__wheelDelta = 0

    def GetRenderWindow(self):
        return self._RenderWindow

    def Render(self):
        self.update()


def QVTKRenderWidgetConeExample():
    """A simple example that uses the QVTKRenderWindowInteractor class."""

    # every QT app needs an app
    app = QApplication(['QVTKRenderWindowInteractor'])

    # create the widget
    widget = QVTKRenderWindowInteractor()
    widget.Initialize()
    widget.Start()
    # if you dont want the 'q' key to exit comment this.
    widget.AddObserver("ExitEvent", lambda o, e, a=app: a.quit())

    ren = vtk.vtkRenderer()
    widget.GetRenderWindow().AddRenderer(ren)

    cone = vtk.vtkConeSource()
    cone.SetResolution(8)

    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())

    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)

    ren.AddActor(coneActor)

    # show the widget
    widget.show()
    # start event processing
    app.exec()


if qt_ver < 6:
    obj = Qt
else:
    obj = Qt.Key

_keysyms = {
    obj.Key_Backspace: 'BackSpace',
    obj.Key_Tab: 'Tab',
    obj.Key_Backtab: 'Tab',
    # Qt.Key_Clear : 'Clear',
    obj.Key_Return: 'Return',
    obj.Key_Enter: 'Return',
    obj.Key_Shift: 'Shift_L',
    obj.Key_Control: 'Control_L',
    obj.Key_Alt: 'Alt_L',
    obj.Key_Pause: 'Pause',
    obj.Key_CapsLock: 'Caps_Lock',
    obj.Key_Escape: 'Escape',
    obj.Key_Space: 'space',
    # Qt.Key_Prior : 'Prior',
    # Qt.Key_Next : 'Next',
    obj.Key_End: 'End',
    obj.Key_Home: 'Home',
    obj.Key_Left: 'Left',
    obj.Key_Up: 'Up',
    obj.Key_Right: 'Right',
    obj.Key_Down: 'Down',
    obj.Key_SysReq: 'Snapshot',
    obj.Key_Insert: 'Insert',
    obj.Key_Delete: 'Delete',
    obj.Key_Help: 'Help',
    obj.Key_0: '0',
    obj.Key_1: '1',
    obj.Key_2: '2',
    obj.Key_3: '3',
    obj.Key_4: '4',
    obj.Key_5: '5',
    obj.Key_6: '6',
    obj.Key_7: '7',
    obj.Key_8: '8',
    obj.Key_9: '9',
    obj.Key_A: 'a',
    obj.Key_B: 'b',
    obj.Key_C: 'c',
    obj.Key_D: 'd',
    obj.Key_E: 'e',
    obj.Key_F: 'f',
    obj.Key_G: 'g',
    obj.Key_H: 'h',
    obj.Key_I: 'i',
    obj.Key_J: 'j',
    obj.Key_K: 'k',
    obj.Key_L: 'l',
    obj.Key_M: 'm',
    obj.Key_N: 'n',
    obj.Key_O: 'o',
    obj.Key_P: 'p',
    obj.Key_Q: 'q',
    obj.Key_R: 'r',
    obj.Key_S: 's',
    obj.Key_T: 't',
    obj.Key_U: 'u',
    obj.Key_V: 'v',
    obj.Key_W: 'w',
    obj.Key_X: 'x',
    obj.Key_Y: 'y',
    obj.Key_Z: 'z',
    obj.Key_Asterisk: 'asterisk',
    obj.Key_Plus: 'plus',
    obj.Key_Minus: 'minus',
    obj.Key_Period: 'period',
    obj.Key_Slash: 'slash',
    obj.Key_F1: 'F1',
    obj.Key_F2: 'F2',
    obj.Key_F3: 'F3',
    obj.Key_F4: 'F4',
    obj.Key_F5: 'F5',
    obj.Key_F6: 'F6',
    obj.Key_F7: 'F7',
    obj.Key_F8: 'F8',
    obj.Key_F9: 'F9',
    obj.Key_F10: 'F10',
    obj.Key_F11: 'F11',
    obj.Key_F12: 'F12',
    obj.Key_F13: 'F13',
    obj.Key_F14: 'F14',
    obj.Key_F15: 'F15',
    obj.Key_F16: 'F16',
    obj.Key_F17: 'F17',
    obj.Key_F18: 'F18',
    obj.Key_F19: 'F19',
    obj.Key_F20: 'F20',
    obj.Key_F21: 'F21',
    obj.Key_F22: 'F22',
    obj.Key_F23: 'F23',
    obj.Key_F24: 'F24',
    obj.Key_NumLock: 'Num_Lock',
    obj.Key_ScrollLock: 'Scroll_Lock',
    }

def _qt_key_to_key_sym(key):
    """ Convert a Qt key into a vtk keysym.

    This is essentially copied from the c++ implementation in
    GUISupport/Qt/QVTKInteractorAdapter.cxx.
    """

    if key not in _keysyms:
        return None

    return _keysyms[key]


if __name__ == "__main__":
    print(PyQtImpl)
    QVTKRenderWidgetConeExample()
