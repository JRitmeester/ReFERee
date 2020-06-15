try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

from gui import GUI
from emotions import EmotionRecogniser
import myLogger
import shared


if __name__ == "__main__":
    print('Starting...')
    shared.init()
    shared.logger = myLogger.Logger()
    gui = GUI()
    emrec = EmotionRecogniser()




