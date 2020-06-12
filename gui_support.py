import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

import shared

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True


def play(buttons):
    shared.previous_system_state = shared.system_state
    shared.system_state = shared.State.ACTIVE
    buttons['play'].configure(state=tk.DISABLED)
    buttons['pause'].configure(state=tk.ACTIVE)
    buttons['stop'].configure(state=tk.ACTIVE)
    buttons['record'].configure(state=tk.ACTIVE)
    buttons['openlogs'].configure(state=tk.DISABLED)
    # buttons['settings'].configure(state=tk.DISABLED)


def pause(buttons):
    shared.previous_system_state = shared.system_state
    shared.system_state = shared.State.PAUSED
    buttons['play'].configure(state=tk.ACTIVE)
    buttons['pause'].configure(state=tk.DISABLED)
    buttons['stop'].configure(state=tk.ACTIVE)
    buttons['record'].configure(state=tk.ACTIVE)
    buttons['openlogs'].configure(state=tk.ACTIVE)
    # buttons['settings'].configure(state=tk.DISABLED)


def stop(buttons):
    shared.previous_system_state = shared.system_state
    shared.system_state = shared.State.STOPPED

    if shared.recording:
        shared.logger.stop()

    buttons['record'].configure(relief=tk.RAISED)
    shared.recording = False

    buttons['play'].configure(state=tk.ACTIVE)
    buttons['pause'].configure(state=tk.DISABLED)
    buttons['stop'].configure(state=tk.DISABLED)
    buttons['record'].configure(state=tk.ACTIVE)
    buttons['openlogs'].configure(state=tk.ACTIVE)
    # buttons['settings'].configure(state=tk.ACTIVE)


def record(buttons):
    shared.previous_system_state = shared.system_state
    shared.recording = not shared.recording
    if shared.recording:
        play(buttons)
        shared.logger.start()
        buttons['record'].configure(relief=tk.SUNKEN)
    else:
        shared.logger.stop()
        buttons['record'].configure(relief=tk.RAISED)


def settings():
    pass
