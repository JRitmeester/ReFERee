from enum import Enum
import numpy as np


# General
class State(Enum):
    LOADING = 0
    STOPPED = 1
    ACTIVE = 2
    PAUSED = 3


system_state = State.LOADING
previous_system_state = State.LOADING

# GUI
window_root = None
webcam_image = None
overlay_visible = True

# Emotions
emotion_names = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
emotion_dict = {i: a for i, a in enumerate(emotion_names)}  # {0: 'Angry', 1: 'Disgusted'...}
all_colors = ['red', 'brown', 'black', 'yellow', 'gray', 'blue', 'purple']
colors_dict = {e: c for e, c in zip(emotion_names, all_colors)}  # {'Angry': 'red', 'Disgusted': 'brown'...}
hsl = {'Happy': 'hsl(60,93%,52%)',
       'Angry': 'hsl(360,79%,52%)',
       'Sad': 'hsl(228,100%,29%)',
       'Surprised': 'hsl(280,61%,48%)',
       'Fearful': 'hsl(0,0%,0%)',
       'Disgusted': 'hsl(30,52%,25%)',
       'Neutral': 'hsl(0,0%,40%)'}

rgb = {'Happy': '#ffff00',
       'Angry': '#ff0000',
       'Sad': '#0000ff',
       'Surprised': '#aa00ff',
       'Fearful': '#000000',
       'Disgusted': '#552200',
       'Neutral': '#808080'}


def init():
    global current_emotion, current_prediction, total_predictions
    current_emotion = None
    current_prediction = np.zeros(7)  # Initialise with 7 zeroes.
    total_predictions = np.zeros(7)


def reset():
    init()


# Logger
logger = None
recording = False
prev_recording = False
