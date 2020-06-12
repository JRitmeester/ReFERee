# Adapted from https://python-mss.readthedocs.io/examples.html#part-of-the-screen

import os
import mss
from pathlib import Path


class Screencapture:

    def __init__(self, abs_folder_path):
        # Create the folder if it doesn't exists already.
        if not os.path.isdir(abs_folder_path):
            try:
                Path(abs_folder_path).mkdir(parents=True, exist_ok=True)
                self.path = abs_folder_path
                print('Created ', abs_folder_path)
            except:
                print(abs_folder_path, 'could not be created.')

    def on_exists(self, fname):
        """
        Callback example when we try to overwrite an existing screenshot.
        """

        if os.path.isfile(fname):
            newfile = fname + "(1)"
            print("{} -> {}".format(fname, newfile))
            os.rename(fname, newfile)

    def capture(self, img_name):
        '''Takes a screenshot of monitor 1 and stores it in self.path. Each screenshot takes roughly 350 ms.
        Returns absolute path of image.'''
        try:
            with mss.mss() as sct:
                filename = sct.shot(output="{}\\{}.png".format(self.path, img_name), callback=self.on_exists)
                print(filename)
                return filename
        except AttributeError:  # self.path wasn't set because Screencapture's path is invalid.
            print('Cannot take screenshot because path is invalid.')