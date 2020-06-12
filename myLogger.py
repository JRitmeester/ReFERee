import csv
from math import floor
from datetime import datetime
import shared
import numpy as np
from screencapture import Screencapture
import os


class Logger():

    start_time = 0

    def log(self, now, emotion):
        if shared.recording:
            self.write(now, emotion)
            print('Writing {}, {}'.format(now, emotion))

    def start(self):  # Nested function. Can only be accessed from within logger.log().
        print('Starting')
        self.start_time = datetime.now()
        filename = 'logs\\{}'.format(self.generate_filename())
        self.logfile = open(filename, 'a+', newline='')
        self.csv_writer = csv.writer(self.logfile, delimiter=',')
        root_dir = os.path.dirname(os.path.realpath(__file__)) # Root folder of the project (skips venv folder somehow).
        image_folder_path = os.path.join(root_dir, filename)[:-4] # Take off '.csv'
        self.sc = Screencapture(image_folder_path)

    def stop(self):  # Nested function. Can only be accessed from within logger.log().
        # Write the emotion totals as the second to last line.
        print('Stopping...')
        self.csv_writer.writerow(shared.total_predictions)
        # Write date, start time, and duration as the last line.
        date = datetime.now().strftime('%d/%m/%y')
        time = self.start_time.strftime('%H:%M:%S')
        seconds = (datetime.now() - self.start_time).total_seconds()
        duration = self.seconds_to_HMS(seconds)
        self.start_time = 0 # Clean slate.
        self.csv_writer.writerow([date, time, duration])
        self.logfile.flush()
        self.logfile.close()

    def write(self, now, emotion):  # Nested function. Can only be accessed from within logger.log().
        time = round((now - self.start_time).total_seconds(),2) # Two decimals is plenty of detail.
        self.csv_writer.writerow([time, emotion])
        self.logfile.flush()  # Force the buffer to be written now, rather than later.
        self.sc.capture(time)
        print('Logged {}'.format([time, emotion]))

    def generate_filename(self):
        now = datetime.now()
        date = now.strftime('%d%m%y')
        time = now.strftime('%H%M%S')
        return 'session_{}_{}.csv'.format(date, time)

    def read(self, filename):
        # Each line is formatted like: <time in seconds>,<emotion>.
        # Copy the CSV content to a list so that the file can be closed after everything is read.
        lines = []
        line_count = 0
        with open('logs/{}'.format(filename)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # line_count = len(list(csv_reader))
            for row in csv_reader:
                # lines.append(", ".join(row))
                lines.append(row)
                line_count += 1

        content = None
        try:
            emotion_totals = np.array(lines[-2]).astype(np.float)
            date, time, duration = lines[-1]
            content = {'date': date,
                       'time': time,
                       'line_count': line_count-2,
                       'duration': duration,
                       'totals': emotion_totals}

            data_as_array = np.array(lines[:-2]) # Get the rest of the data.
            data_T = data_as_array.T # Transpose, so that all timestamps and all emotions are grouped in lists
            timestamps = data_T[0].astype(np.float) # All timestamps in floats
            labels = data_T[1] # All emotion labels
            # Now you have access to all data in different forms!
            content['data']= {'paired': list(zip(timestamps, labels)),
                              'timestamps': timestamps,
                              'labels': labels}

            return content
        except:  # Log file is corrupt, probably. Could be many different errors, so just grab them all I guess...
            print(filename, ' is corrupted. Repair it or delete it.')
            return None

    def seconds_to_HMS(self, seconds):
        hour = floor(seconds / 3600.0)
        mins = floor(seconds / 60.0)
        secs = floor(seconds % 60)

        hour_str = str(hour) if hour >= 10 else '0'+str(hour)
        mins_str = str(mins) if mins >= 10 else '0'+str(mins)
        secs_str = str(secs) if secs >= 10 else '0'+str(secs)

        duration = ':'.join([hour_str, mins_str, secs_str])
        return duration
