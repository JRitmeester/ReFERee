import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

from PIL import Image, ImageTk
from datetime import datetime
import shared
import time # For waiting
import threading


class EmotionRecogniser(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
        self.mode = "display"

    def run(self):
        # Create the model
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        # If you want to train the same model or try other models, go for this
        if self.mode == "train":
            train_dir = 'data/train'
            val_dir = 'data/test'

            self.num_train = 28709
            self.num_val = 7178
            self.batch_size = 64
            self.num_epoch = 50

            self.train_datagen = ImageDataGenerator(rescale=1. / 255)
            self.val_datagen = ImageDataGenerator(rescale=1. / 255)

            self.train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(48, 48),
                batch_size=self.batch_size,
                color_mode="grayscale",
                class_mode='categorical')

            self.validation_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(48, 48),
                batch_size=self.batch_size,
                color_mode="grayscale",
                class_mode='categorical')
            self.train()
        elif self.mode == "display":
            self.model.load_weights('model.h5')
            self.display()

    def plot_model_history(self, model_history):

        """
        Plot Accuracy and Loss curves given the model_history
        """

        fig, axs = plt.subplots(1,2,figsize=(15,5))
        # summarize history for accuracy
        axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
        axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
        axs[0].legend(['train', 'val'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
        axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        fig.savefig('plot.png')
        plt.show()

    def train(self):
        self.model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['acc'])
        self.model_info = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.num_train // self.batch_size,
            epochs=self.num_epoch,
            validation_data=self.validation_generator,
            validation_steps=self.num_val // self.batch_size)
        self.plot_model_history(self.model_info)
        self.model.save_weights('model.h5')

    def display(self):
        if shared.system_state == shared.State.LOADING:
            shared.system_state = shared.State.STOPPED
            print('Done loading.')
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

        # start the webcam feed
        cap = cv2.VideoCapture(0)
        while True:
            if shared.system_state == shared.State.ACTIVE: # If Play is pressed in the GUI, continue.

                # Find haar cascade to draw bounding box around face

                ret, frame = cap.read()
                if not ret:
                    break
                facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    if w > 125 and h > 125:
                        roi_gray = gray[y:y + h, x:x + w] # Get just the pixels within the rectangle.
                        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                        # model.predict takes up about 80% of the time it takes for each loop...
                        prediction = self.model.predict(cropped_img) # Contains all the confidence intervals
                        self.set_emotion_data(prediction)
                        self.set_webcam_image(roi_gray)
            elif shared.system_state != shared.State.LOADING:
                time.sleep(0.5)

        cap.release()
        cv2.destroyAllWindows()

    def set_emotion_data(self, prediction):
        shared.current_prediction = prediction[0] # Prediction is a list in a list, for some reason...
        if not any([np.isnan(e) for e in prediction[0]]):
            maxindex = int(np.argmax(prediction))  # Index of strongest emotion for emotion_dict
            shared.current_emotion = shared.emotion_dict[maxindex]  # Emotion as string
            shared.total_predictions[maxindex] += 1  # Increment total for current emotion
            shared.logger.log(datetime.now(), shared.current_emotion) # Write the current emotion to the log.

    def set_webcam_image(self, img):
        image = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=image)
        shared.webcam_image = imgtk


