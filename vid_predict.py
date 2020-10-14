import argparse
import cv2
import numpy as np
from imutils.video import FileVideoStream
from keras.models import model_from_json
from keras.preprocessing import image

# Parse the video file path argument
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='path to input video file')
args = vars(ap.parse_args())

# Loading JSON model
json_file = open('Saved-Models\\model8402.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Loading weights
model.load_weights('Saved-Models\\model8402.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

fvs = FileVideoStream(args['video']).start()

while fvs.more():
    img = fvs.read()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]

        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    resized_img = cv2.resize(img, (1000, 700))
    cv2.imshow('Facial Emotion Recognition', resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fvs.stop()
cv2.destroyAllWindows()
