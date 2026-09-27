import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/emotion_model.keras")

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        roi = gray[y:y+h, x:x+w]

        roi = cv2.resize(
            roi,
            (48,48)
        )

        roi = roi.astype("float") / 255.0

        roi = np.expand_dims(
            roi,
            axis=0
        )

        roi = np.expand_dims(
            roi,
            axis=-1
        )

        prediction = model.predict(
            roi,
            verbose=0
        )

        emotion_index = np.argmax(prediction)

        emotion = emotion_labels[
            emotion_index
        ]

        confidence = np.max(
            prediction
        ) * 100

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        text = f"{emotion} {confidence:.1f}%"

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow(
        "Real Time Emotion Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()