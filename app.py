import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );
}

[data-testid="stHeader"]{
    display:none;
}

[data-testid="stToolbar"]{
    display:none;
}

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

.block-container{
    padding-top:0rem;
}

.title{
    text-align:center;
    color:white;
    font-size:50px;
}

.result-box{
    height:350px;

    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;

    border-radius:25px;

    background:linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    color:white;

    box-shadow:0 10px 30px rgba(0,0,0,.35);
}

.footer{
    text-align:center;
    margin-top:30px;
    color:#cccccc;
}

[data-testid="stImage"] img{
    border-radius:20px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_file():
    return load_model("models/emotion_model.keras")

model = load_model_file()

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

emotion_emoji = {
    "Angry":"😠",
    "Disgust":"🤢",
    "Fear":"😨",
    "Happy":"😊",
    "Neutral":"😐",
    "Sad":"😭",
    "Surprise":"😲"
}

st.sidebar.title("ℹ️ About")

st.sidebar.info("""
Deep Learning based Emotion Detection System.

Model: CNN trained on FER-2013 Dataset.

Developed using:

• TensorFlow

• OpenCV

• Streamlit
""")

st.markdown(
    """
    <h1 class='title'>
    🧠 Emotion Detection AI
    </h1>
    """,
    unsafe_allow_html=True
)

st.write(
    "Upload a face image and let AI predict the emotion."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image_np = np.array(image)

    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )

    face_detector = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:

        st.error(
            "No face detected. Please upload a clear face image."
        )

    else:

        x, y, w, h = faces[0]

        roi = gray[y:y+h, x:x+w]

        roi = cv2.resize(
            roi,
            (48, 48)
        )

        roi = roi.astype("float32") / 255.0

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

        emotion_index = np.argmax(
            prediction
        )

        emotion = emotion_labels[
            emotion_index
        ]

        confidence = (
            np.max(prediction) * 100
        )

        col1, col2 = st.columns(
            [4, 5],
            gap="large"
        )

        with col1:

            st.image(
                image,
                use_container_width=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="result-box">
                    <div style="font-size:75px;">{emotion_emoji[emotion]}</div>
                    <div style="font-size:45px;font-weight:bold;">
                        {emotion}
                    </div>
                    <div style="font-size:30px;margin-top:15px;">
                        {confidence:.2f}% Confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📊 Emotion Probabilities")

        for i, label in enumerate(
            emotion_labels
        ):

            value = float(
                prediction[0][i]
            )

            st.write(
                f"{label}: {value*100:.2f}%"
            )

            st.progress(value)

st.markdown("""
<div class="footer">
Developed by Ayaz | B.Tech CSE | JMI
</div>
""",
unsafe_allow_html=True)