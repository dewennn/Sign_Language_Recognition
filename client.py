import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model import inverse_transform_fn, return_histogram, model, orb

st.header('Alphabet Sign Language Recognition👋')

# Define the navbar options
tabs = ["Explanation", "BoF Model", "CNN Model"]

# Create a variable to store the active tab
active_tab = st.selectbox("Choose another page 😊", tabs)

if active_tab == 'Explanation':
  st.markdown(
  '''
    <h4>Crafted 2 model implementing</h4>
    <ol>
      <li>
        <h5>Bag Of Feature</h5>
        <p>Implemented using orb to find each image local features (edge -> keypoints + descriptor). Then we build a vocabulary using K-Means (k: 1000) since our dataset contains 87.000 image. Count each image histogram value (map each descriptor using K-Means to get the visual-word then count them). Finally feed that histogram and their label to the ANN. <br><br> <b>Final Result: Training Accuracy: 81% | Validation Accuracy: 83% | Test Accuracy: 70%</b></p>
      </li>

      <br>

      <li>
        <h5>CNN</h5>
        <p>Straight up use CNN to process the image feature. <br><br> <b>Final Result: Training Accuracy: 81% | Validation Accuracy: 83% | Test Accuracy: 70%</b></p>
      </li>
    </ol>
  ''', unsafe_allow_html=True)

elif active_tab == 'BoF Model':
  st.subheader('Bag of Feature Model')

  # Initialize camera
  cap = cv2.VideoCapture(0)

  # Run a while loop in Streamlit to capture and process frames
  stframe = st.empty()  # Placeholder for the video frame

  while cap.isOpened():
    ret, frame = cap.read()

    if not ret or active_tab != 'BoF Model':
        st.write("Stopped the video feed.")
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (200, 200))

    # Apply Gaussian Blur and ORB keypoints
    temp = cv2.GaussianBlur(frame, (7, 7), 0)
    orb_keypoints, _ = orb.detectAndCompute(temp, None)

    # Dummy prediction
    prediction = model.predict(return_histogram(resized_frame))
    predicted_class = inverse_transform_fn(prediction)

    # Draw ORB keypoints
    frame_with_keypoints = cv2.drawKeypoints(frame, orb_keypoints, None, color=(0, 0, 255))

    # Overlay the prediction
    cv2.putText(frame_with_keypoints, f"Prediction: {predicted_class}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert BGR to RGB (Streamlit expects RGB format)
    frame_rgb = cv2.cvtColor(frame_with_keypoints, cv2.COLOR_BGR2RGB)

    # Display the frame in Streamlit
    stframe.image(frame_rgb, channels="RGB")

  cap.release()

elif active_tab == 'CNN Model':
  st.subheader('CNN Model')