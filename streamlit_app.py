import streamlit as st
import requests
import cv2
import numpy as np
from config import URL, HEADERS
from config import DATA
from bbox import bbox_convert, draw_bbox

# st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Brain object detection model demo by Trieu Nguyen")
st.write("")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    
    # image = Image.open(uploaded_file)

    # st.write(type(image_byte))
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    
    # Resquest API
    with st.spinner("Model is predicting ..."):
        response = requests.post(URL, headers=HEADERS, data=DATA, files={"image": uploaded_file})
    
    # Draw result
    # with st.spinner("Predict..."):
        # Read the results
    results = response.json()["data"]
    st.write(results)
    bboxes = bbox_convert(results)
    
    # Draw bbox
    image_byte = uploaded_file.getvalue()   # byte like object
    image_np = np.frombuffer(image_byte, np.uint8)  # convert to numpy array for cv2 read
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    draw_bbox(image, bboxes, "bbox.jpg")
    st.write("Object detect result")
    st.image("bbox.jpg")
    st.success("Model runs successfully")
