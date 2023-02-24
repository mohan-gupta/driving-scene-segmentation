import streamlit as st

import numpy as np
from PIL import Image

import os
import base64

from predict import get_results

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

contnt = '<p style="font-family:sans-serif; color:White;"> Self-driving cars also known as Autonomous driving '\
        "cars are key innovation which has revolutionized the automobile industry. Autonomous driving is a "\
        "complex task which requires precise understanding of the environment. Semantic Segmentation provides "\
        "the pixel wise information about the driving scenes.</p>"\
        '<p style="font-family:sans-serif; color:White;"> We have trained U-Net model on Indian Driving '\
        "Dataset(IDD) with Dice Score = 80.04%</p>"

dir_path = os.path.dirname(__file__)


if __name__ == '__main__':
    add_bg_from_local(os.path.join(dir_path, "../artifacts/Background.jpg"))
    new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Autonomous Driving Scene Parsing</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown(contnt, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a file")

    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            img = Image.open(uploaded_file)

            img = img.resize((720, 480))
            st.image(img)
        
    if st.button("Predict"):
        img_arr = np.array(img)
        pred = get_results(img_arr)

        masked_image = Image.fromarray(pred, 'RGB')
        masked_image = masked_image.resize((720, 480))
        with col2:
            st.image(masked_image)