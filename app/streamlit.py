import streamlit as st

import numpy as np
from PIL import Image
import cv2
import moviepy.editor as moviepy

import os
import base64
import tempfile

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
        "complex task which requires precise understanding of the environment. In order to get precise pixel wise "\
        "information of the driving scenes, semantic segmentation is used. To perform semantic segmentation we "\
        "have used U-Net model with slight modifications.</p>"\
        '<p style="font-family:sans-serif; color:White;"> We have trained U-Net model on Indian Driving '\
        "Dataset(IDD) with Dice Score = 80.04%</p>"

dir_path = os.path.dirname(__file__)


if __name__ == '__main__':
    add_bg_from_local(os.path.join(dir_path, "../artifacts/Background.jpg"))
    new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Autonomous Driving Scene Parsing</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown(contnt, unsafe_allow_html=True)

    choice = st.selectbox(label="Do you want to upload an image or video?",
                          options=("Image", "Video"))

    uploaded_file = st.file_uploader("Choose a file")
    
    if choice == "Image":
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
    
    else:
        if uploaded_file is not None:
            output_frames = []

            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())

            vf = cv2.VideoCapture(tfile.name)

            while vf.isOpened():
                ret, frame = vf.read()  # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                
                pred = get_results(frame)

                masked_image = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

                output_frames.append(masked_image)

            output_file = tempfile.NamedTemporaryFile(suffix='.avi', delete=False, dir=dir_path)

            height, width, _ = output_frames[0].shape

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            video = cv2.VideoWriter(output_file.name, fourcc, 1, (width, height))

            for frame in output_frames:
                video.write(frame)

            video.release()

            clip = moviepy.VideoFileClip(output_file.name)
            mp4_file = output_file.name[:-3]+"mp4"
            print(mp4_file)

            clip.write_videofile(mp4_file)

            st.video(mp4_file)