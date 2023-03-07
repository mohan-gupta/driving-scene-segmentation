# Driving Scene Segmentation
<img src = "artifacts/segment.gif"> <br>

## Performing Segmentation on Indian Roads.

Self-driving cars also known as Autonomous driving cars are key innovation which has revolutionized the automobile industry. Autonomous driving is a complex task which requires precise understanding of the environment. In order to get precise pixel wise information of the driving scenes, semantic segmentation is used.

# Approach

## DataSet
The environment of Indian roads are unconstrained and unconstrained. So, inorder to perform segmentation on the Indian roads. I have used [Indian Driving Dataset(IDD)](http://idd.insaan.iiit.ac.in/). This dataset consist of 16,000 images with 21 classes collected from 182 drive sequences.

## Model
To Perform Segmentation, I have used slightly modified U-Net. I have added BatchNorm after each conv layer and instead of upconv, I have used convtranspose layer.

## Training
I have used JaccardLoss from [segmentation_models_pytorch](https://smp.readthedocs.io/en/latest/index.html) library. Optimizer Used Adam with a Learning rate of 1e-3. Trained for 100 epochs. Dice Score achieved after Training 80.04

# To run the project

```bash
git clone https://github.com/mohan-gupta/driving-scene-segmentation.git  # clone
cd driving-scene-segmentation
pip install -r requirements.txt  # install
cd app
streamlit run streamlit.py  #run
```