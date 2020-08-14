from fastai.vision import open_image, load_learner, image, torch
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import PIL.Image
import requests
from io import BytesIO
import time

# app title
st.title("Watch Brand Classifier")

# predicting
def predict(img, display_img):

    # display test image
    st.image(display_img, use_column_width=True)

    # temporary message
    with st.spinner('Thinking...'):
        time.sleep(3)

    # load trained model
    model = load_learner('models/', 'WatchProj_classifier.pkl')

    # prediction
    pred_class = model.predict(img)[0]

    # confidence percentage
    pred_prob = round(torch.max(model.predict(img)[2]).item()*100)

    if str(pred_class) == 'rolex':
        st.success("I am " + str(pred_prob) + '%' " confident this is a Rolex.")

    if str(pred_class) == 'audemarspiguet':
        st.success("I am " + str(pred_prob) + '%' " confident this is an Audemars Piguet.")

    if str(pred_class) == 'breitling':
        st.success("I am " + str(pred_prob) + '%' " confident this is an Breitling.")

    if str(pred_class) == 'iwc':
        st.success("I am " + str(pred_prob) + '%' " confident this is an IWC.")

    if str(pred_class) == 'jaegerlecoultre':
        st.success("I am " + str(pred_prob) + '%' " confident this is a Jaegerlecoultre.")

    if str(pred_class) == 'omega':
        st.success("I am " + str(pred_prob) + '%' " confident this is an Omega.")

    if str(pred_class) == 'panerai':
        st.success("I am " + str(pred_prob) + '%' " confident this is a Panerai.")

    if str(pred_class) == 'patekphilippe':
        st.success("I am " + str(pred_prob) + '%' " confident this is a Patek Philippe.")

    if str(pred_class) == 'cartier':
        st.success("I am " + str(pred_prob) + '%' " confident this is a Cartier.")

    if str(pred_class) == 'gucci':
        st.success("I am " + str(pred_prob) + '%' " confident this is a Gucci.")

    if str(pred_class) == 'seiko':
        st.success("I am " + str(pred_prob) + '%' " confident this is a Seiko.")

    if str(pred_class) == 'movado':
        st.success("I am " + str(pred_prob) + '%' " confident this is a Movado.")

    if str(pred_class) == 'zenith':
        st.success("I am " + str(pred_prob) + '%' " confident this is an Zenith.")

# select source of input image
option = st.radio('', ['Choose a test image', 'Choose your own image'])

if option == 'Choose a test image':

    # test image selection
    test_images = os.listdir('scraper/test_images/')
    test_image = st.selectbox(
    'Please select a test image:', test_images)

    # read test image
    file_path = 'scraper/test_images/' + test_image
    img = open_image(file_path)

    # display test image
    display_img = mpimg.imread(file_path)

    # predict and display test image
    predict(img, display_img)

else:

    # loading bar (cosmetic)
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):

        # update progress bar w/ each iteration
        latest_iteration.text(f'Loading... {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.01)

    'Ready!'

    url = st.text_input("Please input a url:")

    if url != "":
        try:
            # read image from the url
            response = requests.get(url)
            pil_img = PIL.Image.open(BytesIO(response.content))
            display_img = np.asarray(pil_img) # image to display

            # transform image to feed into model
            img = pil_img.convert('RGB')
            img = image.pil2tensor(img, np.float32).div_(255)
            img = image.Image(img)

            # predict and display image
            predict(img, display_img)

        except:
            st.text("Invalid url!")
