import json
import requests
from wsgiref import headers
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
import logging
import json
import requests
import pickle

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Function that transforms the image in the required format for the model
def get_test_generator():
    data_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return data_datagen.flow_from_directory(
        "savedimage", target_size=(244, 244), batch_size=int(1)
    )
tab1, tab2 = st.tabs(["Description", "Application"])
with tab1:
    st.title("Thorax Disease Detection")

    st.markdown("### Introduction")
    st.markdown("Hi, I'm Marvin Hoxha, a 3rd-year bachelor student for Computer Engineering at Epoka University.")
    st.markdown("This web page is built as part of my bachelor thesis, where I have trained different deep learning "
                "image classification models for thorax disease detection.")

    st.markdown("The website allows you to check whether an uploaded image has Effusion or Cardiomegaly. "
                "Please follow the instructions below to use the website effectively.")

    st.header("Instructions")
    st.markdown("1. Upload an image of a thorax scan.")
    st.markdown("2. Check the radio button for the disease you want to detect (Effusion or Cardiomegaly).")
    st.markdown("3. Select a model from the dropdown list. Note that for the deployed website, "
                "I have included the better performing models to save storage and reduce complexity.")
    st.markdown("4. Click the 'Predict' button to get the prediction result.")

    st.markdown("Please note that the predictions provided by the models are based on the trained data, "
                "and their accuracy may vary. If you suspect you have a thorax disease, it is important "
                "to consult with a qualified medical professional for proper diagnosis and advice.")
    
    st.markdown("The full code for this website, including Python files for training different models "
                "and a Dockerized version of the website, can be found on "
                "[GitHub](https://github.com/marvinhoxha/Bachelor-Thesis).")
    
    st.markdown("Additionally, the GitHub repository includes additional models trained with more diseases "
                "from the NIH dataset, providing broader detection capabilities.")
    
    st.markdown("All work presented on this website is the result of my own efforts as a student.")

with tab2:
    # Display the header and upload the xray photo
    st.header("**Upload an X-ray photo and press the 'Predict' button to get a prediction!**")

    st.markdown("You can access a folder of testing photos for your convenience by "
                "clicking [here](https://drive.google.com/drive/folders/1H0CD9A7nd-GGCK9caZQNHAYuIsFYurub?usp=sharing).")

    image = st.file_uploader("X-ray photo: ", type=["jpg", "png", "jpeg"], key=1)

    radio_options = ["Effusion","Cardiomegaly"]
    radio = st.radio("Disease",radio_options)
    if radio == "Effusion":
        # Select the deep learning model
        option = st.selectbox('Pick a deep learning model for Effusion:',
                ('ResNet50', 'DenseNet', 'VGG', 'Inception', 'Model'))

        # Determine the model's URL based on the selected option
        if option == 'ResNet50':
            url = "http://tfserve:8501/v1/models/Effusion_model_resnet:predict"
        elif option == 'DenseNet':
            url = "http://tfserve:8501/v1/models/Effusion_model_densenet:predict"
        elif option == 'VGG':
            url = "http://tfserve:8501/v1/models/Effusion_model_vgg:predict"
        elif option == 'Inception':
            url = "http://tfserve:8501/v1/models/Effusion_model_inception:predict"
        elif option == 'Model':
            url = "http://tfserve:8501/v1/models/Effusion_model_self:predict"
    else:
        # Select the deep learning model
        option = st.selectbox('Pick a deep learning model for Cardiomegaly:',
                ('ResNet50', 'DenseNet', 'VGG', 'Inception', 'Model'))

        # Determine the model's URL based on the selected option
        if option == 'ResNet50':
            url = "http://tfserve:8501/v1/models/Cardiomegaly_model_resnet:predict"
        elif option == 'DenseNet':
            url = "http://tfserve:8501/v1/models/Cardiomegaly_model_densenet:predict"
        elif option == 'VGG':
            url = "http://tfserve:8501/v1/models/Cardiomegaly_model_vgg:predict"
        elif option == 'Inception':
            url = "http://tfserve:8501/v1/models/Cardiomegaly_model_inception:predict"
        elif option == 'Model':
            url = "http://tfserve:8501/v1/models/Cardiomegaly_model_self:predict"






    # If an image is uploaded, perform prediction when the Predict button is clicked
    if image is not None:
        with open(os.path.join("savedimage/xray", "xray_image.png"), "wb") as f:
            f.write((image).getbuffer())
        with st.spinner("Loading image..."):
            time.sleep(0.2)
            st.image(image, use_column_width=True)
            predict_button = st.button("Predict", 2)

            # If the Predict button is clicked, transform the image, serve it to the model, and output the prediction
            if predict_button:
                try:
                    test_generator = get_test_generator()
                    image = test_generator.next()[0][0]
                    image = image[None, ...]

                    with st.spinner("Predicting the diagnosis..."):
                        data = {
                            "signature_name": "serving_default",
                            "instances": image.tolist(),
                        }
                        headers = {"Content-Type": "application/json"}
                        response = requests.post(url, json=data, headers=headers)
                        logging.info(response)

                        if response.status_code == 200:
                            prediction = json.loads(response.text)["predictions"]
                            pred = tf.argmax(prediction, axis=1)
                            if radio == "Cardiomegaly":
                                with open("./models/Cardiomegaly_labels.pickle", "rb") as handle:
                                    idx_to_class1 = pickle.load(handle)
                            else:
                                with open("./models/Effusion_labels.pickle", "rb") as handle:
                                    idx_to_class1 = pickle.load(handle)

                            idx_to_class = {value: key for key, value in idx_to_class1.items()}
                            label = idx_to_class[pred.numpy()[0]]
                            result = label.split(".")[-1].replace("_", " ")

                            st.success(f"Diagnosis: {result} :scream:")
                        else:
                            st.error("Prediction failed. Please try again.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                
                # Get all labels and their probabilities
                probs = tf.nn.softmax(tf.convert_to_tensor(prediction))[0]  # Convert prediction to a tensor and apply softmax
                labels = []

                for i in range(len(idx_to_class)):
                    label = idx_to_class[i]
                    label = label.split(".")[-1].replace("_", " ")
                    probability = probs[i].numpy() * 100  # Convert probability to percentage
                    labels.append((label, probability))

                st.write("Probability:")
                for label, probability in labels:
                    st.metric(label=label, value=f"{probability:.2f}%")  # Display probability as percentage with 2 decimal places
