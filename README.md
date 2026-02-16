# FlowerDetection
Flower Detection System 🌼

⚓Overview
 
The Flower Detection Project is a machine learning application designed to classify images of flowers into five specific categories: Daisy, Dandelion, Rose, Sunflower, and Tulip. This Model is capable of accurately identifying different types of flowers from images.


🪶Features

Image Recognition: Upload an image of a flower, and the model will classify it into one of the predefined categories with a confidence score.

Dataset Exploration: Explore the dataset used for training the flower detection model, including sample images and categories

Model Training: Learn about the architecture of the convolutional neural network (CNN) used for training the flower detection model, as well as the training process
 
Deployment: Deploy the trained model for real-world applications, such as integrating it into web or mobile apps for automatic flower recognition.

Multiway Testing: The scripts are capable of recognising a single or multiple flowers at a time
 
Data Augmentation: See how data augmentation techniques such as random flipping, rotation, and zooming are applied to enhance the model's robustness.


🖧 Project Structure
/project-directory
│
├── app.py                  # Streamlit web application
├── train.py                # Model training script
├── requirements.txt        # List of dependencies
├── flower_model.keras      # The trained model (generated after running train.py)
└── flowers/                # Dataset directory (must be structured by class)
    ├── daisy/
    ├── dandelion/
    ├── rose/
    ├── sunflower/
    └── tulip/


⚙️ Requirements
  
The project relies on the following Python libraries :-
tensorflow: For building and training the deep learning model.
streamlit: For the web interface.
numpy: For numerical operations and array manipulation.
pillow: For image processing.
matplotlib: For visualization



📥 Installation


#pip install -r requirements.txt

📀 Module traning

Training Script (train.py) :- 

#python train.py

output :- flower_model.keras


🌐 Web Application

(app.py) This script launches the user interface.

#streamlit run app.py

output:-

Displays the uploaded image.

Shows the top 3 predicted classes with their confidence scores (percentage).

Visualizes confidence using a progress bar.


➡️ Flow of data

Dataset :The dataset used for training the flower detection model consists of images of various flowers, including daisy, dandelion, rose, sunflower, and tulip. It is structured into different directories, each corresponding to a specific flower category.

Prepare Data: Ensure your flowers folder is populated with images.

Train Model: Run python train.py. Wait for the 10 epochs to finish. Ensure flower_model.keras is created.

Start App: Run streamlit run app.py.

Test: Open the local URL provided (usually http://localhost:8501), upload a flower image, and view the results.

🚨 NOTE:-

This model works only in python 3.10.0  version
