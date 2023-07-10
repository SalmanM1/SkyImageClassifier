# SkyImageClassifier

SkyImageClassifier is a deep learning project designed to classify images based on the time of day depicted in them. The project leverages TensorFlow, a powerful library for machine learning and neural networks, along with a pre-trained EfficientNet model from TensorFlow Hub. The model is further trained on a custom dataset of images, tagged with labels indicating the time of day they represent. The trained model is then capable of accepting an input image and accurately predicting the time of day based on the visual features of the image.
## How it Works

The project is implemented in Python and uses TensorFlow to build, train and test the model. The model has been trained on a custom dataset which includes images labeled by the time of day they represent (morning, afternoon, evening, night).

EfficientNet, a powerful image classification model, is used as a feature extractor. We append a dense layer to it to perform the final classification. The model is trained with a combination of the Adam optimizer and the Sparse Categorical Crossentropy loss function.

Images are loaded and preprocessed before being fed to the model. The preprocessing includes rescaling the images, resizing them to the required dimensions, and splitting the dataset into training and validation subsets.

The accuracy of the model is evaluated and printed after training.

Finally, the model is used to classify test images, with the results being displayed on the console.

## Instructions for Use

This model uses TensorFlow to build, train, and evaluate the model. The code is organized into a Python class with several methods for different stages of the process, such as loading the data, creating the model, training the model, and predicting with the model. Each method includes a docstring to describe what it does, which should make the code easy to understand and modify if needed.

1. Clone this repository.
2. Replace the placeholder 'Data Set Path Here' in the script with the path to your image dataset.
3. If you want to test the model with your own images, replace 'Put Test Image URL Here' and 'Expected Class Here' with your image URL and expected class, respectively.
4. Run the script with Python.

#Data Set
Download: https://drive.google.com/drive/folders/1GIWGxs1LOrz0RC2kc67xE4uxu5CJsxty?usp=sharing
