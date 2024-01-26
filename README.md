# AI-pipeline
# Image Classifier Project
In this project, we'll be creating a deep learning network to classify flowers per the labels provided. The project utilizes transfer learning to import already trained classifiers from the PyTorch package while modifying the classifier attribute of each package.

## Project Breakdown
The files work through the project in the following manners:
 - **Creating the Datasets**: Utilizing the images provided by Udacity, the first part of the project looks to import the data while applying proper transforms and segmenting them into respective training, validation, and testing datasets
 - **Creating the Architecture**:We used a pre-trained PyTorch neural network classifier, primarily based on the VGG16 architecture, with customizable hyperparameters such as dropout rate and the number of hidden layers. The model is designed to classify images into one of 102 classes. The code includes functions for initializing the classifier, setting up the model architecture, and making it adaptable to various hyperparameter configurations. We define incorporate also a negative log-likelihood loss function and the Adam optimizer.
 - **Training the Model**: The function conducts training over a specified number of epochs, with GPU acceleration, and includes validation at regular intervals. Training metrics, such as training and validation losses, and validation accuracy, are recorded and plotted using Matplotlib.
 - **Saving / Loading the Model**: To practice utilizing the model in other platforms, we export the model to a 'checkpoint.pth' file and re-load / rebuild it in another file.
 - **Class Prediction**: Finally, we use our newly trained model to make a prediction of a flower given a testing input image.

## Files Included
These are the files included as part of the project and what each contains:
 - **Image Classifier Project.ipynb**: This is the Jupyter notebook where I conducted all my activities, including a little more than what is included in the predict.py and train.py files.
 - **Image Classifier Project.html**: Same as the file above, except in HTML form.
 - **train.py**: This file accepts inputs from the command line prompt and takes the work from the Jupyter notebook for the following activities:
  - Creating the Datasets
  - Creating the Architecture
  - Training the model
  - Saving the Model

- **predict.py**: This file accepts inputs from the command line prompt and takes the work from the Jupyter notebook for the following activities
  - Loading the Model
  - Class Prediction
