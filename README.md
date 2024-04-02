# deep-learning-challenge
Deep-learning Challenge related files are in the repository

# Solution
In the deep-learning challenge, there is a Report and there is one folder called Starter_Code in which you will find the two file with the code for original model and optimized model. and analysis report and the Resources folder. 

## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

## Instructions

### Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.

Determine the number of unique values for each column.

For columns that have more than 10 unique values, determine the number of data points for each unique value.

Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Create the first hidden layer and choose an appropriate activation function.

If necessary, add a second hidden layer with an appropriate activation function.

Create an output layer with an appropriate activation function.

Check the structure of the model.

Compile and train the model.

Create a callback that saves the model's weights every five epochs.

Evaluate the model using the test data to determine the loss and accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

### Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.

### Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?
What variable(s) are the features for your model?
What variable(s) should be removed from the input data because they are neither targets nor features?
Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

### Step 5: Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.

Download your Colab notebooks to your computer.

Move them into your Deep Learning Challenge directory in your local repository.

Push the added files to GitHub.

## Analysis Report

# Module 21 Report

## Overview of the Analysis

The purpose of this analysis is to develop a deep learning model to predict the success of charitable organizations based on various features such as application type, affiliation, classification, use case, organization type, and other factors. The model aims to assist in identifying factors that contribute to the success of fundraising efforts for charitable organizations.

## Results

Data Preprocessing

- What variable(s) are the target(s) for your model?

Target Variable(s) Original & Optimized Data:
The target variable for our model is IS_SUCCESSFUL, indicating whether the charitable organization was successful in achieving its fundraising goals.

- What variable(s) are the features for your model?

Original Data:
The feature variables include APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT.

Optimized Data:
The feature variables include APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, and, INCOME_AMT.

- What variable(s) should be removed from the input data because they are neither targets nor features?

Original Data:
The non-beneficial ID columns, namely EIN and NAME, were removed from the dataset as they do not contribute to the model's predictive power.

Optimized Data:
In the optimized dataset, additional columns such as SPECIAL_CONSIDERATIONS, STATUS, and ASK_AMT were also removed as they were deemed non-beneficial.

Compiling, Training, and Evaluating the Model

- How many neurons, layers, and activation functions did you select for your neural network model, and why?

Neural Network Architecture Original Data:
The neural network model consists of three layers:
Input layer with 80 neurons and ReLU activation function
Hidden layer with 30 neurons and ReLU activation function
Output layer with 1 neuron and Sigmoid activation function

Neural Network Architecture Optimized Data:
For the optimized model to get the highest level of accuracy, I played around with the epoch, hidden layers, activation function and so on. Despite multiple tries, I was able to get accuracy for 72.66%.
The neural network model consists of four layers:
Input layer with 15 neurons and ReLU activation function
Hidden layer with 10 neurons and Sigmoid activation function
Hidden layer with 8 neurons and Sigmoid activation function
Output layer with 1 neuron and Sigmoid activation function

Compilation:
The model was compiled using binary cross-entropy loss function and the Adam optimizer.

Training:
The model was trained with the scaled training data over 100 epochs for original data set and 70 epochs for optimized model.

Original Model Evaluation:
The model achieved an accuracy of approximately 72.45% on the test data, with a loss of 0.5604.

Optimized Model Evaluation:
The model achieved an accuracy of approximately 72.66% on the test data, with a loss of 0.5516.

- Were you able to achieve the target model performance?
The model's performance did not reach the target level.

- What steps did you take in your attempts to increase model performance?
Ways to improve model performance include adjusting the architecture of the neural network, tuning hyperparameters, exploring different activation functions, increasing the dataset size, or utilizing advanced techniques such as dropout or regularization.

## Summary

In summary, both the original and optimized deep learning models showed moderate levels of accuracy in predicting the success of charitable organizations. However, there is room for improvement in performance. A recommendation for improving this classification problem is to explore different neural network architectures to capture complex patterns and relationships within the data more effectively.