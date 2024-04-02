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


