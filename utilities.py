# Importing neccesary libraries/modules
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import math
import tensorflow as tf

# Define a random fixed state
tf.random.set_seed(2)

# Function for plotting the price data
def price_plot(data):
    fig, ax = plt.subplots(figsize = (20,10))
    for name in data.columns[0:4]:
        ax = sns.lineplot(x=data.index, y=name, data=data, label=name)
    plt.ylabel("Price")
    plt.legend(fontsize = 15)
    plt.show()

# Function for outlier checking using boxplot
def boxplot(data, col):
    fig, ax = plt.subplots(7,1,figsize=(10,10))
    feat = data.columns[:]
    for name, ax in zip(feat, ax.flatten()):
        sns.boxplot(x=name, data=data,ax=ax,color=col)
    plt.tight_layout()

# Function for creating SVM model, fitting the model and calculate the result
def SVM_fit_model(X_train, y_train, X_test, y_test):
    # Create a SVR model
    SVR_model = SVR()

    # Fitting the training data into the model to train it
    SVR_model.fit(X_train, y_train)

    # Use model to predict the outcome on training data
    SVM_pred_train = SVR_model.predict(X_train)

    # Use model to predict the outcome on testing data
    SVM_pred_test = SVR_model.predict(X_test)

    # Calculate the root mean square
    rmse_train, rmse_test = root_mean_square(y_train, y_test, SVM_pred_train, SVM_pred_test)
    
    return SVM_pred_train, SVM_pred_test, rmse_train, rmse_test

# Function for creating LSTM model, fitting the model and calculate the result
def LSTM_fit_model(X_train, y_train, X_test, y_test):
    # Create a LSTM Model
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(64,input_shape=(X_train.shape[1],1),activation='relu',return_sequences=True))
    LSTM_model.add(LSTM(32,activation='relu',return_sequences=False))
    LSTM_model.add(Dense(16))
    LSTM_model.add(Dense(1))

    # Compile the model
    LSTM_model.compile(optimizer='adam',loss='mse')
    
    # Define number of batch per training and the training frequency
    batch_size = 50
    epochs = 50
    
    # Fit the training data to train the model
    LSTM_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    # Use model to predict the outcome on training data
    LSTM_pred_train = LSTM_model.predict(X_train, verbose=0)
    
    # Use model to predict the outcome on testing data
    LSTM_pred_test = LSTM_model.predict(X_test, verbose=0)
    
    # Calculate the root mean square
    rmse_train, rmse_test = root_mean_square(y_train, y_test, LSTM_pred_train, LSTM_pred_test)
    
    return LSTM_pred_train, LSTM_pred_test, rmse_train, rmse_test
    
# Function for calculating the Root Mean Square Error
def root_mean_square(y_train, y_test, pred_train, pred_test):
    # Calculate the Root Mean Square Score for the model prediction
    rmse_train = math.sqrt(mean_squared_error(y_train, pred_train))
    rmse_test = math.sqrt(mean_squared_error(y_test, pred_test))
    print("Root Mean Square Error on training data:  ", rmse_train)
    print("Root Mean Square Error on testing data:  ", rmse_test)
    return rmse_train, rmse_test
    
# Function for visualizing the graph after prediction
def visualize(y_train, y_test, pred_train, pred_test, date_train, date_test):
    # Plot data
    Open_train = y_train
    Open_val = y_test
    Open_pred_train, Open_pred_test = pred_train.reshape(-1), pred_test.reshape(-1)

    # Visualise the data
    plt.figure(figsize=(20,12))
    plt.title('Model Prediction',fontsize=25)
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Open', fontsize = 18)
    sns.lineplot(x = date_train, y=Open_train)
    sns.lineplot(x = date_test, y=Open_val)
    sns.lineplot(x = date_train, y=Open_pred_train)
    sns.lineplot(x = date_test, y=Open_pred_test)
    plt.legend(['Training', 'Validation', 'Predictions (Training)', 'Predictions (Testing)'], loc = 'lower right')
    plt.show()

# Function to compare the best training result
def best_training_performance(rmse_svm_train_1, rmse_svm_train_2, rmse_lstm_train_1,  rmse_lstm_train_2):
    # Find the minimum value of Root Mean Square Error for the prediction on training data
    best = min([rmse_svm_train_1, rmse_svm_train_2, rmse_lstm_train_1, rmse_lstm_train_2])
    if (best == rmse_svm_train_1):
        name, method = "SVM", 1
    elif (best == rmse_svm_train_2):
        name, method = "SVM", 2
    elif (best == rmse_lstm_train_1):
        name, method = "LSTM", 1
    elif (best == rmse_lstm_train_2):
        name, method = "LSTM", 2

    return best, name, method

# Function to compare the best testing result
def best_testing_performance(rmse_svm_test_1, rmse_svm_test_2, rmse_lstm_test_1,  rmse_lstm_test_2):
    # Find the minimum value of Root Mean Square Error for the prediction on testing data
    best = min([rmse_svm_test_1, rmse_svm_test_2, rmse_lstm_test_1,  rmse_lstm_test_2])
    if (best == rmse_svm_test_1):
        name, method = "SVM", 1
    elif (best == rmse_svm_test_2):
        name, method = "SVM", 2
    elif (best == rmse_lstm_test_1):
        name, method = "LSTM", 1
    elif (best == rmse_lstm_test_2):
        name, method = "LSTM", 2

    return best, name, method
