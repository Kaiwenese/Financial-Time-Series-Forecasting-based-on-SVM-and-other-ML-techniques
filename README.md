# Financial Time Series Forecasting based on SVM and other ML techniques

## Abstract

This project aims to predict the stock price of financial time series using two Machine Learning techniques: Support Vector Machines (SVM) and Long Short-Term Memory (LSTM) networks. And from the prediction, the performance for the models is evaluated. The project is implemented using Jupyter Notebook, a web-based interactive development environment that allows combining live code, equations, visualizations, and explanatory text. There is a python file that is used to store the functions that are used in the main file multiple times and this file will be discussed in the following section. This project has discussed about background reading of different Machine Learning techniques on financial time series forecasting and the stages of designing, processing and analyzing financial time series data using the two Machine Learning algorithms in the final report. This is a technical documentation that describes the project source codes, the dataset chosen, the application of the two ML algorithms and the performance metric used to measure the effectiveness of the models.

## Getting Strated

To run the code in this project, Jupyter Notebook has to be installed on local machine. the following Python libraries also have to be installed:

* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* Statsmodels
* Keras
* TensorFlow

Install the libraries using the command:

    pip install pandas numpy matplotlib seaborn scipy scikit-learn statmodels keras tensorflow
    
There is and additional file called *'utilities.py'* which consists a set of functions that need to be imported into the notebook.

After installing the required libraries, clone this repository and open the Jupyter Notebook file *'Project (without outlier).ipynb'* and *'Project (with outlier).ipynb'* to see the implementation details.

## utilities.py

This file contains different functions that are being utilise in *'Project (without outlier).ipynb'* and *'Project (with outlier).ipynb'*. Some libraries is also being imported in this file such as Matplotlib, Seaborn, NumPy, scikit-learn, keras, math and TensorFlow. The description of each function is shown as below:

* price_plot(data): function to plot the price data by parsing the parameter "data". The figure size is set to be 20x10. The data taken into the plotting includes the first 4 columns which are Open, High, Low and Close in the data. The name of y axis is set to "Price" and the legend is displayed in the font size of 15.
* boxplot(data, col): function to visualize the boxplot. The figure size is set to be 10x10. The visualization contains the 7 subplots which are the boxplots that are all the columns from the dataset and is arranged vertically. The parameters that need to parse in the function is "data" and "col".
* SVM_fit_model(X_train, y_train, X_test, y_test): function that create a SVR model, fit the training data into the model and train it. The training data and testing data are used to make predictions, the Root Mean Square Error is calculated. The parameters for this function include "X_train", "y_train", "X_test" and "y_test". The result is also formulated and returned as values.
* LSTM_fit_model(X_train, y_train, X_test, y_test): function that is similar to the SVM_fit_model function where it is designed to create a LSTM model, fit the training data into the model and train it. The training data and testing data are used to make predictions, the Root Mean Square Error (RMSE) is calculated. The parameters for this function include "X_train", "y_train", "X_test" and "y_test". The result is being calculated and returned.
* root_mean_square(y_train, y_test, pred_train, pred_test): function that calculate the Root Mean Square Error of the prediction and the actual data. It consists the parameter of "y_train", "y_test", "pred_train" and "pred_test".
* visualize(y_train, y_test, pred_train, pred_test, date_train, date_test): function that visualize the actual Open price data with the predicted Open price data using seaborn library. "y_train", "y_test", "pred_train", "pred_test", "date_train" and "date_test" are the parameters for the function.
* best_training_performance(rmse_svm_train_1, rmse_svm_train_2, rmse_lstm_train_1,  rmse_lstm_train_2): function that compares the best training RMSE out of the 2 prediction models with 2 different methods.
* best_testing_performance(rmse_svm_test_1, rmse_svm_test_2, rmse_lstm_test_1,  rmse_lstm_test_2): function that compares the best testing RMSE out of the 2 prediction models with 2 different methods, similar to best_training_performance.

## Dataset

The dataset used in this project is the historical stock price of Facebook obtained from Kaggle. The dataset contains the daily stock prices from 18th May 2012 to 1st October 2021. The dataset is pre-processed to remove missing values, remove duplicated data and normalize the features. The *'Project (without outlier).ipynb'* file consists additional pre-processing step where the outliers are removed. On the other hand, the outliers in *'Project (with outlier).ipynb'* is remained.

## Different Approach

There are two different approaches when dealing with the data before fit into the training models: 

* First Approach (Method 1): first normalize the data, then split the data into training and testing data, and fit the training data to the models.
* Second approach (Method 2): this approach is more related to the real-world application, where the data is first split into training and testing data, then normalize the data, and fit the training data to the models.

## SVM-based forecasting

The SVM-based forecasting model uses the Support Vector Regression (SVR) algorithm from scikit-learn (sklearn) module. The SVR algorithm is trained on the historical stock prices of Facebook from the beginning of the dataset until end of 2018 and the model is used to predict the testing dataset which contains the date from start of 2019 to the end of the dataset. This model applies the two different approaches to perform the forecasting.

## LSTM-based forecasting

The LSTM-based forecasting model uses a sequential neural network with a combination of LSTM layers and Dense Layers from Keras. The neural network is trained on the historical stock prices of Facebook from the beginning of the dataset until end of 2018 and the model is used to predict the testing dataset which contains the date from start of 2019 to the end of the dataset. This model also applies the two different approaches to perform the forecasting.

## Results

The performance of the SVM-based and LSTM-based forecasting models is evaluated using Root Mean Squared Error (RMSE) metric since the orginal data and prediction result are in regression form. The results show that the LSTM-based model outperforms the SVM-based model in terms of forecasting accuracy.

## Conclusion

This project discusses the effectiveness of Machine Learning techniques such as SVM and LSTM in financial time series forecasting. The LSTM-based model shows better performance compared to the SVM-based model in terms of forecasting accuracy for both the approaches. The code in this project can be adapted to other financial time series datasets to predict future stock prices or other financial metrices.
