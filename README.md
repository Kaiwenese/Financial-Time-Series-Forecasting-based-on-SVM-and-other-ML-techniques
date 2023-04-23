# Financial Time Series Forecasting based on SVM and other ML techniques

## Abstract

This project aims to predict the stock price of financial time series using two Machine Learning techniques: Support Vector Machines (SVM) and Long Short-Term Memory (LSTM) networks. The project is implemented using Jupyter Notebook, a web-based interactive development environment that allows combining live code, equations, visualizations, and explanatory text. There is a python file that is used to store the functions that are used in the main file multiple times and this file will be discussed in the following section. This project has discussed about background reading of different Machine Learning techniques on financial time series forecasting and the stages of designing, processing and analyzing financial time series data using the two Machine Learning algorithms in the final report. This is a technical documentation that describes the project source codes, the dataset chosen, the application of the two ML algorithms and the performance metric used to measure the effectiveness of the models.

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
    
There is and additional file called utilities.py which consists a set of functions that need to be imported into the notebook.

After installing the required libraries, clone this repository and open the Jupyter Notebook file *'Project (without outlier).ipynb'* and *'Project (with outlier).ipynb'* to see the implementation details.

## Dataset

The dataset used in this project is the historical stock price of Facebook obtained from Kaggle. The dataset contains the daily stock prices from 18th May 2012 to 1st October 2021. The dataset is pre-processed to remove missing values, remove duplicated data and normalize the features. The *'Project (without outlier).ipynb'* file consists additional pre-processing step where the outliers are removed. On the other hand, the outliers in *'Project (with outlier).ipynb'* is remained.

## Different Approach

There are two different approaches when dealing with the data before fit into the training models: 

* First Approach: first normalize the data, then split the data into training and testing data, and fit the training data to the models.
* Second approach: this approach is more related to the real-world application, where the data is first split into training and testing data, then normalize the data, and fit the training data to the models.

## SVM-based forecasting

The SVM-based forecasting model uses the Support Vector Regression (SVR) algorithm from scikit-learn (sklearn) module. The SVR algorithm is trained on the historical stock prices of Facebook from the beginning of the dataset until end of 2018 and the model is used to predict the testing dataset which contains the date from start of 2019 to the end of the dataset. This model applies the two different approaches to perform the forecasting

## LSTM-based forecasting

The LSTM-based forecasting model uses a sequential neural network with a combination of LSTM layers and Dense Layers from Keras. The neural network is trained on the historical stock prices of Facebook from the beginning of the dataset until end of 2018 and the model is used to predict the testing dataset which contains the date from start of 2019 to the end of the dataset. This model also applies the two different approaches to perform the forecasting

## Results

The performance of the SVM-based and LSTM-based forecasting models is evaluated using Root Mean Squared Error (RMSE) metrics. The results show that the LSTM-based model outperforms the SVM-based model in terms of forecasting accuracy.

## Conclusion

This project discusses the effectiveness of Machine Learning techniques such as SVM and LSTM in financial time series forecasting. The LSTM-based model shows better performance compared to the SVM-based model in terms of forecasting accuracy for both the approaches. The code in this project can be adapted to other financial time series datasets to predict future stock prices or other financial metrices.
