# Thesis-work
Thesis Title - Masked Prediction on time series data using Transformer based models

Our thesis was to predict missing values of signals received from trucks' battery modules. The objective of this project is to create a model that performs a masked prediction of time series data. After the insights obtained from the literature review and EDA on the signals received from the trucks, it was decided to build a transformer-based model for predicting the significant features. The initial goal for the thesis project is to predict missing values for only one feature and obtain very high accuracy on that. The scope of the thesis in the future would involve modifying the model to predict all the significant features of the dataset. In addition, it will forecast the features, whose duration will be decided in the future.

Methodology followed to approach the problem in hand:

## Methodology Overview

### 3.1 Introduction
The methodology section outlines the techniques and processes employed for data visualization, pre-processing, model architecture, and evaluation metrics. This aims to provide a comprehensive understanding of the research approach.

### 3.2 Data Visualization
Visualization plays a crucial role in understanding time-series data trends. Techniques such as Plotly graphs are utilized to visualize features across time, ensuring insights into forecast predictions.

### 3.3 Plotly Graphs
Utilized Plotly in Python to visualize various data features over time. The graphs provide real-time insights, allowing users to focus on specific features within the dataset.

### 3.4 Data Pre-processing
Data pre-processing ensures the data quality before model training. Steps include data loading, feature selection, outlier removal, and standardization using z-score.

#### 3.4.1 Data Loading
Signals retrieved from field test trucks are processed to select significant features, remove outliers, and handle NaN values.

#### 3.4.2 Feature Selection
Selected one battery pack with minimal NaN values and extracted essential columns such as current, voltage, and ambient air temperature.

#### 3.4.3 Outlier Removal
Implemented thresholds for current and voltage to remove outliers based on standard battery module settings.

#### 3.4.4 Status Consideration
Focused on the 'Driving Not charging' status due to its significant impact on sensor value changes.

#### 3.4.5 Batch Separation
Divided data into batches based on vehicle conditions, ensuring model learning from diverse scenarios.

#### 3.4.6 Standardization
Implemented z-score standardization to maintain data consistency across features.

### 3.5 Model Architecture
Based on the Transformer encoder, the model processes time-series data, ensuring attention to temporal aspects and feature interdependencies.

#### 3.5.1 Data Input
Adopted techniques like data slicing and padding to maintain consistent input lengths for the model.

#### 3.5.2 Model Mechanism
Explained the Transformer encoder's working, focusing on input encoding, attention layers, and learning interdependencies.

### 3.6 Evaluation Metrics
Various evaluation metrics, including visual and quantitative methods, assess model performance.

#### 3.6.1 Visual Metrics
Used methods such as Prediction vs Original plots, PCA, and Boxplots to visualize and compare predicted and actual values.

#### 3.6.2 Quantitative Metrics
Implemented metrics like Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) to quantitatively evaluate model accuracy.


-------

Paper referred -  [A Transformer-based Framework for Multivariate Time Series Representation Learning](https://arxiv.org/abs/2010.02803)
