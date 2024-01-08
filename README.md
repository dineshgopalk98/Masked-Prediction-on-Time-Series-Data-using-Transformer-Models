## Masked Prediction on Time Series Data using Transformer Models

### Introduction

The project addresses the challenge of imputing missing data in time series datasets efficiently. Traditional methods, such as auto-regressive approaches, are often constrained by computational costs and time limitations. This project focuses on designing a neural network model, specifically a Time Series Transformer (TST), for masked prediction in time series data.

### Project Goals

1. **Understanding Data Source:** Thoroughly study the data source to grasp the meaning and significance of the data. Utilize various data visualization methods for in-depth analysis, gaining insights into the importance of each dataset.

2. **Selection of Evaluation Metrics:** Identify and justify appropriate evaluation metrics to assess the reliability and performance of the TST model. This decision is crucial and will be made as part of the pre-study.

3. **Data Pre-processing:** Prepare the data for modeling by applying various pre-processing techniques. The goal is to maintain data properties while extracting valuable information through visual representations.

4. **Neural Network Model Creation:** Develop a robust neural network model, TST, tailored to the project's objectives. This selection is informed by a thorough literature study conducted during the project's initial phases.

5. **Visualization and Evaluation:** Present the predicted data in a 2-dimensional view, allowing for easy comparison of correlations between different features in both original and predicted data. Assess the quality of predicted data using chosen evaluation metrics, ensuring alignment with the project's proposed objectives.

### Project Overview

This project involves the creation of a machine learning model that predicts missing values in time series data. By leveraging a specialized Transformer model, the Time Series Transformer, the aim is to enhance the quality of collected data and subsequently improve vehicle performance. The README provides a concise overview, emphasizing the significance of understanding the data, selecting appropriate metrics, pre-processing, model creation, and the ultimate goal of visualizing and evaluating predicted data.

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
