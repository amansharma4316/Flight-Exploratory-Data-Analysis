# Flight Data - Exploratory Data Analysis and Machine Learning Model
This project involves conducting Exploratory Data Analysis on flight journey details. It encompasses analyzing various parameters and visualizing the data. Additionally, a machine learning model is constructed using the Naive Bayes method to predict the arrival status of the flight (on-time or late).

In PART 1 of this project, an Exploratory Data Analysis will be conducted on the provided dataset to analyze various features (columns). Various methods will be used to interpret their relation, and they will be visualized.

In PART 2, the models will be trained based on the Naive Bayes theorem to predict whether the flight arrives on time or experiences delays.

## Highlights of the data
The data consists of 10 features (columns) namely
- Carrier
- Departure_Time
- Destination
- Date
- Flight_Number
- Origin
- Day_of_the_Week
- Day_of_the_Month
- Day of the Week

To view the dataset, click [Dataset](Flight-Data.csv)

Each number represents the corresponding day

- 1- Monday
- 2- Tuesday
- 3- Wednesday
- 4- Thursday
- 5- Friday
- 6- Saturday
- 7- Sunday

Day of the Month

Each number represents the days in a month.

Abbreviations

### Carrier

- CO - Continental
- DH - Atlantic Coast
- DL - Delta
- MQ - American Eagle
- OH - Comair
- RU - Continental Express
- UA - United
- US - USAirways

### Destination

- JFK - Kennedy
- LGA - LaGuardia
- EWR - Newark

### Origin

- DCA - Reagan National
- IAD - Dulles
- BWI - Baltimore–Washington Int’l

## Libraries Used

- Pandas is for converting the required scrapped data list into DataFrame
- Numpy for numerical operations, providing support for large, multi-dimensional arrays
- Matplotlib for creating static, interactive, and animated plots and visualizations such as line plots, bar charts, histograms,
- Seaborn for providing attractive and informative statistical graphics.

      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      import seaborn as sns

## Data Preprocessing

Data preprocessing is done to clean, transform, and prepare raw data before it is used for analysis or modeling. It is an essential step in the data analysis pipeline because raw data often contains noise, inconsistencies, missing values, and other issues that can negatively impact the accuracy and reliability of the results.

Additionally, the dataset undergoes statistical analysis to assess its characteristics, which includes examining the presence of null values, calculating frequencies, and other relevant measures.

## Data Visualization

Data visualization is the graphical representation of data and information. It uses visual elements like charts, graphs, and maps to present complex data in a more understandable and visually appealing way. The goal of data visualization is to distill large datasets into actionable insights and patterns, enabling easier interpretation and communication of information.

`Visualization` is performed on multiple features, showcasing their relationships with each other and various parameters. This project shows the visualizations in forms of

- Pairplot
- Relplot
- Histogram
- Pie-chart
- Barplot
- Strip-plot
- Jointplot

## Machine Learning Model

### Model Classification and Fitting

For creating an ML model to predict emotions, several libraries are imported and utilized to train the existing dataset. Various Machine Learning algorithms are employed to build a model that can provide predictions based on the available data. The system incorporates multiple functions and libraries to ensure accurate and precise predictions. Among the algorithms used for predicting the desired parameters, one notable method is the Naive Bayes Probability theorem.

Naive bayes can be classified into three types namely

  - GAUSSIAN     : for continous data
  - MULTINOMIAL  : for discrete data
  - BERNOULLI    : for binary data

`Gaussian model` is used here for the prediction
  
      from sklearn.preprocessing import LabelEncoder
      from sklearn.naive_bayes import GaussianNB
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import confusion_matrix , plot_confusion_matrix
      from sklearn.metrics import accuracy_score,precision_score,recall_score

- LabelEncoder for converting `categorical` labels into `numerical` format, commonly used for transforming target labels in machine learning tasks.
- GaussianNB for implementing Gaussian Naive Bayes algorithm, a probabilistic classifier for classification tasks based on `Bayes' theorem`.
- Train test split for `splitting` datasets into `training` and `testing` subsets, essential for `evaluating` machine learning models.
- Confusion Matrix to compute a confusion matrix, showing the performance of a classification model by comparing predicted and true labels.
- Plot confusion matrix for `plotting` a `confusion matrix`, offering a `visual representation of model classification performance`.
- Accuracy Score for calculating the accuracy score, a metric that measures the ratio of correctly predicted samples to the total number of samples.
- Precision score for calculating the precision score, a metric that measures the ability of the model to correctly predict positive samples.
- Recall score for calculating the recall score, a metric that measures the ability of the model to correctly identify positive samples.
 
      x_train,x_test,y_train,y_test=train_test_split(gets,target,test_size=99,random_state=5)
      model=GaussianNB()
      model.fit(x_train,y_train)
      model.score(x_test,y_test)

Our model generated an  accuracy score of `0.8282828282828283`

### Confusion Matrix

A confusion matrix is a performance evaluation tool used in the field of machine learning and classification tasks. It is a square matrix that summarizes the performance of a classification model by comparing the predicted class labels with the actual class labels.

The confusion matrix has four key elements:

 1. True Positives (TP): The number of instances that are correctly predicted as positive by the model. In other words, these are the cases where the model predicted the positive class, and the actual class is also positive.
 2. True Negatives (TN): The number of instances that are correctly predicted as negative by the model. These are the cases where the model predicted the negative class, and the actual class is also negative.
 3. False Positives (FP): The number of instances that are incorrectly predicted as positive by the model. These are the cases where the model predicted the positive class, but the actual class is negative.
 4. False Negatives (FN): The number of instances that are incorrectly predicted as negative by the model. These are the cases where the model predicted the negative class, but the actual class is positive.

For viewing the jupyter notebook code, click [Flight Data EDA](FLIGHT-DATA-EDA.ipynb)

