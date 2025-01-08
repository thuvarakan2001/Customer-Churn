# Customer Churn Prediction Model

![Customer Churn](https://github.com/user-attachments/assets/646a3608-ba63-4fdc-8041-f7e727a3037e)



## Table of Contents
1. Introduction
2. Data Loading
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis
6. Model Building
7. Model Evaluation
8. Conclusion and Future Work

### Introduction
Using the telco churn [dataset](https://www.kaggle.com/datasets/hassanelfattmi/why-do-customers-leave-can-you-spot-the-churners/data) containing information about a fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3. The dataset provides data on Satisfaction Score, Churn Score, and Customer Lifetime Value (CLTV) index and unique data for each customer including unique identifier, location data, payment information, service info and status analysis for each customer. The data can provide in building a logistic regression prediction model to predict if a customer will churn or not. Using python data science libraries as the main technologies in building this logistic regression prediction model and conducting data cleaning for best accuracy.

Main goals from this project and future expansion: 
 * Using python data science libraries to further understand the dataset and build a prediction model
 * Apply proper data cleaning and preprocessing techniques to dataset to ensure best accuracy from model
 * Future expansions to the analysis like future feature building and creating more advanced machine learning models with the dataset
 * Understand Logistic regression, it's use case in the context of this dataset and build more complex models


### Data Loading
First we load the dataset into python and the necessary libraries needed for the analysis and model building. In this case the libraries needed for this would be pandas, numpy, seaborn, matplotlib, and scikit-learn for logistic regression. Pandas for data manipulation, exploratory data analysis, data cleaning, numpy for any numerical calculations and any future mathematical operations we may need. Seaborn and matplotlib for any data visualizations when exploring the dataset and looking at any useful insights. Finally Scikit-learn machine learning library for building the logistic regression prediction model including train_test_split, LogisticRegression, accuracy_score, confusion_matrix, classification_report,roc_curve, auc etc. 


### Data Cleaning and Preprocessing
Before diving into analysis and model building, we need to clean and preprocess the data. This includes handling missing values, correcting data types, and merging datasets where necessary. The data was pretty useble from the start and only a few column names had to be fixed once merging all of the datasets together. For my analysis I combined all of the datasets as I may in the future may use some of the other columns for building more models etc, but normally I would drop any columns I am not using in my analysis. 

### Exploratory Data Analysis
A deeper dive into the dataset looking at potential insights in the analysis that can be looked at and understanding potential features and target values for the prediction model involving logistic regression. We will be using the seaborn library to look at some basic countplots to understand the data and how the distributions are looking for certain columns which can be used the future for selecting the correct features and target value for the prediction model. 

**Churn Distribution:**

![Churn Distribution](https://github.com/user-attachments/assets/97dd39a2-e6b8-4055-ae1c-1797b39e8871) 

Looking at the churn distribution, we can see a majority of customer stayed with the company this quarter and did not not leave the company. 

**Customer Married Distribution:**

![image](https://github.com/user-attachments/assets/10b64cbc-2fbd-46e2-937f-ce0df1246e3a)

Looking at the countplot, majority of customers are not married which may provide insight into or can be futher looked into if that is a reason they left or stayed with the service. 

**Customer Age Distribution:**

![image](https://github.com/user-attachments/assets/94236f22-2fa0-4d90-816d-6c70151d05b2)

Customer's age can be a factor if they churn or not, older customers may not be interested in the service as a majority of the customers are around 40+ in age which can be something to further investigate. 

**Correlation Heatmap:**

![image](https://github.com/user-attachments/assets/5497551a-f973-4615-b0e9-2bfc49149c4f)

Looking at the correlation heatmap, we can see there are a lot of negative and a few positive correlation between each of the columns. Some positive correlations that can be further explored include Total revenue and tenure, total revenue and total charges, a customer's monthly charges compared to total charges. Some negative correlations are churn value and satisfaction score, satisfaction and monthly charges etc.  



### Model Building


### Model Evaluation


### Conclusion and Future Work
