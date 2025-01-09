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
 * Understand Logistic regression, it's use case in the context of this dataset and build more complex models with the dataset. 


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
* For the model, a logistic regression model should be sufficient in identifying potential customers who will churn. Starting with a simple logistic regression model with the target value being **churn_value** since the value is already a binary value 0 or 1.
* We choose a logistic regression model because our main goal was predicting if a customer will churn with this project, which logistic regression will prove to be the best. 
* Understanding logistic regression, we need target variable(binary) and features(independent variables).
* For metrics of the model, we will use the scikit-learn libaries importing metrics such as accuracy score, confusion matrix, classification report, roc curve and AUC score. 


### Model Evaluation
* Looking at the results from the logistic regression prediction model, the results showed an accuracy of 83%, the results are decent for a prediction model.
* The results show a heavy imbalance between predicting non-churners as the dataset contains a lot of non-churn customers which may be a reason. 



**Classification Report:**

![image](https://github.com/user-attachments/assets/abd60168-a760-4980-a7db-91f6ee5741e4)

**Dataset Imbalance Impact:**

- The model performs significantly better in predicting non-churned customers (0) compared to churned customers (1).
- Precision, recall, and F1-score for Class 0 (88%, 89%, 0.88) are much higher than for Class 1 (68%, 65%, 0.67).
- This suggests that the model is biased towards the majority class (non-churned customers).
- The model definitely struggled in preodicting churners which was the main focus in attempting due to class imbalance
- A look into techniques like oversampling (e.g., SMOTE), undersampling, or using class weights in the logistic regression algorithm could help address this issue.


**Confusion Matrix:**

![Confusion Matrix](https://github.com/user-attachments/assets/1646c4ec-8729-4abf-a20b-f0f0c524dabd)

![image](https://github.com/user-attachments/assets/3e4b545c-8f6d-4f9b-a8f5-14481942291e)

**Confusion Matrix Analysis:**

- True Negatives (924): The model accurately predicts most non-churned customers.
- False Positives (112): Some customers who are not likely to churn are incorrectly predicted as churners.
- False Negatives (131): A notable number of churned customers are incorrectly classified as non-churned, indicating a need for better churn prediction.
- True Positives (242): The model successfully identifies many churned customers but struggles with recall for this group.


**ROC Curve:**

![image](https://github.com/user-attachments/assets/1b80b2be-1ce1-4c18-a940-ddc4284169be)

- Looking at the ROC curve shows the tradeoff between sensitivity (recall for churners) and specificity (ability to avoid false positives for non-churners
- The AUC is a single scalar value that can be used to summarize the ROC curve.
- The AUC score of 88 shows the model is highly effective but could benefit from slight refinements, such as better feature selection, handling class imbalance, or hyperparameter tuning.

Room for Improvement in Recall for Churners:

- While AUC is high, your recall for predicting churners (Class 1) is only 65%. This means 35% of actual churners are being misclassified as non-churners.
- Ways to fix this issue of churners and non-churners inbalance, Adjust the threshold to increase recall for churners and address class imbalance using techniques like oversampling (SMOTE) or undersampling.

### Conclusion and Future Work
* The model showed 83% accuracy which is decent results but does not paint the whole picture in terms of inbalance. 
* For future expansion, building more complex models such as Random Forest, Gradient boosting, AdaBoostRegressor and KNeighborsRegressor
* Look into solving the data imbalance in non-churn and churn customers in the dataset with techniques like oversampling (e.g., SMOTE), undersampling, or using class weights in the logistic regression algorithm could help address this issue.
* Learning more about predictive modeling can be beneficial, building more feature engineering with more refine features and compare alternative other predictive models. 


