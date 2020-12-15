# Predicting Churn of customers
![telecom-photo](/notebooks/report/figures/Read_me.png)

# Table of Contents

<!--ts-->
 * [General Setup Instructions](https://github.com/howen7/Alzeimers#general-setup-instructions)
 * [Context of Project](https://github.com/howen7/Alzeimers#Context)
 * [Data](https://github.com/howen7/Alzeimers#Data)
 * [Process](https://github.com/howen7/Alzeimers#models-used--methodology)
 * [Results & Next Steps](https://github.com/howen7/Alzeimers#Results)
<!--te-->
```
.
├── README.md     
├── environment.yml
├── notebooks
│   ├── exploratory
│   │   ├── Hunter.ipynb
│   │   ├── Hunter2.ipynb
│   │   ├── 01_oz_data_collections_exploration.ipynb
│   │   ├── oz_01_data_collections_exploration.ipynb
│   │   ├── oz_02_log_reg.ipynb
│   │   ├── oz_03_rad_forest.ipynb 
│   └── report
│       ├── figures
│       │   ├── CostDayCharge.png
│       │   ├── CustumorServiceCalls.png
│       │   ├── Early_Models.png
│       │   ├── Feat.png
│       │   ├── FeatureImportanceEarly.png
│       │   ├── Final_model.png
│       │   ├── Final_model_features.png
│       │   ├── Internationalplan.png
│       │   ├──RandomForest.png
│       │   ├──RandomForestclassifier.png
│       └── FinalNotebook.ipynb
│   
└── src
    ├── my_mods.py
    └── data
        ├──X_dataframe.csv
        ├──syriatel_customer_churn.csv
        └──y_dataframe.csv
```
#### Repo Navigation Links 
 - [Final summary notebook](https://github.com/howen7/Alzeimers/tree/main/notebooks/report/FinalNotebook.ipynb)
 - [Exploratory notebooks folder](https://github.com/howen7/Alzeimers/tree/main/notebooks/exploratory)
 - [src folder](https://github.com/howen7/Alzeimers/tree/main/src)
 - [Presentation.pdf](https://github.com/howen7/Alzeimers/tree/main/reports)

# General Setup Instructions 

Ensure that you have installed [Anaconda](https://docs.anaconda.com/anaconda/install/).

### `alz-env` conda Environment

This project relies on you using the [`environment.yml`](environment.yml) file to recreate the `alz-env` conda environment. To do so, please run the following commands *in your terminal*:
```bash
# create the alz-env conda environment
conda env create -f environment.yml
# activate the conda environment
conda activate alz-env
# if needed, make alz-env available to you as a kernel in jupyter
python -m ipykernel install --user --name alz-env --display-name "Python 3 (alz-env)"
```


# Project Goal:

In this project we looked at, and analyzed, SyriaTel's customer data to see what was driving their churn rate. Our goal was to come up with a model that could help us predict, with high precision and recall, when a customer churns, or leaves the company, and then use that model to help us narrow down which factors were the most important in driving customer churn.


# Data:

The data we used can be found here: https://www.kaggle.com/becksddf/churn-in-telecoms-dataset
As it stands, their current churn rate is 15%. 


# Models used + Methodology:

EDA:

After identifying and separating our target variable, 'churn', we checked for and found a severe class imbalance here. We knew this will have to be addressed in the preprocessing part.

We then studied the features we had. We found that some features were directly correlated with each other, specifically all the mintue and charges columns for each time of day, and international calls. The columns were: day total mintues and charges, evening total mintues and charges, evening total mintues and charges, and the international total minutes and charges. From these columns, we dropped all the total minute columns, and kept the charges columns.

We also saw that the phone number column contained all unique values, so we dropped this as well. 

Furthermore, the area code column only contained 3 unique values even though we had 51 states represented in the data. We concluded that this was most likely a voice-over-IP service like Vonage, where the company was assigning phone numbers with area codes from the location their company was operating from, in this case it was most likely San Francisco, since all 3 area codes were from there. We dropped this column as well.

After we ran our first simple models (FSM), specifcally the logistic regression FSM, we checked the coefs and saw that the states had little affect on our models prediction, so we ended up dropping this column as well.

---

TRAIN-TEST-SPLIT:

Once we had narrowed down and finalized our feature columns, we split our data into training and test sets. We did this right now, and not after preprocessing our data, because we didn't want our data leaking between our training and testing set. We then moved on to preprocessing our data to prepare for modeling.

---

PREPROCESSING:

We began by first separating our numerical columns from our categoricals. 

-OneHotEncoding-
We were left with only 2 categorical columns: International plan, and voicemail plan. Both columns had either a 'yes' or 'no', to indicate if the customer had any of these plans. We used OneHotEncoder to dummy out these columns.


-Scaling-
We then joined our dummy and numerical columns togther, and scaled them all using standard scaler. We decided to scale our dummy columns because we saw that doing so improved our FSM's score vs only scaling our numericals.


-SMOTE-
As mentioned earlier, we had a severe class imbalance within this data set: 85% False vs 15% True. We used SMOTE to deal with this. And we did this only within our training data set.

---

MODELING:

We tried several modeling algorithms (in order):

Logistic Regression

KNN

Random Forest

XGBoost

For each we did a first simple model initially, and compared each model's scores. At this point, once we had decided on our final dataframe (keep in mind we were still tuning the features as we ran FSMs), we decided to create a pipeline containing the following:
- OneHotEncoder to dummy out our categoricals
- StandardScalar to scale all our data
- SMOTE to balance our classs within our train dataset
- Model algorithm. 

# Results:
-Final Model-
Once we set up our pipeline, we used gridsearch to help us tune the parameters for each modeling algorithm to see which one gave us the best scores, and in the end, Random Forest and XGBoost gave us the best scores, with XGBoost being slightly better overall, so we decided to go with that as our final model, and with the following params:
- learnin_rate = 0.01
- max_depth = 4
- n_estimators = 500
- gamma = 0.3

This gave us the following scores:
False --> Precision 97% -- Recall 97% -- f1 score 97%
True  --> Precision 83% -- Recall 84% -- f1 score 83%

Out of the 101 people who churned (chrun = True), our model correctly predicted 85
Out of the 566 people who statyed, our model correctly predicted 548


# Further Analysis
We looked at the feature importance of our model to see which features best predicted churn, and made a list of the top 5, in order:

- International Plan
- Customer Service Calls
- Total Day Charge
- Total International Calls
- Voicemail Plan

We went back to our original data to see the correlation between these features and churn, and confirmed that these features did indeed have a high correlation with an increased churn rate, which validates our model.

---

# CONCLUSION:

Our findings are stated below, along with our recommendations:

- International Plan: 
Customers with international plans churned more, indicating customers might not be happy with the international plan. 
RECOMMENDATION: Reach out to these customers and ask them what they like and don't like about the international plan

- Customer Service Calls
The more customers reached out to customer service, the more likely they were to churn. Specially after 3 customer service calls, after which the churn rate increased significantly.
RECOMMENDATION: Have a manager reach out to customers who have made more than 3 customer service calls to find out and address their concerns. In addition, perhaps look at improving customer service department.

- Total Day Charge
This was the most concerning to us. Here we saw that the more customers used your service, the more likely they were to churn, specially if they were using more than 235 mins a day. Reach out to these customers and see what they like and don't like about the service.

- Total International Calls
This further validates our findings concerning the higher churn rate of customers who have an International Plan, in that they're most likely not happy with the quality of this plan. See our recommendations above for this.

- Voicemail Plan
Customers who DID NOT have a voicemail plan tended to churn more.
RECOMMENDATION: reach out to customers who don't have a voicemail plan, and offer them a promotion perhaps to get them to sign up with the voicemail plan.

