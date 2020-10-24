# Predictive_Modeling
3rd project for flatiron


The goal of this project is to create a model that can accurately predict Syria Tel’s churn rate, or which of Syria Tel’s customers are likely to leave. The analysis is done with the hopes that if Syria Tel can understand who is likely to leave, than perhaps they can invest resources in preventing the customers from doing so. 

The data set we used comes directly from Syria Tel and can be found on Kaggle. After eliminating the Area Code and the account numbers, our dataset consisted of customer profiles that included whether or not the the customer had an international plan, a voice mail plan, the minutes charged, the number of customer service calls they mad, their state and….



As it stands, the churn rate for Syria Tel is about 15%. 






We found that the best predictor for whether or not a customer would leave is whether or not they had an international plan. 



Image 






We assumed complaints would be a solid indicator but wanted to see if this assumption was reflected in the data. As you can see, if a customer had at least 4 interactions with customer service, they were much more likely to leave. 













We tested several different types of classification models including: 

Decision Tree

Logistic Regression

Random Forest 

KNN 

AdaBoost

Gradient Descent 


SInce our goal for the project was to help identify which customers were at a high risk of leaving, we decided to rely on the Recall Metric which minimizes false negatives. Eventually, our best model was a Gradient Boost model, that produced an accuracy score of …. and a 



Above is our confusion matrice for our best model. 





Recommendations. 
