# Regression Analysis of Steel Products Offer Prices.

## INTRODUCTION
- We have 181630 rows of data of various steel items having following features: 'Id',   'item_date',   'quantity tons',   'customer',   'country',   'status',   'item type',   'application', 'thickness',   'width',   'material_ref’,   'product_ref',   'delivery date',   'selling_price'.
- The project performs a regression analysis with selling_price feature to be predicted.
- We divide data into 90-10% train-test split respectively. Analysis and model is done on the train set and tested on unseen test data.

## SUMMARY
- Selected Model: XGBoost
- Hyperparameter search method: Bayesian Search CV
- Best r2 score:
  - Test r2 score: 0.90
  - Train r2 score:  0.92
- Features used:
  - ‘quantity tons’
  - ‘application’
  - ‘thickness’
  - ‘width’
  - ‘product_ref’
  - ‘delivery date’
  - ‘status’ .
- Categorical features shortlisting criteria: ANOVA test F statistical scores. (for details, refer EDA notebook)
- High value of ANOVA F statistics shows that, the mean and spread of data among different categories are easily separable so such         features gives better r2 score and explaining variance for model becomes easy.
- Numerical features were power transformed and standardized to get near gaussian like distribution.
- Categorical features were one-hot encoded.
- Total input features: 98
- Important Features:
  - ‘status’
  - ‘delivery date’

|Model|Train r2 score|Val r2 score|Test r2 score|Hyperparameter search method|
|----------------|--------------|------------|-------------|----------------------------|
|Linear Regresion|0.83          |0.83        |0.82         |Grid Search                 |
|Decision Tree   |0.86          |0.80        |not performed|Grid Search                 |
|Random Forest   |0.70          |0.69        |not performed|Bayesian Search             |
|XGBoost         |0.92          |0.89        |0.90         |Bayesian Search             |
|Neural Network  |0.90          |0.89        |0.89         |None                        |





