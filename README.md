# Regression Analysis of steel products offer prices.

## INTRODUCTION
- We have 181630 rows of data of various steel items having following features: 'Id',   'item_date',   'quantity tons',   'customer',   'country',   'status',   'item type',   'application', 'thickness',   'width',   'material_ref’,   'product_ref',   'delivery date',   'selling_price'.
- The project performs a regression analysis with selling_price feature to be predicted.
- We divide data into 90-10% train-test split respectively. Analysis and model is done on the train set and tested on unseen test data.

## SUMMARY
- Selected Model: XGBoost
- Hyperparameter search method: bayesian Search CV
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
- Categorical features shortlisting criteria: ANOVA test F statistical scores.


