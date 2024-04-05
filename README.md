# Predicting-Credit-Card-Approvals
This project aims to predict credit card approvals using machine learning techniques. We use a dataset that contains various information about the applicants and whether their credit card application was approved or not.

## Code Overview
The code snippet provided shows the initial data preparation steps:
'''python
X=df.drop('not.fully.paid', axis=1)
y=df['not.fully.paid']
y = y.astype(int)
y = pd.to_numeric(y, errors='coerce')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
'''python
