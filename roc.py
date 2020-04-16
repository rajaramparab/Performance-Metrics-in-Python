from sklearn.metrics import plot_roc_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

#making dummy dataset and splittign into train and test
X, y = make_classification(n_samples=5000, n_features=5, n_redundant=2,n_classes=2, weights=[0.7], class_sep=0.7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#fitting Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train,y_train)

#plotting roc curve
plot_roc_curve(lr,X_test,y_test)
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve and AUC')
plt.legend()
