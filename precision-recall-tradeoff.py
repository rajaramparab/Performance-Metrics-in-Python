from sklearn.metrics import precision_recall_curve,confusion_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

#making dummy dataset and splittign into train and test
X, y = make_classification(n_samples=5000, n_features=5, n_redundant=2,n_classes=2, weights=[0.7], class_sep=0.7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#fitting Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train,y_train)

#plotting precision-recall Curve with Thresholds
plt.plot(precision_recall_curve(y_test,lr.predict_proba(X_test)[:,1])[0], label ='Precision')
plt.plot(precision_recall_curve(y_test,lr.predict_proba(X_test)[:,1])[1], label = 'Recall')
plt.plot(precision_recall_curve(y_test,lr.predict_proba(X_test)[:,1])[2], label = 'Thresholds')
plt.title('Precision- Recall Curve')
plt.legend()
