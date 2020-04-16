from sklearn.metrics import precision_recall_curve,confusion_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
%matplotlib inline

X, y = make_classification(n_samples=5000, n_features=5, n_redundant=2,n_classes=2, weights=[0.7], class_sep=0.7)
lr = LogisticRegression()
lr.fit(X,y)
print('Confusion Matrix: \n',confusion_matrix(y,lr.predict(X)))
plt.plot(precision_recall_curve(y,lr.predict_proba(X)[:,1])[0], label ='Precision')
plt.plot(precision_recall_curve(y,lr.predict_proba(X)[:,1])[1], label = 'Recall')
plt.title('Precision- Recall Curve')
plt.legend()
