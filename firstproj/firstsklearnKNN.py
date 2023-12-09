import pandas as pd 
from sklearn.datasets import load_iris 

iris = load_iris()

X = iris.data 
y = iris.target

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size= 0.1, random_state=1 
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 

classifier_knn = KNeighborsClassifier(n_neighbors=3)
classifier_knn.fit(X_train, y_train) 
y_pred = classifier_knn.predict(X_test) 

print("Accuracy :", metrics.accuracy_score(y_test, y_pred) )

sample = [[5,5,3,2],[2,4,3,5]]

preds = classifier_knn.predict(sample) 

pred_species = [iris.target_names[p] for p in preds] 
print("Predictions :", pred_species) 
 
from joblib import dump
dump(classifier_knn, 'iris_classknn_job.joblib')
