from sklearn.datasets import load_iris
iris = load_iris()

from joblib import load 
classKNN = load('iris_classknn_job.joblib')

sampledata = [[1,2,3,4],[4,5,6,7]]
preds = classKNN.predict(sampledata) 

pred_species = [iris.target_names[p] for p in preds]
print("Predictions: ", pred_species )