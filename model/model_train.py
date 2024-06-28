import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sn

# Load X and y arrays from disk
X = joblib.load('X.pkl')
y = joblib.load('y.pkl')

print("Loaded X and y arrays.")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Define the SVM pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])
pipe.fit(X_train, y_train)
print(f"Initial SVM score: {pipe.score(X_test, y_test)}")

# Define model parameters for GridSearch
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    }
}

scores = []
best_estimators = {}

# Perform GridSearch for each model
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

# Print the best scores and parameters
df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)

# Evaluate the best SVM model
best_clf = best_estimators['svm']
print(f"Best SVM score: {best_clf.score(X_test, y_test)}")

# Confusion matrix
cm = confusion_matrix(y_test, best_clf.predict(X_test))
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

class_dict = {
    'lionel_messi': 0,
    'maria_sharapova': 1,
    'roger_federer': 2,
    'serena_williams': 3,
    'virat_kohli': 4
}

import joblib 
# Save the model as a pickle in a file 
joblib.dump(best_clf, 'saved_model.pkl') 

import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))