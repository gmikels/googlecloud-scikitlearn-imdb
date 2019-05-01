import datetime
import pandas as pd

from google.cloud import storage

from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

# TODO: REPLACE 'YOUR_BUCKET_NAME' with your GCS Bucket name.
BUCKET_NAME = 'gmikels-imdb'

# Public bucket holding the census data
bucket = storage.Client().bucket(BUCKET_NAME)

# Path to the data inside the public bucket
blob = bucket.blob('train.csv')
# Download the data
blob.download_to_filename('train.csv')

# Import Training Data
df = pd.read_csv('./train.csv', delimiter=',')
data = df['reviewtext'].tolist()
target = df['sentiment'].tolist()
print("n_samples: %d" % len(data))

# split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    data, target, test_size=0.25, random_state=None)

# TASK: Build a vectorizer / classifier pipeline that filters out tokens
# that are too rare or too frequent
pipeline = Pipeline([
    ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
    ('clf', XGBClassifier()),
])

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
grid_search.fit(docs_train, y_train)

# TASK: print the mean and std for each candidate along with the parameter
# settings for all the candidates explored by grid search.
n_candidates = len(grid_search.cv_results_['params'])
for i in range(n_candidates):
    print(i, 'params - %s; mean - %0.2f; std - %0.2f'
             % (grid_search.cv_results_['params'][i],
                grid_search.cv_results_['mean_test_score'][i],
                grid_search.cv_results_['std_test_score'][i]))

# TASK: Predict the outcome on the testing set and store it in a variable
# named y_predicted
y_predicted = grid_search.predict(docs_test)

# Print the classification report
print(metrics.classification_report(y_test, y_predicted))

# Print and plot the confusion matrix
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)

# Export the model to a file
model = 'model.joblib'
joblib.dump(pipeline, model)

# Upload the model to GCS
bucket = storage.Client().bucket(BUCKET_NAME)
blob = bucket.blob('{}/{}'.format(
    datetime.datetime.now().strftime('imdb_xgboost_%Y%m%d_%H%M%S'),
    model))
blob.upload_from_filename(model)

# Write Results to GCS
out = open("results.txt", "w")
out.write("xgboost")
out.write(",")
out.write(str(metrics.accuracy_score(y_test, y_predicted)))  
out.close()
blob = bucket.blob('{}.txt'.format(
    datetime.datetime.now().strftime('accuracy/results_xgboost_%Y%m%d_%H%M%S')))
blob.upload_from_filename('results.txt')