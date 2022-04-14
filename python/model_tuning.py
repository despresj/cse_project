import time
import pandas as pd
import numpy as np
import joblib
import multiprocessing
from sklearn import model_selection

start = time.time()
df = pd.read_csv("data/processed/wine_data_combined.csv")
cols_to_adjust = [x for x in df.columns if x not in ["quality", "is_red"]]

retrain = True # Set to True and rerun the model gridsearches
short_runtime = False # Set to true for a narrow hyperparam search

model_path = "short_run_params" if short_runtime else "saved_models/"

cores = multiprocessing.cpu_count()

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import svm

df_train, df_test = model_selection.train_test_split(
    df, test_size=0.3, random_state=55, stratify=df["quality"]
)

from sklearn.preprocessing import StandardScaler 
from sklearn.compose import ColumnTransformer

stratified = model_selection.StratifiedKFold()
df_transform = ColumnTransformer(
    [(" ", StandardScaler(), cols_to_adjust)],
    remainder="passthrough",
)
df_train = pd.DataFrame(df_transform.fit_transform(df_train), columns=df.columns)
X_train = df_train.drop("quality", axis=1)
y_train = pd.Categorical(df_train["quality"], ordered=True)
models = []

### Logistic Reg ###

multi_logistic_reg = LogisticRegression(penalty="elasticnet", solver="saga", tol=1e-2, max_iter=500)

params_logistic = {"C": np.logspace(-5, 0, 100), "penalty": ["l1", "l2"]}

logistic_grid_search = model_selection.GridSearchCV(
    multi_logistic_reg, params_logistic, n_jobs=cores)
# Not all of these converge given the low tolerance I set above

if short_runtime:
    params_logistic = {"C": np.logspace(-5, 0, 10), "penalty": ["l1", "l2"]}

if retrain:
    logistic_grid_search.fit(X_train, y_train)
    joblib.dump(logistic_grid_search, "saved_models/logistc_grid_search.dat")
    
logistic_grid_search = joblib.load(f"{model_path}logistc_grid_search.dat")
models.append(("logistic_reg", logistic_grid_search))

print(logistic_grid_search.best_params_)
### XGB ###

params_xgb = {'max_depth': np.arange(3, 20, 1),
             'colsample_bytree' : np.arange(0.2, 1, 0.1),
             "learning_rate": np.logspace(-5, -1, 9)}

if short_runtime:
    params_xgb = {'max_depth': np.arange(13, 13, 1),
             'colsample_bytree' : np.arange(0.5, 0.7, 0.2),
             "learning_rate": np.logspace(-4, -3, 1)}

xgboost = xgb.XGBClassifier()

xgb_grid_search = model_selection.GridSearchCV(
    xgboost, params_xgb, n_jobs=cores, cv=stratified
)

if retrain:
    xgb_grid_search.fit(X_train, y_train)
    joblib.dump(xgb_grid_search, f"{model_path}xgb_grid_search.dat")

xgb_grid_search = joblib.load(f"{model_path}xgb_grid_search.dat")

models.append(("xgb", xgb_grid_search))

print(xgb_grid_search.best_params_)

### SVM ###

params_svm = {"C": np.logspace(-3, 3, 500)}

support_vector_machine = svm.SVC(gamma='auto',probability=True)

if short_runtime:
    params_svm = {"C": np.logspace(-3, 3, 10)}
    
svm_grid_search = model_selection.GridSearchCV(support_vector_machine, params_svm, n_jobs=cores, cv=stratified)

if retrain:
    svm_grid_search.fit(X_train, y_train)
    joblib.dump(svm_grid_search, f"{model_path}svm_gridsearch.dat")
    
svm_grid_search = joblib.load(f"{model_path}svm_gridsearch.dat")
print(svm_grid_search.best_params_)  

models.append(("svm", svm_grid_search))

### Ensemble ###

from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(models, voting="soft", weights=[1,3,1], n_jobs=cores)

if retrain:
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, f"{model_path}ensemble.dat")

ensemble = joblib.load(f"{model_path}ensemble.dat")

from sklearn.pipeline import Pipeline

if retrain:
    pipeline = Pipeline([("scaler", df_transform), ("ensemble", ensemble)])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f"{model_path}pipeline.dat")
    
pipeline = joblib.load(f"{model_path}pipeline.dat")

### Metrics ###

yhat = pipeline.predict_proba(X_train)
from sklearn import metrics
auc = metrics.roc_auc_score(y_train, yhat, multi_class='ovr')
print("accuracy",np.sum(pipeline.predict(X_train) == y_train)/len(y_train))
print("accuracy",np.sum(yhat == y_train)/len(y_train))

end = time.time()
print(f"runtime {(end - start)/60}")  
