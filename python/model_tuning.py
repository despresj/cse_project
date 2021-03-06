import time
import pandas as pd
import numpy as np
import joblib
import multiprocessing
from sklearn import model_selection

start = time.time()
df = pd.read_csv("data/processed/wine_data_combined.csv")
cols_to_adjust = [x for x in df.columns if x not in ["quality", "is_red"]]

retrain = False # Set to True and rerun the model gridsearches
train_ensemble = True

model_path = "saved_models/_yao_transform_"

cores = multiprocessing.cpu_count()
cores -= 2 # so i can do other stuff when this trains

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score

df_train, df_test = model_selection.train_test_split(
        df, test_size=0.2, random_state=66, stratify=df["quality"]
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

df_transform = ColumnTransformer(
    [("yao", PowerTransformer(), cols_to_adjust)],
    remainder="passthrough",
)

df_train = pd.DataFrame(df_transform.fit_transform(df_train), columns=df.columns)
df_train\
        .groupby('quality', group_keys=False)\
        .apply(lambda x: x.sample(min(len(x), 5500)))

X_train = df_train.drop("quality", axis=1)
y_train = pd.Categorical(df_train["quality"], ordered=True)

### Impute outliers ###

from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=2)
knn_inputed = knn_imputer.fit(X_train)
 
def knn_impute_outliers(df, inputer, threash=0.98):
     new = np.where( (df > (np.quantile(df, threash, axis=0)) ) | ( df < (np.quantile(df, 1-threash, axis=0)) ), np.nan, df)
     return pd.DataFrame(inputer.fit_transform(new), columns=df.columns)

X_train = knn_impute_outliers(X_train, knn_inputed)

models = []

### Grouping ###

### Logistic Reg ###

multi_logistic_reg = LogisticRegression(penalty="elasticnet", solver="saga", tol=1e-2, max_iter=500)

params_logistic = {"C": np.logspace(-3, 0, 250), "penalty": ["l1", "l2"]}

logistic_grid_search = model_selection.GridSearchCV(
    multi_logistic_reg, params_logistic, n_jobs=cores)
# Not all of these converge given the low tolerance I set above

if retrain:
    logistic_grid_search.fit(X_train, y_train)
    joblib.dump(logistic_grid_search, f"{model_path}logistc_grid_search.dat")
    

logistic_grid_search = joblib.load(f"{model_path}logistc_grid_search.dat")
models.append(("logistic_reg", logistic_grid_search))

### XGB ###

params_xgb = {'max_depth': np.arange(3, 15, 1 ),
        'colsample_bytree' : np.arange(0.2, 0.6, 0.1),
        "learning_rate": np.logspace(-3, -1, 9)}

xgboost = xgb.XGBClassifier()

xgb_grid_search = model_selection.GridSearchCV(
    xgboost, params_xgb, n_jobs=cores
)

if retrain:
    xgb_grid_search.fit(X_train, y_train)
    joblib.dump(xgb_grid_search, f"{model_path}xgb_grid_search.dat")

xgb_grid_search = joblib.load(f"{model_path}xgb_grid_search.dat")
print(xgb_grid_search.best_params_)
models.append(("xgb", xgb_grid_search))

### SVM ###

params_svm = {"C": np.logspace(0, 2, 20)}
support_vector_machine = svm.SVC(gamma='auto',probability=True)
svm_grid_search = model_selection.GridSearchCV(support_vector_machine, params_svm, n_jobs=cores)

if retrain:
    svm_grid_search.fit(X_train, y_train)
    joblib.dump(svm_grid_search, f"{model_path}svm_gridsearch.dat")
    
svm_grid_search = joblib.load(f"{model_path}svm_gridsearch.dat")
print(svm_grid_search.best_params_)
models.append(("s", svm_grid_search))

from sklearn import metrics
df_test = pd.DataFrame(df_transform.fit_transform(df_test), columns=df.columns)
X_test = df_test.drop("quality", axis=1)
X_train = knn_impute_outliers(X_train, knn_inputed)
y_test = pd.Categorical(df_test["quality"], ordered=True)


### Metrics ###


def num(x):
    return pd.to_numeric(x)
### Test Data ###
auc_list = []
accuracy_list = []
rmse_list = []
yhat = xgb_grid_search.predict(X_test)
phat = xgb_grid_search.predict_proba(X_test)
rmse = metrics.mean_squared_error(num(yhat), num(y_test), squared=False)
rmse_list.append(rmse)

auc = metrics.roc_auc_score(y_test, phat, multi_class='ovr')
accuracy = metrics.accuracy_score(y_test, yhat)

auc_list.append(auc)
accuracy_list.append(accuracy)
print("xgb  test")
print("auc", auc)
print("accuracy", accuracy)

phat = logistic_grid_search.predict_proba(X_test)
yhat = logistic_grid_search.predict(X_test)

auc = metrics.roc_auc_score(y_test, phat, multi_class='ovr')
accuracy = metrics.accuracy_score(y_test, yhat)
rmse = metrics.mean_squared_error(num(yhat), num(y_test), squared=False)
rmse_list.append(rmse)
auc_list.append(auc)
accuracy_list.append(accuracy)
print("logistic  testing")
print("auc", auc)
print("accuracy", accuracy)

svm_grid_search 

phat = svm_grid_search.predict_proba(X_test)
yhat = logistic_grid_search.predict(X_test)
rmse = metrics.mean_squared_error(num(yhat), num(y_test), squared=False)
rmse_list.append(rmse)
auc = metrics.roc_auc_score(y_test, phat, multi_class='ovr')
accuracy = metrics.accuracy_score(y_test, yhat)

auc_list.append(auc)
accuracy_list.append(accuracy)

print("svm  testing")
print("auc", auc)
print("accuracy", accuracy)

### Ensemble ###

if train_ensemble:

    from sklearn.ensemble import VotingClassifier
    
    ensemble = VotingClassifier(models, voting="soft", n_jobs=cores, weights=[1, 3, 1])
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, f"{model_path}ensemble.dat")
    ensemble = joblib.load(f"{model_path}ensemble.dat")
    yhat = ensemble.predict(X_train)
    phat = ensemble.predict_proba(X_train)
    auc = metrics.roc_auc_score(y_train, phat, multi_class='ovr')
    accuracy = metrics.accuracy_score(y_train, yhat)

    print("training set")
    print("auc", auc)
    print("accuracy", accuracy)

    yhat = ensemble.predict(X_test)
    phat = ensemble.predict_proba(X_test)
    auc = metrics.roc_auc_score(y_test, phat, multi_class='ovr')
    accuracy = metrics.accuracy_score(y_test, yhat)
    auc_list.append(auc)
    accuracy_list.append(accuracy)
    rmse = metrics.mean_squared_error(num(yhat), num(y_test), squared=False)
    rmse_list.append(rmse)
    print("final test")
    print("auc", auc)
    print("accuracy", accuracy)

df = X_test
df["quality"] = y_test
df["prediction"] = yhat
df.to_csv("graphics/graphics_data/predictions.csv")
from beepy import beep
beep()
end = time.time()
print(f"runtime {(end - start)/60}")

pd.DataFrame([["XGBoost", "Elastic Net", "SVM", "Ensemble"],
    accuracy_list,
    auc_list,
    rmse_list])\
        .to_csv("graphics/graphics_data/accuracy_table.csv") 



