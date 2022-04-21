import time
import pandas as pd
import numpy as np
import joblib
import multiprocessing
from sklearn import model_selection
import glob

retrain = True # Set to True and rerun the model gridsearches
train_ensemble = True

start = time.time()

list_of_df = []
for file in glob.glob("data/vivino/"+ "/*csv"):
    df = pd.read_csv(file, index_col=None, header=0)
    df["wine_type"] = file.replace("../data/vivino/", "").replace(".csv", "")
    list_of_df.append(df)

frame = pd.concat(list_of_df, axis=0, ignore_index=True)

frame = frame.drop("Winery", axis=1)
frame = frame.drop("Region", axis=1)
df = pd.get_dummies(frame)
print(frame.shape)
from re import sub
def snake_case(s):
  return '_'.join(
    sub('([A-Z][a-z]+)', r' \1',
    sub('([A-Z]+)', r' \1',
    s.replace('-', ' ').replace("'", ""))).split()).lower()

df.columns = [snake_case(x) for x in df.columns]

df.rename(columns={"rating":"quality"}, inplace=True)
cols_to_adjust = ["quality", "number_of_ratings", "price"] 

model_path = "saved_models/_yao_transform_price_model"

cores = multiprocessing.cpu_count()
cores -= 2 # so i can do other stuff when this trains

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score

df_train, df_test = model_selection.train_test_split(
        df, test_size=0.2, random_state=66
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

df_transform = ColumnTransformer(
    [("yao", PowerTransformer(), cols_to_adjust)],
    remainder="passthrough",
)

df_train = pd.DataFrame(df_transform.fit_transform(df_train), columns=df.columns)

X_train = df_train.drop("quality", axis=1)
y_train = pd.Categorical(df_train["quality"], ordered=True)

### Impute outliers ###

models = []

### Grouping ###

###  Reg ###
# from sklearn.linear_model import ElasticNet
# elastic_net = ElasticNet()
# params_ = {"max_iter": [10],
#                       "alpha": [1],
#                       "l1_ratio": np.arange(0.0, 1.0, 0.5)}
# _grid_search = model_selection.GridSearchCV(
#     elastic_net, params_, n_jobs=cores)
# # Not all of these converge given the low tolerance I set above

# if retrain:
#     _grid_search.fit(X_train, y_train)
#     joblib.dump(_grid_search, f"{model_path}logistc_grid_search.dat")
    

# _grid_search = joblib.load(f"{model_path}logistc_grid_search.dat")
# models.append(("_reg", _grid_search))

### XGB ###

params_xgb = {'max_depth': np.arange(3, 15, 1 ),
        'colsample_bytree' : np.arange(0.2, 0.6, 0.1),
        "learning_rate": np.logspace(-3, -1, 9)}

xgboost = xgb.XGBRegressor()

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
models.append(("svm", svm_grid_search))

from sklearn import metrics
df_test = pd.DataFrame(df_transform.fit_transform(df_test), columns=df.columns)
X_test = df_test.drop("quality", axis=1)
y_test = pd.Categorical(df_test["quality"], ordered=True)


### Metrics ###
### Test Data ###
auc_list = []
accuracy_list = []

yhat = xgb_grid_search.predict(X_test)
phat = xgb_grid_search.predict_proba(X_test)

auc = metrics.roc_auc_score(y_test, phat, multi_class='ovr')
accuracy = metrics.accuracy_score(y_test, yhat)
from sklearn.metrics import mean_squared_error

rmse = metrics.mean_squared_error(yhat, ytest, squared=False)

auc_list.append(auc)
accuracy_list.append(accuracy)
print("xgb  test")
print("auc", auc)
print("accuracy", accuracy)

phat = _grid_search.predict_proba(X_test)
yhat = _grid_search.predict(X_test)

auc = metrics.roc_auc_score(y_test, phat, multi_class='ovr')
accuracy = metrics.accuracy_score(y_test, yhat)
auc_list.append(auc)
accuracy_list.append(accuracy)
print("  testing")
print("auc", auc)
print("accuracy", accuracy)

svm_grid_search 

phat = svm_grid_search.predict_proba(X_test)
yhat = _grid_search.predict(X_test)

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
    auc_list])\
        .to_csv("graphics/graphics_data/accuracy_table.csv") 



