import pandas as pd
import numpy as np
import joblib
from sklearn import model_selection
import xgboost as xgb
model_path = "saved_models/"

df = pd.read_csv("data/processed/wine_data_combined.csv")
cols_to_adjust = [x for x in df.columns if x not in ["quality", "is_red"]]

df_train, df_test = model_selection.train_test_split(
     df, test_size=0.3, random_state=55, stratify=df["quality"]
 )

X_train = df_train.drop("quality", axis=1)
X_test = df_test.drop("quality", axis=1)
y_train = pd.Categorical(df_train["quality"], ordered=True)
y_test = pd.Categorical(df_test["quality"], ordered=True)

pipeline = joblib.load(f"{model_path}pipeline.dat")
phat = pipeline.predict_proba(X_train)
yhat = pipeline.predict(X_train)

from sklearn import metrics
print(X_train)
print(X_test)
print("accuracy",np.sum(yhat == y_train)/len(y_train))
auc = metrics.roc_auc_score(y_train, phat, multi_class='ovr')
print("auc", auc)
conf = metrics.confusion_matrix(y_train, yhat)
print("confusion matrix", conf) 
phat = pipeline.predict_proba(X_test)
yhat = pipeline.predict(X_test)
print("final accuracy",np.sum(yhat == y_test)/len(y_test))
auc = metrics.roc_auc_score(y_test, phat, multi_class='ovr')
print("final auc", auc)
conf = metrics.confusion_matrix(y_test, yhat)
print("confusion matrix", conf) 
xgb_grid_search = joblib.load(f"{model_path}xgb_grid_search.dat")

yhat = xgb_grid_search.predict(X_train)
print("xgb final train accuracy",np.sum(yhat == y_train)/len(y_train))
print(yhat)

yhat = xgb_grid_search.predict(X_test)
print("xgb final accuracy",np.sum(yhat == y_test)/len(y_test))
print(yhat)
print(type(yhat))