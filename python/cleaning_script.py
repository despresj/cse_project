import pandas as pd
import numpy as np
red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red_df = pd.read_csv(red_url, sep=";")
white_df = pd.read_csv(white_url, sep=";")
red_df['is_red'] = 1
white_df['is_red'] = 0
df_raw = pd.concat([red_df, white_df])
df_raw.columns = [x.lower().replace(" ", "_") for x in df_raw.columns]
df_raw = df_raw.groupby(df_raw.columns.tolist(), as_index=False).size()
df_raw.rename(columns = {'size':'duplicates'}, inplace = True)
y = df_raw.quality
X = df_raw.drop("quality", axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

from sklearn import preprocessing
already_scaled = X_train[["is_red", "duplicates"]].to_numpy()
to_scale = X_train.drop(["is_red", "duplicates"], axis=1)
scaler = preprocessing.StandardScaler().fit(to_scale)
scaled = scaler.transform(to_scale)

# anything 3x larger than scaled iqr is gona get replaced 
# by a knn inpute
upper_quartile = np.quantile(scaled, 0.75, axis=0) * 3
scaled = np.where(scaled > upper_quartile, np.nan, scaled)
lower_quartile = np.quantile(scaled, 0.25, axis=0) * 3
scaled = np.where(scaled < lower_quartile, np.nan, scaled)

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
inputed = imputer.fit_transform(scaled)
X_train = np.concatenate((inputed, already_scaled), axis=1)
# im gona regret not putting this in a function
test_scaled = X_test[["is_red", "duplicates"]].to_numpy()
test_to_scale = X_test.drop(["is_red", "duplicates"], axis=1)
test_scaled = scaler.transform(test_to_scale)
test_scaled = np.where(test_scaled > upper_quartile, np.nan, test_scaled)
test_scaled = np.where(test_scaled < lower_quartile, np.nan, test_scaled)
test_inputed = imputer.fit_transform(test_scaled)
X_test = np.concatenate((test_inputed, test_scaled), axis=1)

np.savez("data/processed/train_test_scale_1.npz", 
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test)