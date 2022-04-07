# # Feature Engineering Testing
# In this part, we will take a look at our variables and determine which feature engineering methods prep our data in a way that is easier for our models to interpret.

import pandas as pd

# red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
# white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
# red_df = pd.read_csv(red_url, sep=";")
# white_df = pd.read_csv(white_url, sep=";")
# red_df["is_red"] = 1
# white_df["is_red"] = 0
# df_raw = pd.concat([red_df, white_df])
# df_raw.columns = [x.replace(" ", "_") for x in df_raw.columns]
# df_raw.to_csv("../data/processed/wine_data_combined.csv", index=False)

df_raw = pd.read_csv("../data/processed/wine_data_combined.csv")
cols_to_adjust = [x for x in df_raw.columns if x not in ["quality", "is_red"]]

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df_train_raw, df_test_raw = train_test_split(
    df_raw, test_size=0.3, random_state=55, stratify=df_raw["quality"]
)


def get_auc(
    df, scaler=None, target="quality", cols_adj=cols_to_adjust, n_iters=1, split=0.4
):

    auc_list = []
    for seed in range(n_iters):
        X = df.drop(target, axis=1)
        y = pd.Categorical(df[target], ordered=True)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=split, random_state=seed, stratify=y
        )

        multi_logit_reg = LogisticRegression(max_iter=5000, solver="saga")

        if scaler is not None:
            df_transform = ColumnTransformer(
                [(" ", scaler, cols_adj)], remainder="passthrough"
            )
            pipeline = Pipeline(steps=[("t", df_transform), ("m", multi_logit_reg)])
            model_fit = pipeline.fit(X_train, y_train)
        else:
            model_fit = multi_logit_reg.fit(X_train, y_train)

        phat = model_fit.predict_proba(X_val)
        auc_list.append(roc_auc_score(y_val, phat, multi_class="ovo"))
    return auc_list


baseline_results = get_auc(df_train_raw)

std_scale_results = get_auc(df_train_raw, preprocessing.StandardScaler())

min_max_results = get_auc(df_train_raw, preprocessing.MinMaxScaler())

robscale_results = get_auc(
    df_train_raw, preprocessing.RobustScaler(quantile_range=(0.2, 0.8))
)  # Drops outliers

pwr_transform_results = get_auc(
    df_train_raw, preprocessing.PowerTransformer()
)  #  yao johnson

quant_transfmr_results = get_auc(
    df_train_raw, preprocessing.QuantileTransformer()
)  # transforms data to unf(0,1)

res_zip = zip(
    baseline_results,
    std_scale_results,
    min_max_results,
    robscale_results,
    pwr_transform_results,
    quant_transfmr_results,
)

results = pd.DataFrame(
    res_zip,
    columns=[
        "baseline_results",
        "std_scale_results",
        "min_max_results",
        "robscale_results",
        "pwr_transform_results",
        "quant_transfmr_results",
    ],
)\
    # .to_csv("../result_logs/transformations.csv")
