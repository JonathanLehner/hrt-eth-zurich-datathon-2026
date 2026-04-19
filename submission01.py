import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================================================
# FINAL SUBMISSION CODE
# Hybrid aggregation:
# value = weight_median * median + (1 - weight_median) * mean
#
# =========================================================
weight_median = 0.7


# =========================================================
# helper: hybrid aggregate
# =========================================================
def hybrid_agg(series, weight_median=0.7):
    s = series.dropna()
    return weight_median * s.median() + (1 - weight_median) * s.mean()


# =========================================================
# helper: build feature table
# =========================================================
def build_feature_table(bars_df, weight_median=0.7):
    bars_df = bars_df.sort_values(["session", "bar_ix"]).copy()

    # return_seen stays unchanged
    return_seen = bars_df.groupby("session").apply(
        lambda x: x["close"].iloc[-1] / x["close"].iloc[0] - 1
    ).rename("return_seen")

    # recent_return = hybrid of last 5 close pct changes
    recent_return = bars_df.groupby("session").apply(
        lambda x: hybrid_agg(
            x.tail(5)["close"].pct_change(),
            weight_median=weight_median
        )
    ).rename("recent_return")

    # oc_change_bar
    bars_df["oc_change_bar"] = (
        (bars_df["close"] - bars_df["open"]) / bars_df["open"]
    )

    # oc_change = hybrid of last 5 oc_change bars
    oc_change = bars_df.groupby("session").apply(
        lambda x: hybrid_agg(
            x["oc_change_bar"].tail(5),
            weight_median=weight_median
        )
    ).rename("oc_change")

    feature_table = pd.concat(
        [return_seen, recent_return, oc_change],
        axis=1
    ).reset_index()

    return feature_table


# =========================================================
# 1) TRAIN ON FULL TRAIN DATA
# =========================================================
train_features = build_feature_table(
    bars_seen_train,
    weight_median=weight_median
)

train_base = data.reset_index()

train_data = train_base[["session", "target"]].merge(
    train_features,
    on="session",
    how="left"
).dropna()

X_train_full = train_data[["return_seen", "recent_return", "oc_change"]]
y_train_full = train_data["target"]

model = LinearRegression()
model.fit(X_train_full, y_train_full)

print("weight_median:", weight_median)
print("coefficients:", dict(zip(X_train_full.columns, model.coef_)))


# =========================================================
# 2) PUBLIC TEST FEATURES + PREDICT
# =========================================================
public_features = build_feature_table(
    bars_seen_public_test,
    weight_median=weight_median
)

X_public = public_features[["return_seen", "recent_return", "oc_change"]]
public_features["target_position"] = model.predict(X_public)

submission_public = public_features[["session", "target_position"]].copy()


# =========================================================
# 3) PRIVATE TEST FEATURES + PREDICT
# =========================================================
private_features = build_feature_table(
    bars_seen_private_test,
    weight_median=weight_median
)

X_private = private_features[["return_seen", "recent_return", "oc_change"]]
private_features["target_position"] = model.predict(X_private)

submission_private = private_features[["session", "target_position"]].copy()


# =========================================================
# 4) FINAL SUBMISSION
# public first, then private
# =========================================================
submission = pd.concat(
    [submission_public, submission_private],
    axis=0,
    ignore_index=True
)

print(submission.head())
print(submission.tail())
print("submission shape:", submission.shape)

submission.to_csv("submission.csv", index=False)
print("saved submission.csv")
