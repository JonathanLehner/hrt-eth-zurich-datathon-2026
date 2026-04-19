import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================================================
# helper: build the 3 features return_seen, recent_return, oc_change
# =========================================================
def build_feature_table(bars_df):
    bars_df = bars_df.sort_values(["session", "bar_ix"]).copy()

    # return_seen
    return_seen = bars_df.groupby("session").apply(
        lambda x: x["close"].iloc[-1] / x["close"].iloc[0] - 1
    ).rename("return_seen")

    # recent_return
    recent_return = bars_df.groupby("session").apply(
        lambda x: x.tail(5)["close"].pct_change().mean()
    ).rename("recent_return")

    # oc_change
    bars_df["oc_change_bar"] = (
        (bars_df["close"] - bars_df["open"]) / bars_df["open"]
    )

    oc_change = bars_df.groupby("session").apply(
        lambda x: x["oc_change_bar"].tail(5).mean()
    ).rename("oc_change")

    feature_table = pd.concat(
        [return_seen, recent_return, oc_change],
        axis=1
    ).reset_index()

    return feature_table


# =========================================================
# 1) TRAIN on all labeled training data
#    data is indexed by session and has target
# =========================================================
train_features = build_feature_table(bars_seen_train)

train_base = data.reset_index()   # brings session out of the index

train_data = train_base[["session", "target"]].merge(
    train_features,
    on="session",
    how="left"
).dropna()

X_train_full = train_data[["return_seen", "recent_return", "oc_change"]]
y_train_full = train_data["target"]

model = LinearRegression()
model.fit(X_train_full, y_train_full)

print("coefficients:", dict(zip(X_train_full.columns, model.coef_)))


# =========================================================
# 2) PUBLIC TEST predictions
# =========================================================
public_features = build_feature_table(bars_seen_public_test)

X_public = public_features[["return_seen", "recent_return", "oc_change"]]
public_features["target_position"] = model.predict(X_public)

submission_public = public_features[["session", "target_position"]].copy()


# =========================================================
# 3) PRIVATE TEST predictions
# =========================================================
private_features = build_feature_table(bars_seen_private_test)

X_private = private_features[["return_seen", "recent_return", "oc_change"]]
private_features["target_position"] = model.predict(X_private)

submission_private = private_features[["session", "target_position"]].copy()


# =========================================================
# 4) FINAL SUBMISSION = public + private
# =========================================================
submission_final = pd.concat(
    [submission_public, submission_private],
    axis=0,
    ignore_index=True
)

# sort by session just to keep it neat
submission_final = submission_final.sort_values("session").reset_index(drop=True)

print(submission_final.head())
print(submission_final.tail())
print(submission_final.shape)

submission_final.to_csv("submission.csv", index=False)
print("saved submission.csv")
