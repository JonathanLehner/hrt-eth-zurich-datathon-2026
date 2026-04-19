import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# Load data
# =========================================================
bars_seen_train = pd.read_parquet("data/bars_seen_train.parquet")
bars_seen_public_test = pd.read_parquet("data/bars_seen_public_test.parquet")
bars_seen_private_test = pd.read_parquet("data/bars_seen_private_test.parquet")

bars_unseen_train = pd.read_parquet("data/bars_unseen_train.parquet").sort_values(["session", "bar_ix"])
data = bars_unseen_train.groupby("session").apply(
    lambda x: x["close"].iloc[-1] / x["open"].iloc[0] - 1
).rename("target").reset_index()

sentiment = pd.read_csv("analysis/company-ids.csv")

# =========================================================
# helper: build sentiment features per session
# Simple means — no uncertainty weighting
# =========================================================
def build_sentiment_features(sentiment_df, bar_cutoff=40):
    def session_feats(g):
        recent = g[g["barIx"] >= bar_cutoff]
        return pd.Series({
            "sent_short":        g["shortTerm"].mean(),
            "sent_long":         g["longTerm"].mean(),
            "sent_short_recent": recent["shortTerm"].mean() if len(recent) else 0.0,
            "sent_long_recent":  recent["longTerm"].mean()  if len(recent) else 0.0,
            "sent_n_headlines":  len(g),
        })
    return sentiment_df.groupby("session").apply(session_feats).reset_index()


# =========================================================
# helper: build price-based features per session
# =========================================================
def build_feature_table(bars_df):
    bars_df = bars_df.sort_values(["session", "bar_ix"]).copy()

    return_seen = bars_df.groupby("session").apply(
        lambda x: x["close"].iloc[-1] / x["close"].iloc[0] - 1
    ).rename("return_seen")

    recent_return = bars_df.groupby("session").apply(
        lambda x: x.tail(5)["close"].pct_change().mean()
    ).rename("recent_return")

    bars_df["oc_change_bar"] = (bars_df["close"] - bars_df["open"]) / bars_df["open"]
    oc_change = bars_df.groupby("session").apply(
        lambda x: x["oc_change_bar"].tail(5).mean()
    ).rename("oc_change")

    volatility = bars_df.groupby("session").apply(
        lambda x: x["close"].pct_change().std()
    ).rename("volatility")

    return pd.concat([return_seen, recent_return, oc_change, volatility], axis=1).reset_index()


FEATURES = ["return_seen", "recent_return", "oc_change", "volatility",
            "sent_short", "sent_long", "sent_short_recent", "sent_long_recent", "sent_n_headlines"]
SENT_COLS = ["sent_short", "sent_long", "sent_short_recent", "sent_long_recent", "sent_n_headlines"]

sent_features = build_sentiment_features(sentiment)


def merge_features(bars_df):
    return (
        build_feature_table(bars_df)
        .merge(sent_features, on="session", how="left")
        .fillna({c: 0.0 for c in SENT_COLS})
    )


# =========================================================
# 1) TRAIN
# =========================================================
train_features = merge_features(bars_seen_train)
train_data = data[["session", "target"]].merge(train_features, on="session", how="left").dropna()

X_train = train_data[FEATURES].values
y_train = train_data["target"].values

model = RandomForestRegressor(n_estimators=500, max_features="sqrt", min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

importances = dict(zip(FEATURES, model.feature_importances_))
print("feature importances:", {k: round(v, 4) for k, v in sorted(importances.items(), key=lambda x: -x[1])})


# =========================================================
# Position sizing using prediction uncertainty
#
# Each tree in the forest gives an independent prediction.
# The std across trees is a natural measure of model uncertainty.
# We scale the position by 1 / (1 + k * std), where k is chosen
# so that the scaling is meaningful relative to typical std values.
# =========================================================
def predict_with_sizing(model, X, k=10.0):
    tree_preds = np.array([t.predict(X) for t in model.estimators_])  # (n_trees, n_samples)
    pred_mean = tree_preds.mean(axis=0)
    pred_std  = tree_preds.std(axis=0)
    confidence = 1.0 / (1.0 + k * pred_std)
    position = pred_mean * confidence
    return position


# =========================================================
# 2) PUBLIC TEST predictions
# =========================================================
public_features = merge_features(bars_seen_public_test)
public_features["target_position"] = predict_with_sizing(model, public_features[FEATURES].values)
submission_public = public_features[["session", "target_position"]].copy()


# =========================================================
# 3) PRIVATE TEST predictions
# =========================================================
private_features = merge_features(bars_seen_private_test)
private_features["target_position"] = predict_with_sizing(model, private_features[FEATURES].values)
submission_private = private_features[["session", "target_position"]].copy()


# =========================================================
# 4) FINAL SUBMISSION = public + private
# =========================================================
submission_final = pd.concat(
    [submission_public, submission_private], axis=0, ignore_index=True
).sort_values("session").reset_index(drop=True)

print(submission_final.head())
print(submission_final.tail())
print(submission_final.shape)

submission_final.to_csv("submission03.csv", index=False)
print("saved submission03.csv")
