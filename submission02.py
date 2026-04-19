import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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
#
# Weight each headline by 1/(uncertainty+1) so low-confidence
# signals contribute less. Compute both full-window and
# recent (last-10-bar) weighted averages of short/long-term.
# =========================================================
def build_sentiment_features(sentiment_df, bar_cutoff=40):
    df = sentiment_df.copy()
    df["w_short"] = 1.0 / (df["shortTermUncertainty"] + 1)
    df["w_long"]  = 1.0 / (df["longTermUncertainty"]  + 1)

    def wavg(vals, weights):
        return (vals * weights).sum() / weights.sum() if weights.sum() > 0 else 0.0

    def session_feats(g):
        recent = g[g["barIx"] >= bar_cutoff]
        return pd.Series({
            "sent_short":        wavg(g["shortTerm"],        g["w_short"]),
            "sent_long":         wavg(g["longTerm"],         g["w_long"]),
            "sent_short_recent": wavg(recent["shortTerm"],   recent["w_short"]) if len(recent) else 0.0,
            "sent_long_recent":  wavg(recent["longTerm"],    recent["w_long"])  if len(recent) else 0.0,
        })

    return df.groupby("session").apply(session_feats).reset_index()


# =========================================================
# helper: build the price-based features per session
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

    return pd.concat([return_seen, recent_return, oc_change], axis=1).reset_index()


FEATURES = ["return_seen", "recent_return", "oc_change",
            "sent_short", "sent_long", "sent_short_recent", "sent_long_recent"]

sent_features = build_sentiment_features(sentiment)
SENT_COLS = ["sent_short", "sent_long", "sent_short_recent", "sent_long_recent"]

# =========================================================
# 1) TRAIN
# =========================================================
train_features = build_feature_table(bars_seen_train).merge(sent_features, on="session", how="left").fillna({c: 0.0 for c in SENT_COLS})

train_data = data[["session", "target"]].merge(train_features, on="session", how="left").dropna()

X_train = train_data[FEATURES]
y_train = train_data["target"]

model = LinearRegression()
model.fit(X_train, y_train)

print("coefficients:", dict(zip(FEATURES, model.coef_)))


# =========================================================
# 2) PUBLIC TEST predictions
# =========================================================
public_features = build_feature_table(bars_seen_public_test).merge(sent_features, on="session", how="left").fillna({c: 0.0 for c in SENT_COLS})
public_features["target_position"] = model.predict(public_features[FEATURES])
submission_public = public_features[["session", "target_position"]].copy()


# =========================================================
# 3) PRIVATE TEST predictions
# =========================================================
private_features = build_feature_table(bars_seen_private_test).merge(sent_features, on="session", how="left").fillna({c: 0.0 for c in SENT_COLS})
private_features["target_position"] = model.predict(private_features[FEATURES])
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

submission_final.to_csv("submission02.csv", index=False)
print("saved submission02.csv")
