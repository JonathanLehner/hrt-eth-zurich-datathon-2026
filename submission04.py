import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# Load data
# =========================================================
bars_seen_train   = pd.read_parquet("data/bars_seen_train.parquet")
bars_seen_public  = pd.read_parquet("data/bars_seen_public_test.parquet")
bars_seen_private = pd.read_parquet("data/bars_seen_private_test.parquet")
bars_all_seen     = pd.concat([bars_seen_train, bars_seen_public, bars_seen_private], ignore_index=True)

bars_unseen_train = pd.read_parquet("data/bars_unseen_train.parquet").sort_values(["session", "bar_ix"])
target = bars_unseen_train.groupby("session").apply(
    lambda x: x["close"].iloc[-1] / x["open"].iloc[0] - 1
).rename("target").reset_index()

sentiment = pd.read_csv("analysis/company-ids.csv")

# =========================================================
# Seen return per session (used for sentiment filtering)
# =========================================================
seen_return = (
    bars_all_seen.sort_values(["session", "bar_ix"])
    .groupby("session")
    .apply(lambda x: x["close"].iloc[-1] / x["close"].iloc[0] - 1)
    .rename("seen_return")
    .reset_index()
)

# =========================================================
# Filter sentiment per session: company IDs are arbitrary
# local labels — the same ID means different companies in
# different sessions. So filter within each session:
# discard a company's headlines if its average shortTerm
# sentiment sign contradicts the session's seen return sign.
# Those companies' news is clearly unrelated to this stock.
# =========================================================
sent_with_seen = sentiment.merge(seen_return, on="session", how="left")

# per (session, company): mean sentiment
per_company = (
    sent_with_seen
    .groupby(["session", "companyId"])[["shortTerm", "longTerm", "seen_return"]]
    .mean()
    .reset_index()
)

# keep company if its shortTerm sign matches seen_return sign,
# or if seen_return is near zero (no directional signal to filter on)
seen_nonzero = per_company["seen_return"].abs() > 1e-6
aligned = np.sign(per_company["shortTerm"]) == np.sign(per_company["seen_return"])
per_company_filtered = per_company[~seen_nonzero | aligned]

# aggregate filtered companies into session-level sentiment features
sent_features = (
    per_company_filtered
    .groupby("session")[["shortTerm", "longTerm"]]
    .mean()
    .rename(columns={"shortTerm": "sent_short", "longTerm": "sent_long"})
    .reset_index()
)


# =========================================================
# Build price-based features per session
# =========================================================
def build_price_features(bars_df):
    bars_df = bars_df.sort_values(["session", "bar_ix"]).copy()

    return_seen = bars_df.groupby("session").apply(
        lambda x: x["close"].iloc[-1] / x["close"].iloc[0] - 1
    ).rename("return_seen")

    recent_return = bars_df.groupby("session").apply(
        lambda x: x.tail(5)["close"].pct_change().mean()
    ).rename("recent_return")

    return pd.concat([return_seen, recent_return], axis=1).reset_index()


SENT_COLS = ["sent_short", "sent_long"]
FEATURES  = ["return_seen", "recent_return", "sent_short", "sent_long"]


def merge_features(bars_df):
    return (
        build_price_features(bars_df)
        .merge(sent_features, on="session", how="left")
        .fillna({c: 0.0 for c in SENT_COLS})
    )


# =========================================================
# 1) TRAIN — small regularised RF to avoid overfitting
#    1000 samples → keep model shallow and trees few
# =========================================================
train_features = merge_features(bars_seen_train)
train_data = target[["session", "target"]].merge(train_features, on="session", how="left").dropna()

X_train = train_data[FEATURES].values
y_train = train_data["target"].values

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=3,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

importances = dict(zip(FEATURES, model.feature_importances_))
print("feature importances:", {k: round(v, 4) for k, v in sorted(importances.items(), key=lambda x: -x[1])})


# =========================================================
# Position sizing: scale by inverse of tree-prediction std
# =========================================================
def predict_with_sizing(model, X, k=10.0):
    tree_preds = np.array([t.predict(X) for t in model.estimators_])
    pred_mean = tree_preds.mean(axis=0)
    pred_std  = tree_preds.std(axis=0)
    return pred_mean / (1.0 + k * pred_std)


# =========================================================
# 2) PUBLIC TEST
# =========================================================
public_features = merge_features(bars_seen_public)
public_features["target_position"] = predict_with_sizing(model, public_features[FEATURES].values)
submission_public = public_features[["session", "target_position"]].copy()

# =========================================================
# 3) PRIVATE TEST
# =========================================================
private_features = merge_features(bars_seen_private)
private_features["target_position"] = predict_with_sizing(model, private_features[FEATURES].values)
submission_private = private_features[["session", "target_position"]].copy()

# =========================================================
# 4) FINAL SUBMISSION
# =========================================================
submission_final = (
    pd.concat([submission_public, submission_private], ignore_index=True)
    .sort_values("session")
    .reset_index(drop=True)
)

print(submission_final.head())
print(submission_final.tail())
print(submission_final.shape)

submission_final.to_csv("submission04.csv", index=False)
print("saved submission04.csv")
