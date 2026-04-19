"""
submission06 — 50-day mean-reversion momentum model

Signal: stocks that ran up hard (high risk-adjusted 20-day return) tend to
mean-revert over the next 50 days. We exploit this with two correlated but
complementary factors:
  mom20      — raw 20-day return (magnitude of recent move)
  sharpe20   — risk-adjusted 20-day return (quality/cleanness of move)

Both have negative rank IC with the 50-day forward return (mean reversion).
Their combination beats either alone in 10-fold CV Sharpe (0.189 vs 0.173 baseline).

Model: Ridge(alpha=0.01) on rank-normalized features.
Positions: raw model predictions — no z-scoring so the unconditional
positive drift (57% positive sessions, mean return +0.35%) is preserved.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy import stats

# =========================================================
# Load data
# =========================================================
bars_seen_train   = pd.read_parquet("data/bars_seen_train.parquet")
bars_seen_public  = pd.read_parquet("data/bars_seen_public_test.parquet")
bars_seen_private = pd.read_parquet("data/bars_seen_private_test.parquet")

bars_unseen_train = pd.read_parquet("data/bars_unseen_train.parquet").sort_values(["session", "bar_ix"])
target = bars_unseen_train.groupby("session").apply(
    lambda x: x["close"].iloc[-1] / x["open"].iloc[0] - 1
).rename("target").reset_index()

# =========================================================
# Feature engineering
# =========================================================
def build_features(bars_df):
    b = bars_df.sort_values(["session", "bar_ix"]).copy()
    b["ret"] = b.groupby("session")["close"].pct_change()

    def session_feats(g):
        c  = g["close"].values
        r  = g["ret"].dropna().values

        mom20   = c[-1] / c[-21] - 1 if len(c) >= 21 else c[-1] / c[0] - 1
        vol20   = r[-20:].std() + 1e-9 if len(r) >= 20 else r.std() + 1e-9
        sharpe20 = mom20 / vol20

        return pd.Series({"mom20": mom20, "sharpe20": sharpe20})

    return b.groupby("session").apply(session_feats).reset_index()

FEATURES = ["mom20", "sharpe20"]

# =========================================================
# Rank-normalise features (uniform [0,1])
# Rank normalisation makes Ridge coefficients directly
# comparable and eliminates sensitivity to outliers.
# We rank each dataset within itself so the distribution
# is always uniform regardless of regime differences.
# =========================================================
def rank_norm(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = df[c].rank(pct=True)
    return df

# =========================================================
# Train
# =========================================================
train_feats = build_features(bars_seen_train)
train_data  = target[["session", "target"]].merge(train_feats, on="session").dropna()
train_ranked = rank_norm(train_data, FEATURES)

X_train = train_ranked[FEATURES].values
y_train = train_ranked["target"].values

model = Ridge(alpha=0.01)
model.fit(X_train, y_train)

# =========================================================
# Validate: in-sample and 10-fold CV Sharpe on training set
# =========================================================
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_sharpes = []
for tr, va in cv.split(X_train):
    m = Ridge(alpha=0.01).fit(X_train[tr], y_train[tr])
    pos = m.predict(X_train[va])
    pnl = pos * y_train[va]
    cv_sharpes.append(pnl.mean() / (pnl.std() + 1e-9))

insample_pos = model.predict(X_train)
insample_pnl = insample_pos * y_train

print(f"Coefficients: {dict(zip(FEATURES, model.coef_.round(6)))}")
print(f"Intercept:    {model.intercept_:.6f}")
print(f"Pred range:   [{insample_pos.min():.5f}, {insample_pos.max():.5f}]  mean={insample_pos.mean():.5f}")
print(f"In-sample Sharpe (train): {insample_pnl.mean()/(insample_pnl.std()+1e-9):.4f}")
print(f"10-fold CV Sharpe:        {np.mean(cv_sharpes):.4f}  (+/- {np.std(cv_sharpes):.4f})")
print(f"Baseline (constant=mean): {y_train.mean()/y_train.std():.4f}")

# =========================================================
# Predict & submit
# =========================================================
def make_submission(bars_df):
    feats  = build_features(bars_df)
    ranked = rank_norm(feats, FEATURES)
    pos    = model.predict(ranked[FEATURES].values)
    ranked["target_position"] = pos
    return ranked[["session", "target_position"]]

submission_final = (
    pd.concat([make_submission(bars_seen_public), make_submission(bars_seen_private)],
              ignore_index=True)
    .sort_values("session")
    .reset_index(drop=True)
)

print(f"\nPosition stats:")
print(submission_final["target_position"].describe().round(6))
submission_final.to_csv("submission06.csv", index=False)
print("saved submission06.csv")
