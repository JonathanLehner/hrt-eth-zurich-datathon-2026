"""
submission09 — Uncertainty-based position sizing

Competition: Sharpe = mean(pnl) / std(pnl) * 16
  pnl_i = position_i * (close_end / close_halfway - 1)

Key findings from IC stability analysis (first 500 vs last 500 training sessions):

  CONSISTENT signals (same sign in both halves):
    recent_return   IC: all=+0.032  h1=+0.052  h2=+0.013
    oc_change       IC: all=+0.028  h1=+0.052  h2=+0.004
    mom20           IC: all=-0.078  h1=-0.099  h2=-0.060  ← hurt in test (test dynamics differ)
    sharpe20        IC: all=-0.087  h1=-0.103  h2=-0.070  ← hurt in test
    unc_short       IC: all=+0.050  h1=+0.085  h2=+0.016  ← BEST single consistent signal

  NOISE (IC flips sign):
    return_seen, sharpe5, unc_long → excluded

  Raw features (no rank normalization) calibrate cleanly:
    sub01 (raw OLS):  CV 2.712 → actual 2.635  (gap 0.077)
    Rank-normalized: CV 3.022 → actual 2.186   (gap 0.836 — rank distortion in test)

Best CV Sharpe (raw features, Ridge, consistent signals only):
    unc_short alone:              0.17907 → 2.865  ← best single feature
    recent_return + unc_short:    0.17901 → 2.864
    recent_return + oc_change + unc_short: 0.17880 → 2.861

unc_short interpretation: average short-term sentiment uncertainty per session.
  Higher uncertainty → more active news flow → higher expected return (positive IC).
  Consistent across both halves of training and same distribution in test (KS p=0.19).
  Does NOT require focal company identification — simpler and more robust.
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
sentiment         = pd.read_csv("analysis/company-ids.csv")

target = (
    bars_unseen_train.groupby("session")["close"].last() /
    bars_seen_train.sort_values(["session","bar_ix"]).groupby("session")["close"].last() - 1
).rename("target").reset_index()

# =========================================================
# Sentiment feature: average short-term uncertainty per session
# All companies included — no focal ID needed (simpler, more robust)
# =========================================================
unc_feat = (sentiment.groupby("session")["shortTermUncertainty"]
            .mean()
            .rename("unc_short")
            .reset_index())

# Fill missing sessions with training mean
UNC_MEAN = unc_feat[unc_feat["session"] < 1000]["unc_short"].mean()

# =========================================================
# Price feature: last-5-bar momentum (consistent positive IC)
# =========================================================
def build_price_features(bars_df):
    b = bars_df.sort_values(["session", "bar_ix"]).copy()
    b["ret"] = b.groupby("session")["close"].pct_change()
    b["oc"]  = (b["close"] - b["open"]) / b["open"]
    rr = b.groupby("session")["ret"].apply(lambda x: x.tail(5).mean()).rename("recent_return")
    oc = b.groupby("session")["oc"].apply(lambda x: x.tail(5).mean()).rename("oc_change")
    return pd.concat([rr, oc], axis=1).reset_index()

FEATURES = ["unc_short", "recent_return", "oc_change"]

def merge_features(bars_df):
    pf = build_price_features(bars_df)
    return (pf.merge(unc_feat, on="session", how="left")
              .fillna({"unc_short": UNC_MEAN}))

# =========================================================
# Train — raw features, no rank normalization
# =========================================================
train_feats = merge_features(bars_seen_train)
train_data  = target.merge(train_feats, on="session").dropna()

X_train = train_data[FEATURES].values
y_train = train_data["target"].values

model = Ridge(alpha=0.01)
model.fit(X_train, y_train)

# =========================================================
# Validation
# =========================================================
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_sharpes, cv_ics = [], []
for tr, va in cv.split(X_train):
    m = Ridge(alpha=0.01).fit(X_train[tr], y_train[tr])
    pos = m.predict(X_train[va])
    pnl = pos * y_train[va]
    cv_sharpes.append(pnl.mean() / (pnl.std() + 1e-9))
    cv_ics.append(stats.spearmanr(pos, y_train[va]).correlation)

insample_pos = model.predict(X_train)
insample_pnl = insample_pos * y_train

print(f"Coefficients: {dict(zip(FEATURES, model.coef_.round(7)))}")
print(f"Intercept:    {model.intercept_:.6f}")
print()
print(f"{'Metric':35s}  {'Per-session':>12s}  {'x16':>8s}")
print("-" * 60)
print(f"{'Baseline (constant long)':35s}  {y_train.mean()/y_train.std():>12.4f}  {y_train.mean()/y_train.std()*16:>8.4f}")
print(f"{'In-sample Sharpe':35s}  {insample_pnl.mean()/(insample_pnl.std()+1e-9):>12.4f}  {insample_pnl.mean()/(insample_pnl.std()+1e-9)*16:>8.4f}")
print(f"{'10-fold CV Sharpe':35s}  {np.mean(cv_sharpes):>12.4f}  {np.mean(cv_sharpes)*16:>8.4f}")
print(f"{'10-fold CV IC':35s}  {np.mean(cv_ics):>12.4f}")
print(f"{'Overfit gap':35s}  {insample_pnl.mean()/(insample_pnl.std()+1e-9) - np.mean(cv_sharpes):>12.4f}")

# =========================================================
# Predict
# =========================================================
def make_submission(bars_df):
    feats = merge_features(bars_df)
    feats["target_position"] = model.predict(feats[FEATURES].values)
    return feats[["session", "target_position"]]

submission_final = (
    pd.concat([make_submission(bars_seen_public), make_submission(bars_seen_private)],
              ignore_index=True)
    .sort_values("session")
    .reset_index(drop=True)
)

print()
print("Position stats:")
print(submission_final["target_position"].describe().round(6))
submission_final.to_csv("submission09.csv", index=False)
print("saved submission09.csv")
