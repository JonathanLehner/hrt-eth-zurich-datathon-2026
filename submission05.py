import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
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
#
# Selected via 10-fold CV forward selection from a broad set:
#   max_drawdown : peak-to-trough decline in seen window
#                  (mean-reversion signal — big drawdown → bounce)
#   avg_hl       : mean (high-low)/close per bar — realized volatility proxy
#                  (wider bars → more vol-adjusted opportunity)
#   early_ret    : return over first 10 bars
#                  (early trend signals whether initial move sustains)
# =========================================================
def build_features(bars_df):
    b = bars_df.sort_values(["session", "bar_ix"]).copy()
    b["ret"] = b.groupby("session")["close"].pct_change()
    b["hl"]  = (b["high"] - b["low"]) / b["close"]
    b["cp"]  = (b["close"] - b["low"]) / (b["high"] - b["low"] + 1e-9)

    def session_feats(g):
        closes = g["close"].values
        rets   = g["ret"].dropna().values
        rv     = rets.std() + 1e-9

        peak        = np.maximum.accumulate(closes)
        max_dd      = ((closes - peak) / peak).min()

        avg_hl      = g["hl"].mean()

        early       = g.head(10)
        early_ret   = early["close"].iloc[-1] / early["close"].iloc[0] - 1

        return pd.Series({
            "max_drawdown": max_dd,
            "avg_hl":       avg_hl,
            "early_ret":    early_ret,
        })

    return b.groupby("session").apply(session_feats).reset_index()

FEATURES = ["max_drawdown", "avg_hl", "early_ret"]

# =========================================================
# Train
# =========================================================
train_feats = build_features(bars_seen_train)
train_data  = target[["session", "target"]].merge(train_feats, on="session").dropna()

X_train = train_data[FEATURES].values
y_train = train_data["target"].values

# CV check
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_r2 = cross_val_score(Ridge(alpha=0.1), X_train, y_train, cv=cv, scoring='r2')
print(f"CV R²: {cv_r2.mean():.5f} +/- {cv_r2.std():.5f}")

model = Ridge(alpha=0.1)
model.fit(X_train, y_train)
print("coefficients:", dict(zip(FEATURES, model.coef_)))

# =========================================================
# Predict + Sharpe-optimal position sizing
#
# For Sharpe, position ∝ predicted_return (Kelly). avg_hl is
# already a feature so the model accounts for vol. We
# z-score predictions across training then rescale to a fixed
# target volatility, so positions are always centred at 0
# with consistent magnitude regardless of feature scale.
# Clip at ±2σ to limit tail risk on outlier sessions.
# =========================================================
TARGET_VOL = 0.01   # target std of position distribution

train_pred = model.predict(X_train)
pred_mean  = train_pred.mean()
pred_std   = train_pred.std() + 1e-9

def make_submission(bars_df):
    feats = build_features(bars_df)
    pred  = model.predict(feats[FEATURES].values)
    pos   = (pred - pred_mean) / pred_std * TARGET_VOL
    pos   = np.clip(pos, -2 * TARGET_VOL, 2 * TARGET_VOL)
    feats["target_position"] = pos
    return feats[["session", "target_position"]]


# =========================================================
# Predict & save
# =========================================================
submission_final = (
    pd.concat([make_submission(bars_seen_public), make_submission(bars_seen_private)],
              ignore_index=True)
    .sort_values("session")
    .reset_index(drop=True)
)

print(submission_final.describe())
print(submission_final.shape)
submission_final.to_csv("submission05.csv", index=False)
print("saved submission05.csv")
