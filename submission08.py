"""
submission08 — Mean-reversion + peer-sentiment spillover

Competition metric: Sharpe = mean(pnl_i) / std(pnl_i) * 16
  pnl_i = position_i * (close_end_i / close_halfway_i - 1)

Signals (10-fold CV Sharpe, rank-normalised features):
  mom20    + sharpe20                    → 0.1889  (3.022)
  mom20    + sharpe20 + peer_short/long  → 0.1908  (3.053)  ← this submission

Focal company identification:
  Each session has 4 companies in the headlines. The TRADED stock
  is identified by measuring which company's sentiment at each barIx
  is most aligned with the contemporaneous bar return (shortTerm * ret).
  The company with the highest cumulative alignment score = focal stock.

  Focal company sentiment has near-zero IC with the unseen target — it
  already moved the price (priced in). PEER companies (the other 3)
  have positive IC (+0.020/+0.033 for short/long term) because their
  news represents sector tailwind not yet reflected in this stock's price.

  This matches the intuition: positive news about related companies that
  hasn't driven this stock up yet → forward-looking spillover signal.
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
# Focal company identification
#
# For each (session, company): compute alignment = sum(shortTerm * bar_ret)
# at every barIx where that company has a headline.
# Company with max alignment = focal (traded) stock.
# All others = peers whose sentiment has not yet moved this stock.
# =========================================================
def identify_focal_and_peer_sentiment(bars_df, sentiment_df):
    b = bars_df.sort_values(["session", "bar_ix"]).copy()
    b["ret"] = b.groupby("session")["close"].pct_change()
    bar_ret  = b[["session", "bar_ix", "ret"]].rename(columns={"bar_ix": "barIx"})

    sent_ret = sentiment_df.merge(bar_ret, on=["session", "barIx"], how="left")
    sent_ret["align"] = sent_ret["shortTerm"] * sent_ret["ret"]

    align = sent_ret.groupby(["session", "companyId"])["align"].sum().reset_index()
    focal_id = (align.loc[align.groupby("session")["align"].idxmax()]
                [["session", "companyId"]]
                .rename(columns={"companyId": "focal_id"}))

    sent_mean = (sentiment_df
                 .groupby(["session", "companyId"])[["shortTerm", "longTerm"]]
                 .mean()
                 .reset_index()
                 .merge(focal_id, on="session"))

    peer = sent_mean[sent_mean["companyId"] != sent_mean["focal_id"]]
    peer_sent = (peer.groupby("session")[["shortTerm", "longTerm"]]
                 .mean()
                 .rename(columns={"shortTerm": "peer_short", "longTerm": "peer_long"})
                 .reset_index())
    return peer_sent

# =========================================================
# Price features
# =========================================================
FEATURES = ["mom20", "sharpe20", "peer_short", "peer_long"]
SENT_COLS = ["peer_short", "peer_long"]

def build_price_features(bars_df):
    b = bars_df.sort_values(["session", "bar_ix"]).copy()
    b["ret"] = b.groupby("session")["close"].pct_change()
    def f(g):
        c = g["close"].values
        r = g["ret"].dropna().values
        mom20    = c[-1] / c[-21] - 1  if len(c) >= 21 else c[-1] / c[0] - 1
        vol20    = r[-20:].std() + 1e-9 if len(r) >= 20 else r.std() + 1e-9
        sharpe20 = mom20 / vol20
        return pd.Series({"mom20": mom20, "sharpe20": sharpe20})
    return b.groupby("session").apply(f).reset_index()

def rank_norm(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = df[c].rank(pct=True)
    return df

def merge_all(bars_df, peer_sent_df):
    return (build_price_features(bars_df)
            .merge(peer_sent_df, on="session", how="left")
            .fillna({c: 0.5 for c in SENT_COLS}))  # 0.5 = neutral rank for missing

# =========================================================
# Build peer sentiment for all sessions (train + test)
# Test sessions also have sentiment in company-ids.csv
# =========================================================
all_bars_seen = pd.concat([bars_seen_train, bars_seen_public, bars_seen_private], ignore_index=True)
peer_sent_all = identify_focal_and_peer_sentiment(all_bars_seen, sentiment)

# =========================================================
# Train
# =========================================================
train_feats  = merge_all(bars_seen_train, peer_sent_all)
train_data   = target.merge(train_feats, on="session").dropna()
train_ranked = rank_norm(train_data, FEATURES)

X_train = train_ranked[FEATURES].values
y_train = train_ranked["target"].values

model = Ridge(alpha=0.001)
model.fit(X_train, y_train)

# =========================================================
# Validation
# =========================================================
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_sharpes, cv_ics = [], []
for tr, va in cv.split(X_train):
    m = Ridge(alpha=0.001).fit(X_train[tr], y_train[tr])
    pos = m.predict(X_train[va])
    pnl = pos * y_train[va]
    cv_sharpes.append(pnl.mean() / (pnl.std() + 1e-9))
    cv_ics.append(stats.spearmanr(pos, y_train[va]).correlation)

insample_pos = model.predict(X_train)
insample_pnl = insample_pos * y_train
baseline_sh  = y_train.mean() / y_train.std()

print(f"Coefficients: {dict(zip(FEATURES, model.coef_.round(6)))}")
print(f"Intercept:    {model.intercept_:.6f}")
print()
print(f"{'Metric':35s}  {'Per-session':>12s}  {'x16 (competition)':>18s}")
print("-" * 70)
print(f"{'Baseline (constant long)':35s}  {baseline_sh:>12.4f}  {baseline_sh*16:>18.4f}")
print(f"{'In-sample Sharpe':35s}  {insample_pnl.mean()/(insample_pnl.std()+1e-9):>12.4f}  {insample_pnl.mean()/(insample_pnl.std()+1e-9)*16:>18.4f}")
print(f"{'10-fold CV Sharpe (mean)':35s}  {np.mean(cv_sharpes):>12.4f}  {np.mean(cv_sharpes)*16:>18.4f}")
print(f"{'10-fold CV Sharpe (std)':35s}  {np.std(cv_sharpes):>12.4f}")
print(f"{'10-fold CV IC (rank corr)':35s}  {np.mean(cv_ics):>12.4f}")
print(f"{'Overfit gap':35s}  {insample_pnl.mean()/(insample_pnl.std()+1e-9) - np.mean(cv_sharpes):>12.4f}")

# =========================================================
# Predict
# =========================================================
def make_submission(bars_df):
    feats  = merge_all(bars_df, peer_sent_all)
    ranked = rank_norm(feats, FEATURES)
    ranked["target_position"] = model.predict(ranked[FEATURES].values)
    return ranked[["session", "target_position"]]

submission_final = (
    pd.concat([make_submission(bars_seen_public), make_submission(bars_seen_private)],
              ignore_index=True)
    .sort_values("session")
    .reset_index(drop=True)
)

print()
print("Submission position stats:")
print(submission_final["target_position"].describe().round(6))
submission_final.to_csv("submission08.csv", index=False)
print("saved submission08.csv")
