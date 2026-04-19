"""
Analysis and visualisation for submission03 (Random Forest + sentiment).
Saves all plots to analysis/plots/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

os.makedirs("analysis/plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# =========================================================
# Rebuild features (mirrors submission03.py)
# =========================================================
bars_seen_train = pd.read_parquet("data/bars_seen_train.parquet")
bars_unseen_train = pd.read_parquet("data/bars_unseen_train.parquet").sort_values(["session", "bar_ix"])
sentiment = pd.read_csv("analysis/company-ids.csv")

data = bars_unseen_train.groupby("session").apply(
    lambda x: x["close"].iloc[-1] / x["open"].iloc[0] - 1
).rename("target").reset_index()


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
train_features = (
    build_feature_table(bars_seen_train)
    .merge(sent_features, on="session", how="left")
    .fillna({c: 0.0 for c in SENT_COLS})
)
train_data = data[["session", "target"]].merge(train_features, on="session", how="left").dropna()

X = train_data[FEATURES].values
y = train_data["target"].values

model = RandomForestRegressor(n_estimators=500, max_features="sqrt", min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(X, y)

tree_preds_train = np.array([t.predict(X) for t in model.estimators_])
pred_mean  = tree_preds_train.mean(axis=0)
pred_std   = tree_preds_train.std(axis=0)
k = 10.0
sized_position = pred_mean * (1.0 / (1.0 + k * pred_std))
residuals = y - pred_mean

# 5-fold OOB-style cross-val predictions for honest residual plot
cv_pred = cross_val_predict(
    RandomForestRegressor(n_estimators=200, max_features="sqrt", min_samples_leaf=5, random_state=42, n_jobs=-1),
    X, y, cv=5
)
cv_residuals = y - cv_pred

submission = pd.read_csv("submission03.csv")

# =========================================================
# Plot 1 — Feature importances
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5))
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
colors = ["#4C72B0" if not f.startswith("sent") else "#DD8452" for f in imp.index]
imp.plot.barh(ax=ax, color=colors)
ax.set_xlabel("Mean decrease in impurity")
ax.set_title("Feature importances (Random Forest)")
ax.axvline(1 / len(FEATURES), color="grey", linestyle="--", linewidth=0.8, label="uniform baseline")
ax.legend()
plt.tight_layout()
fig.savefig("analysis/plots/01_feature_importances.png", dpi=150)
plt.close()

# =========================================================
# Plot 2 — Actual vs predicted (CV) + residuals
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(cv_pred, y, alpha=0.4, s=20, edgecolors="none")
lim = max(abs(y).max(), abs(cv_pred).max()) * 1.05
ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1)
ax.set_xlabel("CV predicted return")
ax.set_ylabel("Actual return")
ax.set_title("Actual vs CV-predicted (5-fold)")

ax = axes[1]
ax.hist(cv_residuals, bins=40, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="red", linewidth=1)
ax.set_xlabel("Residual (actual − predicted)")
ax.set_title("CV residual distribution")

plt.tight_layout()
fig.savefig("analysis/plots/02_actual_vs_predicted.png", dpi=150)
plt.close()

# =========================================================
# Plot 3 — Prediction uncertainty vs position sizing
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(pred_std, pred_mean, alpha=0.4, s=20, edgecolors="none", label="raw prediction")
ax.scatter(pred_std, sized_position, alpha=0.4, s=20, edgecolors="none", label="sized position")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Prediction std (tree disagreement)")
ax.set_ylabel("Position / prediction")
ax.set_title("Effect of uncertainty-based sizing")
ax.legend()

ax = axes[1]
ax.scatter(pred_std, abs(pred_mean) - abs(sized_position), alpha=0.4, s=20, edgecolors="none")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Prediction std")
ax.set_ylabel("Size reduction |raw| − |sized|")
ax.set_title("Position shrinkage by uncertainty")

plt.tight_layout()
fig.savefig("analysis/plots/03_uncertainty_sizing.png", dpi=150)
plt.close()

# =========================================================
# Plot 4 — Submission position distribution
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(submission["target_position"], bins=60, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="red", linewidth=1)
ax.set_xlabel("Position size")
ax.set_title(f"Submission position distribution  (n={len(submission)})")

stats = submission["target_position"].describe()
textstr = "\n".join([
    f"mean  {stats['mean']:.4f}",
    f"std   {stats['std']:.4f}",
    f"min   {stats['min']:.4f}",
    f"max   {stats['max']:.4f}",
    f"% long  {(submission['target_position'] > 0).mean():.1%}",
])
ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plt.tight_layout()
fig.savefig("analysis/plots/04_position_distribution.png", dpi=150)
plt.close()

# =========================================================
# Plot 5 — Feature correlations with target
# =========================================================
corr_df = train_data[FEATURES + ["target"]].corr()[["target"]].drop("target").sort_values("target")
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#d62728" if v < 0 else "#2ca02c" for v in corr_df["target"]]
corr_df["target"].plot.barh(ax=ax, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson correlation with target")
ax.set_title("Feature–target correlations (training set)")
plt.tight_layout()
fig.savefig("analysis/plots/05_feature_target_correlations.png", dpi=150)
plt.close()

# =========================================================
# Plot 6 — Sentiment vs target scatter (2-panel)
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, feat in zip(axes, ["sent_short", "sent_long"]):
    ax.scatter(train_data[feat], train_data["target"], alpha=0.3, s=15, edgecolors="none")
    m, b = np.polyfit(train_data[feat], train_data["target"], 1)
    xs = np.linspace(train_data[feat].min(), train_data[feat].max(), 100)
    ax.plot(xs, m * xs + b, "r-", linewidth=1.5)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel(feat)
    ax.set_ylabel("target return")
    ax.set_title(f"{feat} vs target  (slope={m:.5f})")
plt.tight_layout()
fig.savefig("analysis/plots/06_sentiment_vs_target.png", dpi=150)
plt.close()

print("Saved 6 plots to analysis/plots/")
print(f"\nCV R²: {1 - np.var(cv_residuals)/np.var(y):.4f}")
print(f"Train R² (in-sample): {1 - np.var(residuals)/np.var(y):.4f}")
print(f"Mean |sized position|: {np.abs(sized_position).mean():.5f}")
print(f"Mean |raw pred|:       {np.abs(pred_mean).mean():.5f}")
