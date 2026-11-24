"""Task 4: Sales Prediction using Python"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

os.makedirs("images", exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)

# ---------------- 1. Load & clean ----------------
df = pd.read_csv("Advertising.csv")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""])  # drop saved index column
print("Shape:", df.shape, "| Missing:", int(df.isna().sum().sum()), "| Duplicates:", df.duplicated().sum())
print(df.describe().round(2).T)
print("\nCorrelation with Sales:\n", df.corr()["Sales"].drop("Sales").round(3))

# ---------------- 2. EDA ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, c in zip(ax, ["TV", "Radio", "Newspaper"]):
    sns.regplot(data=df, x=c, y="Sales", ax=a, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    a.set_title(f"Sales vs {c} (corr = {df[c].corr(df['Sales']):.2f})")
plt.tight_layout(); plt.savefig("images/sales_vs_channels.png", dpi=120); plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig("images/correlation.png", dpi=120); plt.close()

# ---------------- 3. Feature engineering / selection ----------------
# TV x Radio interaction: does spending on both together do more than the sum of its parts?
df["TV_x_Radio"] = df["TV"] * df["Radio"]
feature_sets = {
    "TV+Radio+Newspaper": ["TV", "Radio", "Newspaper"],
    "TV+Radio": ["TV", "Radio"],
    "TV+Radio+TVxRadio": ["TV", "Radio", "TV_x_Radio"],
}
y = df["Sales"]
idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=SEED)

# ---------------- 4. Model comparison (repeated 5-fold CV on training data) ----------------
candidates = {
    "Linear (TV+Radio+Newspaper)": (LinearRegression(), "TV+Radio+Newspaper"),
    "Linear (TV+Radio)": (LinearRegression(), "TV+Radio"),
    "Linear (TV+Radio+TVxRadio)": (LinearRegression(), "TV+Radio+TVxRadio"),
    "Ridge (TV+Radio+TVxRadio)": (make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "TV+Radio+TVxRadio"),
    "Random Forest (all 3 channels)": (RandomForestRegressor(n_estimators=300, random_state=SEED), "TV+Radio+Newspaper"),
    "Gradient Boosting (all 3 channels)": (GradientBoostingRegressor(random_state=SEED), "TV+Radio+Newspaper"),
}
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
rows, fitted = [], {}
for name, (model, fs) in candidates.items():
    cols = feature_sets[fs]
    Xtr, Xte = df.loc[idx_train, cols], df.loc[idx_test, cols]
    cvr2 = cross_val_score(model, Xtr, y[idx_train], cv=cv, scoring="r2")
    model.fit(Xtr, y[idx_train]); p = model.predict(Xte)
    fitted[name] = (model, cols)
    rows.append((name, cvr2.mean(), cvr2.std(), r2_score(y[idx_test], p),
                 mean_absolute_error(y[idx_test], p), np.sqrt(mean_squared_error(y[idx_test], p))))
res = pd.DataFrame(rows, columns=["Model", "CV R2 (mean)", "CV R2 (std)", "Test R2", "Test MAE", "Test RMSE"]).round(3)
print("\n", res.to_string(index=False))
res.to_csv("model_comparison.csv", index=False)

# ---------------- 5. Final model ----------------
best_name = res.sort_values("CV R2 (mean)", ascending=False).iloc[0]["Model"]
best, cols = fitted[best_name]
pred = best.predict(df.loc[idx_test, cols])
print(f"\nSelected model: {best_name}")
print(f"Test R2 = {r2_score(y[idx_test], pred):.3f} | MAE = {mean_absolute_error(y[idx_test], pred):.2f} | "
      f"RMSE = {np.sqrt(mean_squared_error(y[idx_test], pred)):.2f}  (Sales units: thousands)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(y[idx_test], pred, alpha=0.7)
lim = [0, max(y[idx_test].max(), pred.max()) + 1]
ax[0].plot(lim, lim, "r--"); ax[0].set(xlabel="Actual sales", ylabel="Predicted sales", title=f"Actual vs predicted - {best_name}")
ax[1].scatter(pred, y[idx_test] - pred, alpha=0.7); ax[1].axhline(0, color="r", ls="--")
ax[1].set(xlabel="Predicted sales", ylabel="Residual", title="Residuals")
plt.tight_layout(); plt.savefig("images/actual_vs_predicted.png", dpi=120); plt.close()

# ---------------- 6. How does advertising impact sales? ----------------
# Plain linear model on all data with bootstrap 95% confidence intervals for each coefficient
# (Sales change per +1 unit of ad budget; budgets are in thousands of dollars)
X3 = df[["TV", "Radio", "Newspaper"]].values
coefs = []
for _ in range(2000):
    s = rng.integers(0, len(df), len(df))
    coefs.append(LinearRegression().fit(X3[s], y.values[s]).coef_)
coefs = np.array(coefs)
lin = LinearRegression().fit(X3, y)
ci = pd.DataFrame({"coef": lin.coef_, "ci_low": np.percentile(coefs, 2.5, axis=0),
                   "ci_high": np.percentile(coefs, 97.5, axis=0)}, index=["TV", "Radio", "Newspaper"]).round(4)
print("\nSales change per +1 unit of ad budget (bootstrap 95% CI):\n", ci)
ci.to_csv("channel_effects.csv")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.errorbar(ci["coef"], ci.index, xerr=[ci["coef"] - ci["ci_low"], ci["ci_high"] - ci["coef"]], fmt="o", capsize=5)
ax.axvline(0, color="r", ls="--"); ax.set_xlabel("Effect on sales per +1 unit budget"); ax.set_title("Channel effects (95% CI)")
plt.tight_layout(); plt.savefig("images/channel_effects.png", dpi=120); plt.close()

# Budget scenarios: same total budget (100 units), different splits. All values stay inside the range seen in
# the data (Radio max is 49.6), because models cannot be trusted outside it. Two different models are shown as
# a cross-check: the selected model and the simple linear model with the TV x Radio interaction.
lin_int, lin_cols = fitted["Linear (TV+Radio+TVxRadio)"]
def predict_budget(model, mcols, tv, radio, news):
    row = pd.DataFrame([{"TV": tv, "Radio": radio, "Newspaper": news, "TV_x_Radio": tv * radio}])
    return float(model.predict(row[mcols])[0])
scen = pd.DataFrame([
    ("All TV", 100, 0, 0), ("TV 75 / Radio 25", 75, 25, 0), ("TV 60 / Radio 40", 60, 40, 0),
    ("TV 50 / Radio 25 / Newspaper 25", 50, 25, 25), ("Even split of 3 channels", 34, 33, 33),
], columns=["Scenario", "TV", "Radio", "Newspaper"])
scen[f"Sales ({best_name.split(' (')[0]})"] = [predict_budget(best, cols, r.TV, r.Radio, r.Newspaper) for r in scen.itertuples()]
scen["Sales (Linear + TVxRadio)"] = [predict_budget(lin_int, lin_cols, r.TV, r.Radio, r.Newspaper) for r in scen.itertuples()]
print("\nSame total budget (100), different splits:\n", scen.round(2).to_string(index=False))
scen.round(2).to_csv("budget_scenarios.csv", index=False)
