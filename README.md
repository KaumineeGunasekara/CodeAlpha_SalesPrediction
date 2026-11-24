# CodeAlpha_SalesPrediction

Task 4 of the CodeAlpha Data Science Internship: predict sales from advertising spend and work out how each advertising channel affects sales.

## Dataset
`Advertising.csv` has 200 markets with advertising budgets for **TV**, **Radio** and **Newspaper**, and the resulting **Sales**. There are no missing values or duplicates. (This dataset is usually described as budgets in thousands of dollars and sales in thousands of units. The file itself doesn't state units, so the results are reported in "units".) The task sheet mentions target segment and platform. This dataset has no segment column, and the three channels play the role of platform.

## Approach
1. **Cleaning**: dropped the saved index column and checked for missing values and duplicates.
2. **EDA**: scatter plots of sales against each channel, correlation matrix.
3. **Feature engineering and selection**: tested dropping Newspaper, and added a **TV x Radio interaction** feature to check whether the two channels work better together.
4. **Modeling**: 80/20 split. Six models compared with 5-fold CV repeated 3 times on the training set: linear regression (three feature sets), Ridge, Random Forest, Gradient Boosting.
5. **Impact analysis**: linear-model coefficients with bootstrap 95% confidence intervals, and budget scenarios with the same total spend.

## Results
| Model | CV R2 | Test R2 | Test MAE |
|---|---|---|---|
| Linear (TV+Radio+Newspaper) | 0.880 | 0.899 | 1.46 |
| Linear (TV+Radio) | 0.882 | 0.901 | 1.44 |
| Linear (TV+Radio+TVxRadio) | 0.960 | 0.974 | 0.67 |
| Ridge (TV+Radio+TVxRadio) | 0.960 | 0.974 | 0.67 |
| Random Forest | 0.968 | 0.983 | 0.61 |
| **Gradient Boosting** | **0.970** | **0.983** | **0.62** |

The selected model (Gradient Boosting) predicts sales with an average error of about **0.6 units** on unseen markets (R2 = 0.98). The simple linear model with the TV x Radio interaction is almost as good (R2 = 0.97) and is much easier to explain.

## Insights for marketing strategy
- **Newspaper adds nothing.** Once TV and Radio are in the model, the Newspaper coefficient is about 0 (95% CI: -0.013 to +0.012), and removing it did not hurt accuracy. Newspaper correlates with sales (0.23) only because markets that spend on newspaper also tend to spend on radio (correlation 0.35).
- **Radio gives the most sales per unit of budget** (about +0.19 sales per unit, versus +0.046 for TV). TV still matters a lot, because its budgets are much larger.
- **TV and Radio work better together.** Adding the TV x Radio interaction raised CV R2 from 0.88 to 0.96. Splitting a budget across both beats putting everything on one channel.
- **Budget scenarios** (total budget 100, see `budget_scenarios.csv`): both models agree that spending everything on TV (about 9) is clearly worse than a TV + Radio mix (about 11-12), and that adding newspaper spend lowers the prediction because it takes money from the channels that work.

## Limitations
- This is observational data, so the results show association, not proven cause and effect.
- Radio spend in the data never exceeds 49.6, so the model can't say what happens beyond that. Returns on radio probably flatten at high spend. Scenarios were kept inside the observed range for this reason.
- With 200 rows, the test set is only 40 markets, so test scores are a bit noisy. The cross-validation scores are the more reliable comparison, and Gradient Boosting, Random Forest and the interaction model are within noise of each other.

## Run
```bash
pip install -r requirements.txt
python sales_prediction.py
```

## Files
- `sales_prediction.py`: full pipeline
- `model_comparison.csv`, `channel_effects.csv`, `budget_scenarios.csv`: results
- `images/`: plots
