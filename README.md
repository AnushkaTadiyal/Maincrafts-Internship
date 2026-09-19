\# California House Price Predictor (Linear Regression)



Maincrafts Technology, AI \& ML Task 1. This project predicts median house

value in California using scikit-learn's LinearRegression on the California

Housing dataset.



\## What's inside

\- `task1\_ml\_linear\_regression.ipynb`: EDA, training, evaluation, diagnostic plots and an improvement experiment

\- `task1\_ml\_report.pdf`: short report on the EDA, model, metrics and improvement ideas

\- `make\_report.py`: rebuilds the PDF report from the notebook's saved results

\- `figures/`: plots used in the report

\- `linear\_regression\_model.pkl`: the trained model



\## Results (test set)

The target is in units of $100,000.



| Metric | Value |

|---|---|

| MAE | 0.53 |

| RMSE | 0.75 |

| R² | 0.57 |



A gradient-boosting model improves R² to about 0.83.



!\[Actual vs predicted](figures/fig4\_actual\_vs\_predicted.png)



\## How to run

1\. Install the requirements: `pip install -r requirements.txt`

2\. Open the notebook with `jupyter notebook task1\_ml\_linear\_regression.ipynb`, then click \*\*Kernel → Restart \& Run All\*\*

3\. Rebuild the report with `python make\_report.py`



\## Author

Anushka

