# TP1 submission

This submission implements all four exercises from the practical-session handout in [`TP1.ipynb`](TP1.ipynb), with the reusable code also available in [`notebook.py`](notebook.py). The notebook has a separate run cell for each exercise and displays the missing-value heatmap inline. It prints the requested dataset summaries, split sizes and proportions, text-dataset details, imputation comparisons, scaling results, encoding mappings, and short discussion answers.

## Run

From the `machine-learning-tp1-starter` directory:

```bash
python submissions/TP1/notebook.py
```


Install the packages listed in `requirements.txt` first. The script saves the missing-value heatmap to `submissions/TP1/results/employees_missing_values.png`. The 20 Newsgroups dataset can require an internet connection on its first load; if unavailable, that section reports the issue and the other exercises continue.

The Iris split uses the four measurements as inputs and `species` as the stratified target. For the employees example, `First Name` is treated as name-like data and is not imputed; numeric fields use the median, and categorical fields use the most frequent value. Placeholder normalization is case-insensitive and limited to whole-cell `-`, `na`, `n/a`, and `unknown` values.
