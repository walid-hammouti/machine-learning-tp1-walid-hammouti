"""Machine Learning - Practical Session 1: exploration and preprocessing.

Run from any directory with: python submissions/TP1/notebook.py
The script resolves the supplied datasets relative to this file. The optional
20 Newsgroups section may download its data on the first run.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    Binarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    normalize,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


def exercise_1() -> None:
    print("\nEXERCISE 1 - TABULAR DATA AND SPLITS")
    D = pd.read_csv(DATA / "Iris_Dataset.csv")
    print("info() reports column types and non-null counts:")
    D.info()
    print("describe() summarizes numerical columns:")
    print(D.describe())
    print("shape (rows, columns):", D.shape)
    print("columns:", D.columns.tolist())
    print("head() previews the first rows:\n", D.head())

    selected_features = D[["sepal_length", "petal_width"]]
    print("First 20 rows of the selected features:\n", selected_features.head(20))
    D_copy = D.copy()
    print("A copy lets preprocessing experiments leave the original data unchanged.")

    # The species column is the target; the four measurements are inputs.
    X = D_copy.drop(columns="species")
    y = D_copy["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Inputs:", X.columns.tolist(), "| target: species")
    for name, part in (("train", X_train), ("test", X_test)):
        print(f"{name}: {len(part)} observations ({len(part) / len(D):.1%})")
    print("Train class proportions:\n", y_train.value_counts(normalize=True).sort_index())
    print("Test class proportions:\n", y_test.value_counts(normalize=True).sort_index())
    print(
        "Keep the test set untouched until final evaluation: using its statistics "
        "for preprocessing or choosing a model leaks information and makes the "
        "reported score optimistically biased."
    )


def exercise_2() -> None:
    print("\nEXERCISE 2 - TEXT DATA")
    text_path = DATA / "txt_file.txt"
    with text_path.open(encoding="utf-8") as text_file:
        for line in text_file:
            print(line.rstrip("\n"))

    try:
        news = fetch_20newsgroups(subset="train")
    except Exception as error:
        print("20 Newsgroups could not be loaded (network/cache may be unavailable):", error)
        print("Re-run with network access or place the dataset in scikit-learn's local cache.")
        return
    print("Document count:", len(news.data))
    print("Label count:", len(news.target), "(one category ID for each document)")
    print("First document:\n", news.data[0])
    first_target = int(news.target[0])
    print("First numeric target:", first_target)
    print("Its class name:", news.target_names[first_target])
    print("Available class names:", news.target_names)
    print("Target IDs index target_names, so target_names[news.target[i]] names document i's class.")
    lengths = np.array([len(document.split()) for document in news.data])
    print("Document word counts (min / median / max):", lengths.min(), np.median(lengths), lengths.max())


def exercise_3() -> None:
    print("\nEXERCISE 3 - MISSING VALUES")
    employees = pd.read_csv(DATA / "employees.csv")
    print("Columns:", employees.columns.tolist())
    print("Data types:\n", employees.dtypes)
    print("Preview:\n", employees.head())
    print("Missing values before adding a row:\n", employees.isna().sum())
    plt.figure(figsize=(10, 5))
    sns.heatmap(employees.isna(), cbar=False, yticklabels=False, cmap="mako")
    plt.title("Missing values in the employees dataset")
    plt.xlabel("Employee fields")
    plt.ylabel("Employee rows")
    plt.tight_layout()
    plt.savefig(OUTPUT / "employees_missing_values.png", dpi=160)
    plt.close()

    employees_work = employees.copy()
    # Eight fields in the CSV order. '-' and 'na' are deliberate text markers;
    # the other fields use np.nan to represent missing values.
    synthetic = {
        "First Name": "-",
        "Gender": "na",
        "Start Date": np.nan,
        "Last Login Time": np.nan,
        "Salary": np.nan,
        "Bonus %": np.nan,
        "Senior Management": np.nan,
        "Team": np.nan,
    }
    employees_work = pd.concat([employees_work, pd.DataFrame([synthetic])], ignore_index=True)
    print("Missing values after synthetic row (literal markers are not counted):\n", employees_work.isna().sum())
    print("Pandas treats actual nulls as missing; '-' and 'na' are ordinary strings until normalized.")

    # Normalize only the agreed, case-insensitive whole-cell placeholders.
    employees_work = employees_work.replace(r"(?i)^\s*(?:-|na|n/a|unknown)\s*$", np.nan, regex=True)
    missing_before_imputation = employees_work.isna().sum()
    print("Missing values after marker normalization:\n", missing_before_imputation)

    # First Name is name-like and not a useful feature to impute; leave it out.
    # Impute numeric and categorical feature groups with appropriate strategies.
    numeric_columns = ["Salary", "Bonus %"]
    categorical_columns = [
        "Gender", "Start Date", "Last Login Time", "Senior Management", "Team"
    ]
    processed = employees_work.copy()
    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    processed[numeric_columns] = numeric_imputer.fit_transform(processed[numeric_columns])
    processed[categorical_columns] = categorical_imputer.fit_transform(processed[categorical_columns])
    print("Missing values after imputation:\n", processed.isna().sum())
    print(
        "Imputation fills gaps with plausible summary values, but can hide uncertainty, "
        "distort distributions, and weaken relationships. For prediction, fit imputers "
        "on training data only and transform held-out data with those fitted imputers."
    )


def exercise_4() -> None:
    print("\nEXERCISE 4A - SCALING AND BINARIZATION")
    matrix = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]], dtype=float)
    print("Original matrix:\n", matrix)
    scalers = {
        "StandardScaler (per feature: zero mean, unit variance)": StandardScaler(),
        "MinMaxScaler (per feature: [0, 1])": MinMaxScaler(),
        "MaxAbsScaler (per feature: [-1, 1] by max absolute value)": MaxAbsScaler(),
        "RobustScaler (per feature: median and interquartile range)": RobustScaler(),
    }
    for name, scaler in scalers.items():
        print(name, ":\n", scaler.fit_transform(matrix))
    print("normalize (per sample/row to unit L2 norm):\n", normalize(matrix))
    threshold = 0.5
    print(f"Binarizer at threshold {threshold} (values > threshold become 1):\n", Binarizer(threshold=threshold).fit_transform(matrix))
    print(
        "Scaling matters for distance- and gradient-based methods because feature "
        "units affect distances and optimization. Tree splits depend on ordering, "
        "so ordinary feature scaling is usually unnecessary for decision trees."
    )

    print("\nEXERCISE 4B - CATEGORICAL ENCODING")
    categories = pd.DataFrame(
        [
            ["male", "from US", "uses Safari"],
            ["female", "from Europe", "uses Safari"],
            ["female", "from Asia", "uses Firefox"],
            ["male", "from Africa", "uses Chrome"],
        ],
        columns=["Feature 1", "Feature 2", "Feature 3"],
    )
    target = pd.Series(["US", "France", "Japan", "Algeria"], name="Target")
    ordinal = OrdinalEncoder()
    X_encoded = ordinal.fit_transform(categories)
    print("Ordinal encoded inputs:\n", X_encoded)
    print("OrdinalEncoder categories_:", ordinal.categories_)
    print("These are arbitrary category codes, not meaningful rankings.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(target)
    print("Target class-to-label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print("Integer encoded target:", y_encoded)
    try:
        one_hot_encoder = OneHotEncoder(sparse_output=False)
    except TypeError:  # Compatibility with older scikit-learn releases.
        one_hot_encoder = OneHotEncoder(sparse=False)
    target_one_hot = one_hot_encoder.fit_transform(target.to_frame())
    print("OneHotEncoder categories_:", one_hot_encoder.categories_)
    print("One-hot encoded target (column order shown above):\n", target_one_hot)
    print(
        "Integer labels use one code per class; one-hot encoding uses one binary "
        "column per class. For nominal inputs, one-hot encoding avoids suggesting "
        "an artificial order or distance between category codes."
    )


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    print("\nConclusion: scale numeric features when the model depends on magnitude or distance; "
          "impute missing values with training-fitted transformers; encode categorical "
          "values according to whether they are nominal or truly ordered; represent text "
          "with text-specific features rather than treating raw strings as numbers.")
