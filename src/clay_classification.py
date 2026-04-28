"""
Clay Classification using Machine Learning
Models: Decision Tree, Random Forest, and SVM
Validation: Leave-One-Out Cross-Validation (LOOCV)
Outputs: metrics, confusion matrices, feature importance plots, PCA plot,
         class distribution plot, robustness results, and new-sample prediction.

How to run:
    python src/clay_classification.py --data "data/Clay classification.csv" --target Output

Author: Dhrubajouti Karmakar
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def create_output_folders(base_output_dir: Path) -> tuple[Path, Path]:
    """Create folders for saving results and figures."""
    results_dir = base_output_dir / "results"
    figures_dir = base_output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, figures_dir


def save_text(text: str, file_path: Path) -> None:
    """Save text output to a file."""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


# -------------------------------------------------------------------
# Data loading and preprocessing
# -------------------------------------------------------------------
def load_data(file_path: Path) -> pd.DataFrame:
    """Load the clay classification dataset."""
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values."""
    return data.dropna().copy()


def split_features_target(data: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features and output labels."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode clay type labels into numeric classes."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded, label_encoder


# -------------------------------------------------------------------
# Visualization functions
# -------------------------------------------------------------------
def plot_class_distribution(data: pd.DataFrame, target_column: str, figures_dir: Path) -> None:
    """Save class distribution bar plot."""
    plt.figure(figsize=(7, 5))
    data[target_column].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Clay Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=300)
    plt.close()


def plot_correlation_heatmap(X: pd.DataFrame, figures_dir: Path) -> None:
    """Save correlation heatmap for input features."""
    plt.figure(figsize=(10, 8))
    corr = X.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=300)
    plt.close()


def plot_pca(X: pd.DataFrame, y: pd.Series, figures_dir: Path) -> None:
    """Save 2D PCA plot of clay samples."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, s=100)
    plt.title("PCA Plot of Clay Samples")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_plot.png", dpi=300)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: np.ndarray,
                          model_name: str,
                          figures_dir: Path) -> None:
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(figures_dir / f"confusion_matrix_{safe_name}.png", dpi=300)
    plt.close()


def plot_accuracy_comparison(results: dict, figures_dir: Path) -> None:
    """Save model accuracy comparison plot."""
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]

    plt.figure(figsize=(7, 5))
    plt.bar(model_names, accuracies)
    plt.title("Model Accuracy Comparison (LOOCV)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy_comparison.png", dpi=300)
    plt.close()


def plot_feature_importance(model, feature_names: list[str], model_name: str, figures_dir: Path) -> pd.Series:
    """Save feature importance plot for tree-based models."""
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    importance.sort_values().plot(kind="barh")
    plt.title(f"Feature Importance - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(figures_dir / f"feature_importance_{safe_name}.png", dpi=300)
    plt.close()

    return importance


def plot_actual_vs_predicted(y_true: np.ndarray,
                             results: dict,
                             class_names: np.ndarray,
                             figures_dir: Path) -> None:
    """Save actual vs predicted clay type plot."""
    plot_df = pd.DataFrame({"Sample": np.arange(1, len(y_true) + 1), "Actual": y_true})

    for model_name, model_result in results.items():
        plot_df[model_name] = model_result["y_pred"]

    plt.figure(figsize=(10, 6))
    plt.plot(plot_df["Sample"], plot_df["Actual"], marker="o", label="Actual", linewidth=2)

    for model_name in results.keys():
        plt.plot(plot_df["Sample"], plot_df[model_name], marker="s", linestyle="--", label=model_name)

    plt.title("Actual vs Predicted Clay Type")
    plt.xlabel("Sample Number")
    plt.ylabel("Encoded Class")
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_predicted.png", dpi=300)
    plt.close()


def plot_robustness_results(robustness_results: dict, figures_dir: Path) -> None:
    """Save robustness accuracy plot."""
    plt.figure(figsize=(7, 5))
    plt.bar(list(robustness_results.keys()), list(robustness_results.values()))
    plt.title("Robustness Check Accuracy (Noisy Inputs)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(figures_dir / "robustness_accuracy.png", dpi=300)
    plt.close()


# -------------------------------------------------------------------
# Modeling functions
# -------------------------------------------------------------------
def define_models() -> dict:
    """Define machine learning models using sklearn pipelines."""
    models = {
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=3)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)),
        ]),
    }
    return models


def evaluate_models_loocv(models: dict,
                          X: pd.DataFrame,
                          y_encoded: np.ndarray,
                          class_names: np.ndarray,
                          figures_dir: Path,
                          results_dir: Path) -> dict:
    """Evaluate all models using Leave-One-Out Cross-Validation."""
    loo = LeaveOneOut()
    results = {}
    report_text = []

    for model_name, model in models.items():
        y_pred = cross_val_predict(model, X, y_encoded, cv=loo)
        accuracy = accuracy_score(y_encoded, y_pred)

        results[model_name] = {
            "y_pred": y_pred,
            "accuracy": accuracy,
        }

        report = classification_report(y_encoded, y_pred, target_names=class_names)
        report_text.append("=" * 60)
        report_text.append(f"{model_name} - LOOCV Results")
        report_text.append("=" * 60)
        report_text.append(f"Accuracy: {accuracy:.4f}")
        report_text.append("\nClassification Report:")
        report_text.append(report)
        report_text.append("\n")

        plot_confusion_matrix(y_encoded, y_pred, class_names, model_name, figures_dir)

    save_text("\n".join(report_text), results_dir / "model_evaluation_report.txt")
    return results


def fit_models_on_full_data(models: dict, X: pd.DataFrame, y_encoded: np.ndarray) -> dict:
    """Fit all models on the full dataset for final model interpretation."""
    fitted_models = {}
    for model_name, model in models.items():
        model.fit(X, y_encoded)
        fitted_models[model_name] = model
    return fitted_models


# -------------------------------------------------------------------
# Robustness and prediction functions
# -------------------------------------------------------------------
def add_noise_to_features(X: pd.DataFrame, noise_level: float = 0.05) -> pd.DataFrame:
    """Add Gaussian noise to numeric input features."""
    X_noisy = X.copy()

    for col in X_noisy.columns:
        std_col = X_noisy[col].std()
        noise = np.random.normal(0, noise_level * std_col, size=len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise

    return X_noisy


def robustness_check(models: dict,
                     X: pd.DataFrame,
                     y_encoded: np.ndarray,
                     results_dir: Path) -> dict:
    """Evaluate model sensitivity to small input noise."""
    loo = LeaveOneOut()
    X_noisy = add_noise_to_features(X, noise_level=0.05)
    robustness_results = {}

    lines = ["Robustness Check using 5% Gaussian Noise", "=" * 45]

    for model_name, model in models.items():
        y_pred_noisy = cross_val_predict(model, X_noisy, y_encoded, cv=loo)
        accuracy_noisy = accuracy_score(y_encoded, y_pred_noisy)
        robustness_results[model_name] = accuracy_noisy
        lines.append(f"{model_name}: {accuracy_noisy:.4f}")

    save_text("\n".join(lines), results_dir / "robustness_results.txt")
    return robustness_results


def predict_new_sample(fitted_models: dict,
                       label_encoder: LabelEncoder,
                       feature_columns: list[str],
                       results_dir: Path) -> None:
    """Predict clay type for one example new sample."""
    new_sample = pd.DataFrame({
        "d90": [20],
        "d10": [2],
        "d50": [8],
        "water absorption": [12],
        "Al/Si": [0.45],
        "SO3": [0.8],
        "CaCO3": [4.5],
        "MgO": [1.2],
        "Na2O": [0.3],
        "Heat": [150],
    })

    # Reorder columns to match training data.
    # This assumes the dataset contains the same feature names.
    new_sample = new_sample[feature_columns]

    lines = ["Prediction for New Sample", "=" * 30]

    for model_name, model in fitted_models.items():
        prediction = model.predict(new_sample)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        lines.append(f"{model_name}: {predicted_label}")

    save_text("\n".join(lines), results_dir / "new_sample_prediction.txt")


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def run_pipeline(data_path: Path, target_column: str, output_dir: Path) -> None:
    """Run the full clay classification machine learning pipeline."""
    results_dir, figures_dir = create_output_folders(output_dir)

    # Load and clean data
    data = load_data(data_path)
    initial_shape = data.shape
    data = clean_data(data)
    cleaned_shape = data.shape

    X, y = split_features_target(data, target_column)
    y_encoded, label_encoder = encode_labels(y)

    # Save basic dataset information
    dataset_summary = [
        "Dataset Summary",
        "=" * 30,
        f"Initial shape: {initial_shape}",
        f"Shape after dropping missing rows: {cleaned_shape}",
        f"Target column: {target_column}",
        "\nInput features:",
        "\n".join(X.columns.tolist()),
        "\nClass mapping:",
    ]

    for idx, class_name in enumerate(label_encoder.classes_):
        dataset_summary.append(f"{class_name} -> {idx}")

    save_text("\n".join(dataset_summary), results_dir / "dataset_summary.txt")

    # Save basic visualizations
    plot_class_distribution(data, target_column, figures_dir)
    plot_correlation_heatmap(X, figures_dir)
    plot_pca(X, y, figures_dir)

    # Model training and evaluation
    models = define_models()
    results = evaluate_models_loocv(models, X, y_encoded, label_encoder.classes_, figures_dir, results_dir)
    plot_accuracy_comparison(results, figures_dir)
    plot_actual_vs_predicted(y_encoded, results, label_encoder.classes_, figures_dir)

    # Fit models on full dataset for interpretation
    fitted_models = fit_models_on_full_data(models, X, y_encoded)

    # Feature importance for tree-based models
    feature_importance_lines = []
    for model_name in ["Decision Tree", "Random Forest"]:
        model = fitted_models[model_name].named_steps["model"]
        importance = plot_feature_importance(model, X.columns.tolist(), model_name, figures_dir)
        feature_importance_lines.append(f"\n{model_name} Feature Importance")
        feature_importance_lines.append("=" * 40)
        feature_importance_lines.append(importance.to_string())

    save_text("\n".join(feature_importance_lines), results_dir / "feature_importance.txt")

    # Robustness check
    robustness_results = robustness_check(models, X, y_encoded, results_dir)
    plot_robustness_results(robustness_results, figures_dir)

    # Optional new-sample prediction
    try:
        predict_new_sample(fitted_models, label_encoder, X.columns.tolist(), results_dir)
    except KeyError:
        message = (
            "New sample prediction was skipped because the example new-sample "
            "feature names do not exactly match the dataset feature columns."
        )
        save_text(message, results_dir / "new_sample_prediction.txt")

    print("Pipeline completed successfully.")
    print(f"Results saved in: {results_dir}")
    print(f"Figures saved in: {figures_dir}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Clay Classification ML Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/Clay classification.csv",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Output",
        help="Name of the target/output column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Folder where results and figures will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(
        data_path=Path(args.data),
        target_column=args.target,
        output_dir=Path(args.output),
    )
