
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"


def main():
    artifact = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("В датасете отсутствует столбец Class.")

    features = artifact["features"]
    model = artifact["model"]
    threshold = float(artifact["threshold"])

    X = df[features]
    y_true = df["Class"].astype(int)

    probabilities = model.predict_proba(X)[:, 1]
    y_pred = (probabilities >= threshold).astype(int)

    print("\n=== MERCURY: ОЦЕНКА МОДЕЛИ ===")
    print(f"Количество транзакций: {len(df):,}")
    print(f"Порог классификации: {threshold:.2f}")
    print(f"Реальное мошенничество: {int(y_true.sum())}")
    print(f"Предсказанное мошенничество: {int(y_pred.sum())}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Обычная операция", "Мошенничество"],
            zero_division=0,
        )
    )

    print("=== CONFUSION MATRIX ===")
    print("Строки — реальные классы, столбцы — предсказания.")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\n=== PROBABILITY METRICS ===")
    print(f"PR-AUC (Average Precision): "
          f"{average_precision_score(y_true, probabilities):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, probabilities):.4f}")

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    print("\n=== THRESHOLD ANALYSIS ===")
    for target_threshold in (0.3, 0.5, 0.7):
        predictions = (probabilities >= target_threshold).astype(int)
        tp = int(((predictions == 1) & (y_true == 1)).sum())
        fp = int(((predictions == 1) & (y_true == 0)).sum())
        fn = int(((predictions == 0) & (y_true == 1)).sum())

        precision_value = tp / (tp + fp) if tp + fp else 0.0
        recall_value = tp / (tp + fn) if tp + fn else 0.0

        print(
            f"Порог {target_threshold:.1f}: "
            f"Precision={precision_value:.4f}, "
            f"Recall={recall_value:.4f}, "
            f"TP={tp}, FP={fp}, FN={fn}"
        )


if __name__ == "__main__":
    main()
