
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"


# ============================================================
# MODEL EVALUATION
# ============================================================

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}"
        )

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Датасет не найден: {DATA_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise TypeError(
            "Некорректный формат модели: ожидался словарь."
        )

    required_keys = {"model", "features", "threshold"}
    missing_keys = required_keys.difference(artifact)

    if missing_keys:
        raise KeyError(
            "В артефакте отсутствуют поля: "
            + ", ".join(sorted(missing_keys))
        )

    model = artifact["model"]
    features = artifact["features"]
    threshold = float(artifact["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Порог классификации должен находиться в диапазоне от 0 до 1."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Датасет пуст.")

    if "Class" not in df.columns:
        raise ValueError(
            "В датасете отсутствует целевой столбец Class."
        )

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "В датасете отсутствуют признаки: "
            + ", ".join(missing_features)
        )

    X = df[features].apply(pd.to_numeric, errors="raise")
    y_true = df["Class"].astype(int)

    if not y_true.isin([0, 1]).all():
        raise ValueError(
            "Столбец Class должен содержать только 0 и 1."
        )

    probabilities = model.predict_proba(X)[:, 1]
    y_pred = (probabilities >= threshold).astype(int)

    # --------------------------------------------------------
    # GENERAL INFORMATION
    # --------------------------------------------------------

    print("\n=== MERCURY: ОЦЕНКА МОДЕЛИ ===")
    print(f"Количество транзакций: {len(df):,}".replace(",", " "))
    print(f"Порог классификации: {threshold:.2f}")
    print(f"Реальное мошенничество: {int(y_true.sum())}")
    print(f"Предсказанное мошенничество: {int(y_pred.sum())}")

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Обычная операция",
                "Мошенничество",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    print("=== CONFUSION MATRIX ===")
    print("Строки — реальные классы, столбцы — предсказания.")
    print(matrix)
    print(f"TN — верно определены обычные операции: {tn}")
    print(f"FP — ложные срабатывания: {fp}")
    print(f"FN — пропущенное мошенничество: {fn}")
    print(f"TP — обнаруженное мошенничество: {tp}")

    # --------------------------------------------------------
    # PROBABILITY METRICS
    # --------------------------------------------------------

    print("\n=== PROBABILITY METRICS ===")
    print(
        "PR-AUC (Average Precision): "
        f"{average_precision_score(y_true, probabilities):.4f}"
    )
    print(
        f"ROC-AUC: {roc_auc_score(y_true, probabilities):.4f}"
    )

    # --------------------------------------------------------
    # THRESHOLD COMPARISON
    # --------------------------------------------------------

    print("\n=== THRESHOLD ANALYSIS ===")

    for candidate_threshold in (0.3, 0.5, 0.7):
        predictions = (
            probabilities >= candidate_threshold
        ).astype(int)

        tp_candidate = int(
            ((predictions == 1) & (y_true == 1)).sum()
        )
        fp_candidate = int(
            ((predictions == 1) & (y_true == 0)).sum()
        )
        fn_candidate = int(
            ((predictions == 0) & (y_true == 1)).sum()
        )

        precision = (
            tp_candidate / (tp_candidate + fp_candidate)
            if tp_candidate + fp_candidate
            else 0.0
        )

        recall = (
            tp_candidate / (tp_candidate + fn_candidate)
            if tp_candidate + fn_candidate
            else 0.0
        )

        print(
            f"Порог {candidate_threshold:.1f}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"TP={tp_candidate}, "
            f"FP={fp_candidate}, "
            f"FN={fn_candidate}"
        )

    # --------------------------------------------------------
    # EVALUATION WARNING
    # --------------------------------------------------------

    print("\n=== ВАЖНО ===")
    print(
        "Если модель обучалась на этих же данных, "
        "метрики не являются независимой оценкой качества."
    )
    print(
        "Для окончательной проверки нужна отдельная тестовая выборка."
    )


if __name__ == "__main__":
    main()
