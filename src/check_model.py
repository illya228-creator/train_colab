from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"


def main():
    print("Проверка Mercury Fraud Detection")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    print("Модель загружена успешно")
    print("Алгоритм:", artifact["model_name"])
    print("Версия:", artifact["model_version"])
    print("Количество признаков:", len(artifact["features"]))
    print("Версия scikit-learn при обучении:",
          artifact["sklearn_version"])
    print("Порог:", artifact["threshold"])


if __name__ == "__main__":
    main()