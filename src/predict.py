
import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
DEFAULT_OUTPUT = ROOT / "data" / "predictions" / "results.csv"


def predict_transactions(input_path: Path, output_path: Path) -> None:
    """Загружает модель, анализирует транзакции и сохраняет результаты."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Входной CSV не найден: {input_path}"
        )

    print("Загрузка модели...")
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    features = artifact["features"]
    threshold = float(artifact["threshold"])

    print("Чтение транзакций...")
    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Входной CSV пуст.")

    missing = [feature for feature in features if feature not in df.columns]

    if missing:
        raise ValueError(
            "В CSV отсутствуют необходимые признаки: "
            + ", ".join(missing)
        )

    # Сохраняем исходный порядок признаков,
    # использованный при обучении модели.
    X = df[features].apply(pd.to_numeric, errors="raise")

    if X.isna().any().any():
        raise ValueError(
            "В признаках есть пропуски. "
            "Проверь входные данные и обработку пропусков."
        )

    print("Вычисление оценок...")
    probabilities = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["fraud_probability"] = probabilities
    result["risk_score"] = (probabilities * 100).round(2)
    result["is_suspicious"] = probabilities >= threshold

    result["risk_level"] = pd.cut(
        probabilities,
        bins=[-0.001, 0.3, 0.7, 1.001],
        labels=["Низкий", "Средний", "Высокий"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    suspicious_count = int(result["is_suspicious"].sum())

    print("\nАнализ завершён.")
    print(f"Всего транзакций: {len(result)}")
    print(f"Подозрительных операций: {suspicious_count}")
    print(f"Порог классификации: {threshold}")
    print(f"Результат сохранён: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mercury Fraud Detection — локальный анализ транзакций"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Путь к входному CSV-файлу",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Путь для сохранения результатов",
    )

    args = parser.parse_args()

    predict_transactions(
        input_path=Path(args.input).resolve(),
        output_path=Path(args.output).resolve(),
    )


if __name__ == "__main__":
    main()
