
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
OUTPUT_PATH = ROOT / "data" / "predictions" / "results.csv"

BG = "#151922"
CARD = "#1D2532"
CHART_BG = "#252E3B"
BORDER = "#303B4B"

TEXT = "#E8EDF5"
MUTED = "#91A0B5"

BLUE = "#65A9F0"
RED = "#F07178"
GREEN = "#58C49A"
ORANGE = "#FF7A30"

WINDOW_STYLE = """
QMainWindow, QWidget {
    background-color: #151922;
    color: #E8EDF5;
    font-family: "Segoe UI";
    font-size: 11px;
}

QFrame#outer {
    background-color: #151922;
    border: 1px solid #303B4B;
    border-radius: 15px;
}

QFrame#metricCard, QFrame#chartCard {
    background-color: #1D2532;
    border: none;
    border-radius: 12px;
}

QLabel {
    background: transparent;
    border: none;
}

QLabel#muted {
    color: #91A0B5;
}

QLabel#metric {
    font-size: 27px;
    font-weight: 700;
}

QLabel#sectionTitle {
    font-size: 18px;
    font-weight: 750;
}

QLabel#chartTitle {
    font-size: 12px;
    font-weight: 650;
}

QLabel#statusBadge {
    color: #FF7A30;
    background-color: #3A2925;
    border-radius: 9px;
    padding: 4px 8px;
    font-size: 10px;
    font-weight: 700;
}
"""


# ============================================================
# LINE CHART
# ============================================================

class RiskBars(QWidget):
    """Линейный график оценок риска первых 1 000 транзакций."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scores = []
        self.setMinimumHeight(96)

    def set_scores(self, scores):
        self.scores = list(scores)
        self.update()

    def paintEvent(self, event):
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(CHART_BG))

        width = self.width()
        height = self.height()

        left = 8
        right = 8
        top = 8
        bottom = 8

        chart_width = width - left - right
        chart_height = height - top - bottom

        if chart_width <= 0 or chart_height <= 0 or not self.scores:
            painter.end()
            return

        values = np.asarray(self.scores[:1000], dtype=float)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=100.0,
            neginf=0.0,
        )
        values = np.clip(values, 0.0, 100.0)

        # Горизонтальные направляющие.
        painter.setPen(QPen(QColor("#303B4B"), 1))

        for fraction in (0.0, 0.5, 1.0):
            y = top + chart_height * fraction
            painter.drawLine(
                int(left),
                int(y),
                int(width - right),
                int(y),
            )

        # Координаты точек.
        points = []

        for index, value in enumerate(values):
            if len(values) == 1:
                x = left + chart_width / 2
            else:
                x = left + index * chart_width / (len(values) - 1)

            y = top + chart_height * (1.0 - value / 100.0)
            points.append((int(x), int(y)))

        # Соединяем точки линией.
        painter.setPen(
            QPen(
                QColor(BLUE),
                1.5,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )

        for index in range(1, len(points)):
            x1, y1 = points[index - 1]
            x2, y2 = points[index]
            painter.drawLine(x1, y1, x2, y2)

        painter.end()


# ============================================================
# DONUT CHART
# ============================================================

class DonutChart(QWidget):
    """Кольцевая диаграмма обычных и подозрительных операций."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.normal = 0
        self.suspicious = 0
        self.setMinimumHeight(115)

    def set_counts(self, normal, suspicious):
        self.normal = int(normal)
        self.suspicious = int(suspicious)
        self.update()

    def paintEvent(self, event):
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        total = self.normal + self.suspicious

        if total <= 0:
            painter.end()
            return

        side = min(100, self.height() - 8)
        x = 5
        y = (self.height() - side) // 2

        ring_rect = (
            x + 8,
            y + 8,
            side - 16,
            side - 16,
        )

        normal_angle = round(360 * 16 * self.normal / total)
        suspicious_angle = 360 * 16 - normal_angle

        # Фоновое кольцо.
        painter.setPen(
            QPen(
                QColor(BORDER),
                13,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.FlatCap,
            )
        )
        painter.drawArc(*ring_rect, 0, 360 * 16)

        # Обычные операции.
        painter.setPen(
            QPen(
                QColor(GREEN),
                13,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.FlatCap,
            )
        )
        painter.drawArc(
            *ring_rect,
            90 * 16,
            -normal_angle,
        )

        # Подозрительные операции.
        if self.suspicious > 0:
            painter.setPen(
                QPen(
                    QColor(RED),
                    13,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.FlatCap,
                )
            )
            painter.drawArc(
                *ring_rect,
                90 * 16 - normal_angle,
                -suspicious_angle,
            )

        # Центр намеренно оставляем пустым.
        # Центральный значок удалён.

        # Легенда.
        legend_x = x + side + 12
        legend_y = max(20, (self.height() - 38) // 2)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(GREEN))
        painter.drawEllipse(legend_x, legend_y, 9, 9)

        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            legend_x + 15,
            legend_y + 8,
            "Обычные",
        )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(RED))
        painter.drawEllipse(legend_x, legend_y + 23, 9, 9)

        painter.setPen(QColor(MUTED))
        painter.drawText(
            legend_x + 15,
            legend_y + 31,
            "Подозрительные",
        )

        painter.end()


# ============================================================
# ANALYSIS WORKER
# ============================================================

class AnalysisWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            result = self.analyze()
            self.completed.emit(result)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            ImportError,
            AttributeError,
            RuntimeError,
            EOFError,
        ) as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def analyze():
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Файл модели не найден:\n{MODEL_PATH}"
            )

        if not DATA_PATH.is_file():
            raise FileNotFoundError(
                f"Файл датасета не найден:\n{DATA_PATH}"
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            artifact = joblib.load(MODEL_PATH)

        version_warnings = [
            str(item.message)
            for item in caught
            if "InconsistentVersionWarning"
            in item.category.__name__
        ]

        if not isinstance(artifact, dict):
            raise TypeError(
                "Некорректный формат артефакта модели. "
                "Ожидался словарь с model, features и threshold."
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
                "Порог классификации должен быть от 0 до 1."
            )

        df = pd.read_csv(DATA_PATH)

        if df.empty:
            raise ValueError("Датасет пуст.")

        missing_features = [
            feature for feature in features
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                "В датасете отсутствуют признаки: "
                + ", ".join(missing_features)
            )

        X = df[features].apply(pd.to_numeric, errors="raise")

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(
                "Датасет содержит пропущенные "
                "или бесконечные значения."
            )

        probabilities = model.predict_proba(X)[:, 1]

        if len(probabilities) != len(df):
            raise ValueError(
                "Количество предсказаний не совпадает "
                "с количеством транзакций."
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Модель вернула некорректные вероятности."
            )

        result = df.copy()
        result.insert(0, "transaction_id", np.arange(len(df)))
        result["fraud_probability"] = probabilities
        result["risk_score"] = (probabilities * 100).round(2)
        result["is_suspicious"] = probabilities >= threshold

        result["risk_level"] = pd.cut(
            probabilities,
            bins=[-0.001, 0.3, 0.7, 1.001],
            labels=["Низкий", "Средний", "Высокий"],
        ).astype(str)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(OUTPUT_PATH, index=False)

        return {
            "df": result,
            "threshold": threshold,
            "version_warnings": version_warnings,
        }


# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fraud Detection")
        self.resize(550, 490)
        self.setMinimumSize(530, 465)
        self.setMaximumSize(650, 560)
        self.setStyleSheet(WINDOW_STYLE)

        self.worker = None
        self.result_df = None
        self.closing = False

        outer = QFrame()
        outer.setObjectName("outer")
        self.setCentralWidget(outer)

        root = QVBoxLayout(outer)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # Header.
        header = QHBoxLayout()
        brand = QVBoxLayout()
        brand.setSpacing(1)

        logo = QLabel("MERCURY")
        logo.setStyleSheet(
            "font-size: 27px; font-weight: 800; letter-spacing: 0.5px;"
        )

        subtitle = QLabel("FRAUD DETECTION / LOCAL ANALYTICS")
        subtitle.setObjectName("muted")
        subtitle.setStyleSheet("font-size: 9px;")

        brand.addWidget(logo)
        brand.addWidget(subtitle)

        header.addLayout(brand)
        header.addStretch()

        self.status = QLabel("Проверка модели")
        self.status.setObjectName("statusBadge")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.status)

        root.addLayout(header)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {BORDER};")
        root.addWidget(divider)

        # Metric cards.
        metrics = QGridLayout()
        metrics.setSpacing(8)

        self.total_metric = self.make_metric(
            metrics, 0, "ТРАНЗАКЦИИ", "—", BLUE
        )
        self.suspicious_metric = self.make_metric(
            metrics, 1, "ПОДОЗРИТЕЛЬНЫЕ", "—", RED
        )
        self.threshold_metric = self.make_metric(
            metrics, 2, "ПОРОГ", "—", "#FFB454"
        )

        root.addLayout(metrics)

        # Analytics heading.
        section_title = QLabel("АНАЛИТИКА")
        section_title.setObjectName("sectionTitle")
        root.addWidget(section_title)

        # Risk card.
        charts_row = QGridLayout()
        charts_row.setSpacing(12)

        risk_card = QFrame()
        risk_card.setObjectName("chartCard")

        risk_layout = QVBoxLayout(risk_card)
        risk_layout.setContentsMargins(12, 10, 12, 10)
        risk_layout.setSpacing(8)

        risk_title = QLabel("Риск по транзакциям")
        risk_title.setObjectName("chartTitle")
        risk_layout.addWidget(risk_title)

        self.risk_bars = RiskBars()
        risk_layout.addWidget(self.risk_bars)

        risk_caption = QLabel("Первые 1 000 операций")
        risk_caption.setObjectName("muted")
        risk_caption.setStyleSheet("font-size: 9px;")
        risk_layout.addWidget(risk_caption)

        # Class distribution card.
        classes_card = QFrame()
        classes_card.setObjectName("chartCard")

        classes_layout = QVBoxLayout(classes_card)
        classes_layout.setContentsMargins(12, 10, 12, 10)
        classes_layout.setSpacing(8)

        classes_title = QLabel("Классы операций")
        classes_title.setObjectName("chartTitle")
        classes_layout.addWidget(classes_title)

        self.donut = DonutChart()
        classes_layout.addWidget(self.donut)

        charts_row.addWidget(risk_card, 0, 0)
        charts_row.addWidget(classes_card, 0, 1)
        charts_row.setColumnStretch(0, 1)
        charts_row.setColumnStretch(1, 1)

        root.addLayout(charts_row)
        root.addStretch(1)

        footer = QLabel(
            "Локальная обработка данных · Риск требует проверки"
        )
        footer.setObjectName("muted")
        footer.setStyleSheet("font-size: 9px;")
        root.addWidget(footer)

        # Automatic model and dataset loading.
        QTimer.singleShot(250, self.run_analysis)

    @staticmethod
    def make_metric(grid, column, title_text, value, accent):
        card = QFrame()
        card.setObjectName("metricCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 11, 12, 10)
        layout.setSpacing(5)

        title = QLabel(title_text)
        title.setObjectName("muted")
        title.setStyleSheet("font-size: 10px;")

        metric = QLabel(value)
        metric.setObjectName("metric")
        metric.setStyleSheet(
            f"font-size: 27px; font-weight: 700; color: {accent};"
        )

        layout.addWidget(title)
        layout.addWidget(metric)
        grid.addWidget(card, 0, column)

        return metric

    def run_analysis(self):
        if self.worker is not None and self.worker.isRunning():
            return

        self.status.setText("Выполняется анализ")

        self.worker = AnalysisWorker()
        self.worker.completed.connect(self.on_completed)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_completed(self, payload):
        df = payload["df"]
        self.result_df = df

        suspicious = int(df["is_suspicious"].sum())
        normal = len(df) - suspicious

        self.total_metric.setText(
            f"{len(df):,}".replace(",", " ")
        )
        self.suspicious_metric.setText(
            f"{suspicious:,}".replace(",", " ")
        )
        self.threshold_metric.setText(
            f'{payload["threshold"]:.2f}'
        )

        self.risk_bars.set_scores(
            df["risk_score"].head(1000).tolist()
        )
        self.donut.set_counts(normal, suspicious)

        if payload["version_warnings"]:
            self.status.setText("Проверка модели")
        else:
            self.status.setText("Модель готова")

    def on_failed(self, message):
        self.status.setText("Ошибка анализа")
        QMessageBox.critical(self, "Ошибка анализа", message)

    def on_worker_finished(self):
        worker = self.worker
        self.worker = None

        if worker is not None:
            worker.deleteLater()

        if self.closing:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.closing = True
            self.status.setText("Завершение анализа...")
            event.ignore()
            return

        event.accept()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
