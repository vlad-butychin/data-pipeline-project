#ЧТЕНИЕ ДАННЫХ
# ===== Основа =====
import pandas as pd
import numpy as np
import logging
import requests as rq

# ===== Данные =====
from sqlalchemy import create_engine

# ===== Статистика и валидация =====
from scipy import stats

# ===== ML =====
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ===== Визуализация =====
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Репорты =====
import openpyxl
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def from_excel(path: str,encoding = 'UTF-8') -> pd.DataFrame:
    return pd.read_excel(path)


def from_csv(path: str,encoding = 'UTF-8') -> pd.DataFrame:
    return pd.read_csv(path)



def from_api(url: str, params: dict = None, headers: dict = None) -> pd.DataFrame:
    """
    Получает данные из API и преобразует в DataFrame.
    url: адрес API
    params: словарь параметров запроса
    headers: словарь заголовков
    """
    response = rq.get(url, params=params, headers=headers)
    response.raise_for_status()  # выбросит ошибку если запрос неудачный
    df = pd.DataFrame(response.json())
    return df



def from_sql(connection_string: str, query: str) -> pd.DataFrame:
    engine = create_engine(connection_string)
    return pd.read_sql(query, engine)
# conn_str = "postgresql://postgres:password@localhost:5432/demo"
# query = "SELECT"
# df = from_sql(conn_str, query)



#df='func'
# print(df.head())
# print(df.info())

#Валидация
import logging
from scipy import stats
import pandas as pd

logging.basicConfig(
    filename="validation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def validation(df):
    logging.info("=== Начало валидации ===")

    try:
        # Пропуски
        missing = df.isna().sum().to_dict()
        logging.info(f"Пропуски: {missing}")

        # Типы данных
        info = df.dtypes.to_dict()
        logging.info(f"Типы данных: {info}")

        # Дубликаты
        duplicates = df.duplicated().sum()
        logging.info(f"Дубликаты: {duplicates}")

        # Выбросы
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            if df[col].std() == 0:
                logging.info(f"Выбросы в {col}: нельзя вычислить (std = 0)")
                continue

            z = stats.zscore(df[col].dropna())
            outliers = int((abs(z) > 3).sum())

            logging.info(f"Выбросы в {col}: {outliers}")

        logging.info("=== Конец валидации ===")

    except Exception as e:
        logging.error(f"Ошибка валидации: {e}")

    print("Валидация выполнена. Проверяй validation.log")

#Отчистка данных
def data_clean1(df, threshold=50):
    df = df.copy()

    # --- 1. Числовые колонки ---
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        missing_percent = df[col].isna().sum() * 100 / len(df)
        if missing_percent > threshold:
            df = df.drop(columns=[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    # --- 2. Строковые колонки ---
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        missing_percent = df[col].isna().sum() * 100 / len(df)
        if missing_percent > threshold:
            df = df.drop(columns=[col])
        else:
            mode_val = df[col].mode()
            fill_value = mode_val[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    # --- 3. Даты ---
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df



#Модели
def train_model(df, target, task=None):
    if target not in df.columns:
        raise ValueError("Такой переменной нет в данных!")

    df = df.copy()
    y = df[target]
    X = df.drop(columns=[target], errors='ignore')

    # Удаляем ID
    id_cols = [c for c in X.columns if "id" in c.lower()]
    X = X.drop(columns=id_cols, errors='ignore')

    # Даты
    for col in X.columns:
        if "date" in col.lower():
            X[col] = pd.to_datetime(X[col], errors="coerce")
            X[col] = X[col].astype("int64") // 10**9

    # Определяем задачу
    if task is None:
        task = "classification" if y.nunique() <= 10 else "regression"

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===== КОДИРОВАНИЕ =====
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # выравниваем колонки
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # ===== МАСШТАБИРОВАНИЕ =====
    numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns
    numeric_cols = [c for c in numeric_cols if "id" not in c.lower()]

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # ===== МОДЕЛЬ =====
    if task == "regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\n=== REGRESSION METRICS ===")
        print("R2:", r2_score(y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

        coef = model.coef_

    else:
        y_train = y_train.astype("category").cat.codes
        y_test = y_test.astype("category").cat.codes

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\n=== CLASSIFICATION METRICS ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, average="macro"))
        print("Recall:", recall_score(y_test, y_pred, average="macro"))
        print("F1:", f1_score(y_test, y_pred, average="macro"))

        coef = model.coef_.mean(axis=0)

    importance = pd.Series(coef, index=X_train.columns).sort_values()
    print("\n=== FEATURE IMPORTANCE ===")
    print(importance)
    return model, X_train.columns, coef



#Репорт
def generate_report(df):
    report = {}

    report["shape"] = df.shape
    report["missing_percent"] = df.isna().mean() * 100
    report["basic_stats"] = df.describe(include='all')
    report["correlation"] = df.corr(numeric_only=True)

    return report

#Визуализация


def plot_hist(df, col):
    plt.figure(figsize=(8,5))
    plt.hist(df[col], bins=30, color='skyblue')
    plt.title(f"Histogram of {col}")
    plt.show()



def plot_correlation(df):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()


import openpyxl

def save_report_excel(report, filename="report.xlsx"):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Report"

    ws.append(["Metric", "Value"])

    ws.append(["Rows, Columns", str(report["shape"])])
    ws.append(["Missing %", str(report["missing_percent"].to_dict())])

    ws.append([])
    ws.append(["Basic Stats"])
    for col in report["basic_stats"].columns:
        ws.append([col, str(report["basic_stats"][col].to_dict())])

    wb.save(filename)
    print(f"Excel отчёт сохранён как {filename}")


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def save_report_pdf(report, filename="report.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 10)

    y = 750
    c.drawString(30, y, "DATA REPORT")
    y -= 20

    c.drawString(30, y, f"Shape: {report['shape']}")
    y -= 20

    c.drawString(30, y, "Missing %:")
    y -= 15
    for col, val in report["missing_percent"].items():
        c.drawString(40, y, f"{col}: {val:.2f}%")
        y -= 15

    c.drawString(30, y, "Basic Stats:")
    y -= 15
    for col in report["basic_stats"].columns:
        c.drawString(40, y, f"{col}: {report['basic_stats'][col].to_dict()}")
        y -= 15

    c.save()
    print(f"PDF отчёт сохранён как {filename}")



logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main_pipeline(path, target):
    print("=== START PIPELINE ===")
    logging.info("Пайплайн запущен")

    # 1. Загрузка
    df = from_csv(path)
    logging.info("Данные загружены")

    # 2. Валидация
    validation(df)
    logging.info("Валидация завершена")

    # 3. Очистка
    df_clean = data_clean1(df)
    logging.info("Очистка завершена")

    # 4. Модель
    model, features, coef = train_model(df_clean, target)
    logging.info("Модель обучена")

    # 5. Отчёт
    report = generate_report(df_clean)
    save_report_excel(report, "report.xlsx")
    save_report_pdf(report, "report.pdf")
    logging.info("Отчёты сохранены")

    # 6. Визуализация
    plot_hist(df_clean, target)
    plot_correlation(df_clean)
    logging.info("Визуализация выполнена")

    print("=== END PIPELINE ===")
    return model, report
