import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
import os

# Стандартный импорт RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Имя локального файла с очищенным датасетом Titanic\ n
LOCAL_CSV = 'titanic_clean.csv'

@st.cache_data
def load_data(path: str = LOCAL_CSV) -> pd.DataFrame:
    """
    Загружает очищенный CSV с данными Titanic.
    """
    if not os.path.exists(path):
        st.error(f"Файл '{path}' не найден. Поместите его рядом с этим скриптом.")
        st.stop()
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобрабатывает DataFrame:
    - Заполняет пропущенные значения и кодирует категории;
    - Оставляет только числовые признаки для модели и колонку 'Survived'.
    """
    df = df.copy()
    # Заполнение пропусков
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Кодирование пола
    if 'Sex' in df.columns:
        df['Sex_male'] = df['Sex'].map({'male': 1, 'female': 0})

    # Кодирование порта посадки
    if 'Embarked' in df.columns:
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df = pd.concat([df, embarked_dummies], axis=1)

    # Определяем признаки, которые используем в модели
    feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male'] + [c for c in df.columns if c.startswith('Embarked_')]

    # Проверка наличия всех признаков, заполняем отсутствующие нулями
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Оставляем только нужные колонки
    df_proc = df[['Survived'] + feature_cols].dropna()
    return df_proc


def user_input() -> pd.DataFrame:
    """Собирает ввод пользователя из sidebar."""
    st.sidebar.header("Параметры пассажира")
    pclass = st.sidebar.selectbox("Класс пассажира", [1, 2, 3], index=1)
    sex = st.sidebar.selectbox("Пол", ['male', 'female'], index=0)
    age = st.sidebar.slider("Возраст", 0.0, 100.0, 30.0)
    sibsp = st.sidebar.slider("Сиблинги/супруги на борту", 0, 8, 0)
    parch = st.sidebar.slider("Родители/дети на борту", 0, 6, 0)
    fare = st.sidebar.slider("Стоимость билета", 0.0, 600.0, 32.0)
    embarked = st.sidebar.selectbox("Порт посадки", ['C', 'Q', 'S'], index=2)

    return pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }])


def transform_input(input_df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    """Преобразует ввод пользователя в формат признаков модели."""
    df = input_df.copy()

    # Кодирование пола
    if 'Sex_male' in model_cols:
        df['Sex_male'] = df['Sex'].map({'male': 1, 'female': 0})

    # Кодирование порта посадки
    for col in model_cols:
        if col.startswith('Embarked_'):
            code = col.split('_')[1]
            df[col] = (df['Embarked'] == code).astype(int)

    # Удаляем исходные колонки
    df.drop(['Sex', 'Embarked'], axis=1, inplace=True, errors='ignore')

    # Заполнение отсутствующих столбцов
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0

    return df[model_cols]


def main():
    st.title("Прогноз выживания пассажиров Titanic")
    st.markdown("Приложение Streamlit + RandomForestClassifier")

    # Загрузка данных
    df_raw = load_data()
    st.subheader("Исходные данные")
    st.dataframe(df_raw.head())

    # Предобработка
    df = preprocess(df_raw)

    # Определение X и y
    if 'Survived' not in df.columns:
        st.error("В данных отсутствует столбец 'Survived'.")
        st.stop()
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Гиперпараметры в sidebar
    st.sidebar.header("Гиперпараметры RandomForest")
    n_estimators = st.sidebar.slider("Число деревьев", 10, 200, 100, step=10)
    max_depth = st.sidebar.slider("Максимальная глубина", 1, 20, 5)

    # Обучение модели
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Метрики модели")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write("**Confusion Matrix:**")
    st.write(cm)

    st.subheader("Важность признаков")
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    st.bar_chart(feat_imp.sort_values(ascending=False))

    # Пользовательский прогноз
    input_df = user_input()
    input_proc = transform_input(input_df, X_train.columns.tolist())
    prediction = model.predict(input_proc)[0]
    proba = model.predict_proba(input_proc)[0][1]

    st.subheader("Прогноз для введенных данных")
    st.write("Выжил" if prediction == 1 else "Не выжил")
    st.write(f"Вероятность выживания: {proba:.2f}")

if __name__ == '__main__':
    main()
