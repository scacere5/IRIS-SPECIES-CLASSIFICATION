import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def cargar_datos():
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )
    df["species"] = iris.target
    df["species_name"] = df["species"].map(
        {i: name for i, name in enumerate(iris.target_names)}
    )
    return df, iris


@st.cache_resource
def entrenar_modelo(df):
    X = df.iloc[:, 0:4] 
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred
    }

    return clf, metrics


def graficar_3d(df, new_point=None):
    x_col = "petal length (cm)"
    y_col = "petal width (cm)"
    z_col = "sepal length (cm)"

    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color="species_name",
        opacity=0.7,
        title="Iris dataset en 3D"
    )

    if new_point is not None:
        fig.add_scatter3d(
            x=[new_point[x_col]],
            y=[new_point[y_col]],
            z=[new_point[z_col]],
            mode="markers",
            marker=dict(size=8, symbol="diamond"),
            name="Nueva muestra"
        )

    return fig

def main():
    st.set_page_config(
        page_title="Iris Species Classification",
        page_icon="🌸",
        layout="wide"
    )

    st.title("🌸 Iris Species Classification")
    st.markdown("""
    Proyecto final de **Data Mining** By Samuel Esteban Caceres Izquierdo.  
    Modelo de clasificación para predecir la especie de una flor Iris a partir de 4 medidas:
    - Sepal length  
    - Sepal width  
    - Petal length  
    - Petal width  
    """)

    df, iris = cargar_datos()
    model, metrics = entrenar_modelo(df)

    st.sidebar.header("Metodología (Workflow)")
    st.sidebar.markdown("""
    1. **Comprensión de datos**  
       - Dataset Iris (150 muestras, 3 especies).  
    2. **Preprocesamiento**  
       - División train/test (80/20).  
    3. **Modelado**  
       - Random Forest Classifier.  
    4. **Evaluación**  
       - Accuracy, Precision, Recall, F1.  
    5. **Dashboard**  
       - Visualizaciones + predicción interactiva.
    """)

    page = st.sidebar.radio(
        "Navegación",
        ["Exploración de datos", "Modelo y métricas", "Predicción interactiva"]
    )

    if page == "Exploración de datos":
        st.subheader("🔎 Exploración de datos")

        st.write("Primeras filas del dataset:")
        st.dataframe(df.head())

        st.write("Dimensiones del dataset:")
        st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

        st.write("Distribución de clases:")
        st.bar_chart(df["species_name"].value_counts())

        st.write("Histogramas de las características:")
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        cols = df.columns[:4]
        for col, ax in zip(cols, axes.flatten()):
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)
        plt.tight_layout()
        st.pyplot(fig)

        st.write("Relación entre características (scatter matrix simplificado):")
        fig2 = px.scatter_matrix(
            df,
            dimensions=iris.feature_names,
            color="species_name",
            title="Scatter Matrix"
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif page == "Modelo y métricas":
        st.subheader("📊 Modelo y métricas")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['precision']:.3f}")
        col3.metric("Recall", f"{metrics['recall']:.3f}")
        col4.metric("F1-score", f"{metrics['f1']:.3f}")

        st.write("Matriz de confusión:")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(
            metrics["conf_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        st.pyplot(fig_cm)

        st.write("Reporte de clasificación:")
        report = classification_report(
            metrics["y_test"],
            metrics["y_pred"],
            target_names=iris.target_names,
            output_dict=False
        )
        st.text(report)

    else:
        st.subheader("🧪 Predicción interactiva")

        st.markdown("Ingresa las medidas de la flor:")

        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.slider(
                "Sepal length (cm)",
                float(df["sepal length (cm)"].min()),
                float(df["sepal length (cm)"].max()),
                float(df["sepal length (cm)"].mean())
            )
            sepal_width = st.slider(
                "Sepal width (cm)",
                float(df["sepal width (cm)"].min()),
                float(df["sepal width (cm)"].max()),
                float(df["sepal width (cm)"].mean())
            )

        with col2:
            petal_length = st.slider(
                "Petal length (cm)",
                float(df["petal length (cm)"].min()),
                float(df["petal length (cm)"].max()),
                float(df["petal length (cm)"].mean())
            )
            petal_width = st.slider(
                "Petal width (cm)",
                float(df["petal width (cm)"].min()),
                float(df["petal width (cm)"].max()),
                float(df["petal width (cm)"].mean())
            )

        input_data = np.array(
            [[sepal_length, sepal_width, petal_length, petal_width]]
        )
        pred = model.predict(input_data)[0]
        pred_name = iris.target_names[pred]

        st.markdown(
            f"### 🌼 La especie predicha es: **{pred_name}**"
        )

        new_point = {
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width,
            "species_name": f"Pred: {pred_name}"
        }

        fig3d = graficar_3d(df, new_point=new_point)
        st.plotly_chart(fig3d, use_container_width=True)


if __name__ == "__main__":
    main()
