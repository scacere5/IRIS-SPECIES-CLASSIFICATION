# 🌸 Iris Species Classification

Proyecto final del curso **Data Mining** – Universidad de la Costa.  
El objetivo es entrenar y desplegar un modelo de clasificación capaz de predecir la especie de una flor *Iris* a partir de cuatro características numéricas.

## 👥 Integrantes

- Samuel Esteban Caceres Izquierdo  

## 🎯 Objetivo

Construir un pipeline de minería de datos end-to-end que incluya:

1. Comprensión de los datos.
2. Preprocesamiento.
3. Entrenamiento de un modelo de clasificación.
4. Evaluación mediante métricas (Accuracy, Precision, Recall, F1).
5. Despliegue en un dashboard interactivo con Streamlit.

## 📊 Dataset

Se utiliza el dataset **Iris** incluido en `scikit-learn`, que contiene:

- 150 muestras
- 4 características:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- 3 clases (especies):
  - Iris setosa
  - Iris versicolor
  - Iris virginica

## 🧠 Metodología (Workflow)

1. **Data Understanding**
   - Carga del dataset desde `sklearn.datasets.load_iris`.
   - Exploración de dimensiones, primeras filas y distribución de clases.
   - Visualizaciones iniciales: histogramas y scatter plots.

2. **Preprocessing**
   - Separación de variables predictoras (X) y etiqueta (y).
   - División en conjuntos de entrenamiento (80 %) y prueba (20 %).

3. **Modeling**
   - Entrenamiento de un **RandomForestClassifier** como modelo principal.
   - Justificación:
     - Adecuado para datos tabulares.
     - Robusto ante ruido.
     - No requiere escalado estricto de variables.

4. **Evaluation**
   - Métricas calculadas sobre el conjunto de prueba:
     - Accuracy
     - Precision
     - Recall
     - F1-score
   - Matriz de confusión y reporte de clasificación.

5. **Deployment – Dashboard con Streamlit**
   El dashboard incluye:
   - Pestaña de **exploración de datos**.
   - Pestaña de **modelo y métricas**.
   - Pestaña de **predicción interactiva**, donde el usuario ingresa:
     - Sepal length
     - Sepal width
     - Petal length
     - Petal width  
     y obtiene:
     - La especie predicha.
     - La visualización de la nueva muestra en un gráfico 3D respecto al dataset original.

## 🖥️ Ejecución del proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/usuario/IRIS-SPECIES-CLASSIFICATION.git
cd IRIS-SPECIES-CLASSIFICATION
