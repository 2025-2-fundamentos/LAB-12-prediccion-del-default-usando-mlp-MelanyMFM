# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class CreditDefaultLearner:
    def __init__(self):
        self.model_pipeline = None
        self.categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
        self.numerical_features = [
            "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
            "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
            "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", 
            "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
        ]

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transformación y limpieza de datos fluida."""
        return (
            df.drop(columns=["ID"])
            .rename(columns={"default payment next month": "default"})
            .dropna()
            .query("EDUCATION != 0 and MARRIAGE != 0")
            .assign(EDUCATION=lambda x: x["EDUCATION"].clip(upper=4))
        )

    def load_data(self, train_path, test_path):
        raw_train = pd.read_csv(train_path, compression="zip")
        raw_test = pd.read_csv(test_path, compression="zip")

        self.train_df = self._preprocess(raw_train)
        self.test_df = self._preprocess(raw_test)

    def build_pipeline(self):
        # Preprocesamiento de columnas
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                ("num", StandardScaler(), self.numerical_features),
            ]
        )

        # Definición del Pipeline secuencial
        return Pipeline([
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("pca", PCA()),
            ("classifier", MLPClassifier(max_iter=15000, random_state=17)),
        ])

    def train(self):
        X_train = self.train_df.drop(columns="default")
        y_train = self.train_df["default"]

        base_pipeline = self.build_pipeline()

        # Configuración exacta de hiperparámetros
        params = {
            "pca__n_components": [None],
            "feature_selection__k": [20],
            "classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
            "classifier__alpha": [0.26],
            "classifier__learning_rate_init": [0.001],
        }

        self.model_pipeline = GridSearchCV(
            base_pipeline,
            param_grid=params,
            cv=10,
            scoring="balanced_accuracy",
            n_jobs=-1,
            verbose=2
        )

        self.model_pipeline.fit(X_train, y_train)

    def save_artifacts(self, model_path, metrics_path):
        # 1. Guardar Modelo
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with gzip.open(model_path, "wb") as f:
            pickle.dump(self.model_pipeline, f)

        # 2. Calcular Métricas
        metrics_data = []
        datasets = {
            "train": (self.train_df.drop(columns="default"), self.train_df["default"]),
            "test": (self.test_df.drop(columns="default"), self.test_df["default"])
        }

        # Generar métricas (Train primero, luego Test)
        for name, (X, y) in datasets.items():
            y_pred = self.model_pipeline.predict(X)
            metrics_data.append({
                "type": "metrics",
                "dataset": name,
                "precision": round(precision_score(y, y_pred), 4),
                "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
                "recall": round(recall_score(y, y_pred), 4),
                "f1_score": round(f1_score(y, y_pred), 4),
            })

        # Generar matrices de confusión (Train primero, luego Test)
        for name, (X, y) in datasets.items():
            y_pred = self.model_pipeline.predict(X)
            cm = confusion_matrix(y, y_pred)
            metrics_data.append({
                "type": "cm_matrix",
                "dataset": name,
                "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
                "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
            })

        # 3. Guardar JSONL
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            for record in metrics_data:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    learner = CreditDefaultLearner()
    learner.load_data(
        "files/input/train_data.csv.zip", 
        "files/input/test_data.csv.zip"
    )
    
    print("Entrenando red neuronal...")
    learner.train()
    
    print("Guardando resultados...")
    learner.save_artifacts(
        "files/models/model.pkl.gz", 
        "files/output/metrics.json"
    )
