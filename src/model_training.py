"""
EduFlow Intelligence — Módulo de Treinamento de Modelos
=======================================================
Treina e compara três modelos de classificação para lead scoring:
  - Regressão Logística (baseline)
  - Random Forest
  - XGBoost

Seleciona o melhor modelo por AUC-ROC, monta um Pipeline completo
(ColumnTransformer + Classificador) e salva em disco.
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Garante que o diretório raiz do projeto esteja no sys.path para imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import criar_pipeline_preprocessamento, preparar_dados  # noqa: E402

# Suprime avisos desnecessários durante o treinamento
warnings.filterwarnings("ignore", category=FutureWarning)


# ===================================================================
# Funções auxiliares
# ===================================================================

def _calcular_scale_pos_weight(y: pd.Series) -> float:
    """
    Calcula o scale_pos_weight para o XGBoost com base no
    desbalanceamento entre classes negativa e positiva.

    Fórmula: count(negativos) / count(positivos)
    """
    contagem = y.value_counts()
    negativos = contagem.get(0, 0)
    positivos = contagem.get(1, 1)  # evita divisão por zero
    return negativos / positivos


def _imprimir_tabela_resultados(resultados: dict) -> None:
    """
    Imprime uma tabela formatada comparando os modelos treinados.
    """
    print("\n" + "=" * 70)
    print("COMPARAÇÃO DE MODELOS — EduFlow Intelligence")
    print("=" * 70)
    print(f"{'Modelo':<30} {'AUC-ROC':>10} {'Avg Precision':>15}")
    print("-" * 70)

    for nome, info in resultados.items():
        auc = info["auc_roc"]
        ap = info["avg_precision"]
        marcador = " *" if info.get("melhor") else ""
        print(f"{nome:<30} {auc:>10.4f} {ap:>15.4f}{marcador}")

    print("-" * 70)
    print("* = Melhor modelo selecionado\n")


# ===================================================================
# Função principal de treinamento
# ===================================================================

def treinar_modelos(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor,
) -> dict:
    """
    Treina e compara os 3 modelos de classificação.

    Parâmetros
    ----------
    X_train : pd.DataFrame
        Features de treino (dados brutos após feature engineering).
    y_train : pd.Series
        Target de treino.
    X_test : pd.DataFrame
        Features de teste.
    y_test : pd.Series
        Target de teste.
    preprocessor : ColumnTransformer
        Transformador de colunas (não-fitado) para pré-processamento.

    Retorna
    -------
    dict
        Dicionário com nome do modelo como chave e métricas/objetos como valor.
    """
    # Calcula peso para classes desbalanceadas (XGBoost)
    scale_pos_weight = _calcular_scale_pos_weight(y_train)
    print(f"[INFO] scale_pos_weight calculado: {scale_pos_weight:.2f}")

    # Definição dos modelos candidatos
    modelos = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        ),
    }

    resultados = {}

    for nome, modelo in modelos.items():
        print(f"\n[TREINANDO] {nome}...")

        # Cria pipeline temporário para cada modelo (preprocessor + classificador)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", modelo),
        ])

        # Treina o pipeline completo
        pipe.fit(X_train, y_train)

        # Predições de probabilidade para a classe positiva
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        # Métricas de avaliação
        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Curva precision-recall
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)

        resultados[nome] = {
            "modelo": modelo,
            "pipeline": pipe,
            "auc_roc": auc,
            "avg_precision": ap,
            "classification_report": report,
            "precision_curve": precision_vals,
            "recall_curve": recall_vals,
            "y_proba": y_proba,
            "melhor": False,
        }

        print(f"  AUC-ROC: {auc:.4f} | Avg Precision: {ap:.4f}")
        print(classification_report(y_test, y_pred))

    # Seleciona o melhor modelo pelo AUC-ROC
    melhor_nome = max(resultados, key=lambda k: resultados[k]["auc_roc"])
    resultados[melhor_nome]["melhor"] = True
    print(f"[RESULTADO] Melhor modelo: {melhor_nome} "
          f"(AUC-ROC = {resultados[melhor_nome]['auc_roc']:.4f})")

    # Imprime tabela comparativa
    _imprimir_tabela_resultados(resultados)

    return resultados


# ===================================================================
# Persistência do modelo
# ===================================================================

def salvar_modelo(
    pipeline: Pipeline,
    metadata: dict,
    model_dir: str = "models",
) -> None:
    """
    Salva o pipeline completo (preprocessor + classificador) e metadados.

    Parâmetros
    ----------
    pipeline : Pipeline
        Pipeline sklearn completo já treinado.
    metadata : dict
        Informações sobre o treinamento (modelo, AUC-ROC, features, etc.).
    model_dir : str
        Diretório de destino para os artefatos.
    """
    # Garante que o diretório exista
    model_path = os.path.join(PROJECT_ROOT, model_dir)
    os.makedirs(model_path, exist_ok=True)

    # Salva o pipeline via joblib
    pipeline_file = os.path.join(model_path, "model.pkl")
    joblib.dump(pipeline, pipeline_file)
    print(f"[SALVO] Pipeline completo -> {pipeline_file}")

    # Salva os metadados em JSON
    metadata_file = os.path.join(model_path, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f"[SALVO] Metadados         -> {metadata_file}")


# ===================================================================
# Execução principal
# ===================================================================

def main() -> None:
    """
    Pipeline completo de treinamento:
      1. Carrega dados brutos
      2. Aplica feature engineering + prepara dados
      3. Cria ColumnTransformer (preprocessor)
      4. Divide em treino/teste
      5. Treina e compara modelos
      6. Salva o melhor pipeline + metadados
    """
    print("=" * 70)
    print("EduFlow Intelligence — Treinamento de Modelos")
    print(f"Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Carrega dados brutos
    # ------------------------------------------------------------------
    csv_path = os.path.join(PROJECT_ROOT, "data", "leads_raw.csv")
    print(f"\n[1/6] Carregando dados de {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  -> {df.shape[0]} registros | {df.shape[1]} colunas")

    # ------------------------------------------------------------------
    # 2. Aplica feature engineering e prepara X, y
    # ------------------------------------------------------------------
    print("\n[2/6] Aplicando feature engineering e preparando dados...")
    X, y, feature_names = preparar_dados(df)
    print(f"  -> {X.shape[1]} features | Target: 'converteu'")
    print(f"  -> Distribuição do target: {dict(y.value_counts())}")

    # ------------------------------------------------------------------
    # 3. Cria preprocessor (ColumnTransformer não-fitado)
    # ------------------------------------------------------------------
    print("\n[3/6] Criando preprocessor (ColumnTransformer)...")
    preprocessor = criar_pipeline_preprocessamento()
    print(f"  -> {len(feature_names)} features configuradas no pipeline")

    # ------------------------------------------------------------------
    # 4. Divide em treino e teste
    # ------------------------------------------------------------------
    print("\n[4/6] Dividindo dados (80% treino / 20% teste)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"  -> Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 5. Treina e compara modelos
    # ------------------------------------------------------------------
    print("\n[5/6] Treinando e comparando modelos...")
    resultados = treinar_modelos(X_train, y_train, X_test, y_test, preprocessor)

    # ------------------------------------------------------------------
    # 6. Salva o melhor modelo
    # ------------------------------------------------------------------
    print("\n[6/6] Salvando o melhor modelo...")

    # Identifica o melhor
    melhor_nome = [n for n, r in resultados.items() if r["melhor"]][0]
    melhor_info = resultados[melhor_nome]

    # O pipeline já está montado dentro de treinar_modelos
    melhor_pipeline = melhor_info["pipeline"]

    # Obtém os nomes das features transformadas (pós one-hot encoding)
    try:
        preprocessor_fitado = melhor_pipeline.named_steps['preprocessor']
        transformed_feature_names = preprocessor_fitado.get_feature_names_out().tolist()
    except Exception:
        transformed_feature_names = []

    # Monta metadados para persistência
    metadata = {
        "projeto": "EduFlow Intelligence",
        "melhor_modelo": melhor_nome,
        "auc_roc": float(melhor_info["auc_roc"]),
        "avg_precision": float(melhor_info["avg_precision"]),
        "feature_names": feature_names,
        "transformed_feature_names": transformed_feature_names,
        "n_features": len(feature_names),
        "n_treino": int(X_train.shape[0]),
        "n_teste": int(X_test.shape[0]),
        "data_treinamento": datetime.now().isoformat(),
        "comparacao_modelos": {
            nome: {
                "auc_roc": float(info["auc_roc"]),
                "avg_precision": float(info["avg_precision"]),
            }
            for nome, info in resultados.items()
        },
    }

    salvar_modelo(melhor_pipeline, metadata)

    print("\n" + "=" * 70)
    print("Treinamento concluído com sucesso!")
    print(f"Melhor modelo: {melhor_nome} (AUC-ROC = {melhor_info['auc_roc']:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
