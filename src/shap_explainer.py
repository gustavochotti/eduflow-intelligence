"""
EduFlow Intelligence — Módulo de Explicabilidade SHAP
=====================================================
Gera explicações interpretáveis para as predições do modelo
de lead scoring utilizando SHAP (SHapley Additive exPlanations).

O explainer opera sobre o classificador extraído do pipeline
(não sobre o pipeline completo). Os dados de entrada (X_background,
X_lead) devem já estar pré-transformados pelo preprocessor.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Garante que o diretório raiz do projeto esteja no sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suprime avisos verbosos do SHAP durante a execução
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


# ===================================================================
# Detecção do tipo de modelo
# ===================================================================

def _eh_modelo_tree(modelo) -> bool:
    """
    Verifica se o modelo é baseado em árvores de decisão.
    Suporta scikit-learn (RandomForest, GradientBoosting, etc.)
    e XGBoost.
    """
    tipos_tree = (
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
    )
    return type(modelo).__name__ in tipos_tree


def _eh_modelo_linear(modelo) -> bool:
    """
    Verifica se o modelo é linear (Regressão Logística, SVM linear, etc.).
    """
    tipos_linear = (
        "LogisticRegression",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SGDClassifier",
        "SGDRegressor",
        "LinearSVC",
        "LinearSVR",
    )
    return type(modelo).__name__ in tipos_linear


# ===================================================================
# Criação do Explainer
# ===================================================================

def criar_explainer(modelo, X_background: np.ndarray):
    """
    Cria o explainer SHAP apropriado com base no tipo do modelo.

    Estratégia de seleção:
      - Modelos baseados em árvore  shap.TreeExplainer
      - Modelos lineares            shap.LinearExplainer
      - Outros modelos              shap.KernelExplainer (fallback)

    Parâmetros
    ----------
    modelo : estimator
        Classificador extraído do pipeline via
        ``pipeline.named_steps['classifier']``.
    X_background : np.ndarray
        Amostra de dados pré-transformados (saída do preprocessor)
        usada como background/referência para o SHAP.
        Para TreeExplainer, é opcional mas recomendado.
        Para KernelExplainer, deve ser um subset representativo
        (ex: 50-100 amostras) para manter a performance.

    Retorna
    -------
    shap.Explainer
        Instância do explainer SHAP pronta para uso.
    """
    nome_modelo = type(modelo).__name__

    if _eh_modelo_tree(modelo):
        print(f"[SHAP] Usando TreeExplainer para {nome_modelo}")
        explainer = shap.TreeExplainer(modelo, data=X_background)

    elif _eh_modelo_linear(modelo):
        print(f"[SHAP] Usando LinearExplainer para {nome_modelo}")
        # LinearExplainer precisa dos dados de background como referência
        explainer = shap.LinearExplainer(modelo, masker=X_background)

    else:
        # Fallback: KernelExplainer (mais lento, mas universal)
        print(f"[SHAP] Usando KernelExplainer (fallback) para {nome_modelo}")
        print(
            "[SHAP]  KernelExplainer pode ser lento. "
            "Considere reduzir X_background para ~50-100 amostras."
        )

        # Função de predição que retorna probabilidades
        def predict_fn(X):
            return modelo.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, X_background)

    return explainer


# ===================================================================
# Explicação de um lead individual
# ===================================================================

def explicar_lead(
    explainer,
    X_lead: np.ndarray,
    feature_names: list,
) -> dict:
    """
    Retorna as top-3 features SHAP mais influentes para um lead específico.

    Parâmetros
    ----------
    explainer : shap.Explainer
        Explainer SHAP já criado via ``criar_explainer()``.
    X_lead : np.ndarray
        Dados pré-transformados de um único lead.
        Shape esperado: (1, n_features).
    feature_names : list
        Nomes das features na ordem correspondente às colunas de X_lead.

    Retorna
    -------
    dict
        {
            'shap_top_features': [
                {'feature': nome, 'impact': valor_shap},
                ...  # top 3
            ],
            'shap_values': list  # todos os SHAP values do lead
        }
    """
    # Garante formato 2D
    if X_lead.ndim == 1:
        X_lead = X_lead.reshape(1, -1)

    # Calcula SHAP values para o lead
    shap_values = explainer.shap_values(X_lead)

    # Trata diferentes formatos de retorno do SHAP
    # Para classificação binária, pode retornar lista [classe_0, classe_1]
    if isinstance(shap_values, list):
        # Usa SHAP values da classe positiva (índice 1)
        valores = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    elif shap_values.ndim == 3:
        # Shape (n_samples, n_features, n_classes) — pega classe positiva
        valores = shap_values[0, :, 1]
    else:
        # Shape (n_samples, n_features) — caso padrão
        valores = shap_values[0]

    # Garante que temos um array 1D
    valores = np.array(valores).flatten()

    # Mapeia feature  impacto e ordena por valor absoluto (maior impacto primeiro)
    n_features = min(len(valores), len(feature_names))
    importancias = [
        {"feature": feature_names[i], "impact": float(valores[i])}
        for i in range(n_features)
    ]
    importancias.sort(key=lambda x: abs(x["impact"]), reverse=True)

    # Top-3 features mais influentes
    top_3 = importancias[:3]

    # Log informativo
    print("\n[SHAP] Top-3 features para este lead:")
    for i, feat in enumerate(top_3, 1):
        direcao = " favorece conversão" if feat["impact"] > 0 else " desfavorece"
        print(f"  {i}. {feat['feature']}: {feat['impact']:+.4f} ({direcao})")

    return {
        "shap_top_features": top_3,
        "shap_values": valores.tolist(),
    }


# ===================================================================
# Explicação de todo o dataset
# ===================================================================

def explicar_dataset(
    explainer,
    X: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Calcula SHAP values para todo o dataset e retorna como DataFrame.

    Parâmetros
    ----------
    explainer : shap.Explainer
        Explainer SHAP já criado via ``criar_explainer()``.
    X : np.ndarray
        Dados pré-transformados de múltiplos leads.
        Shape: (n_samples, n_features).
    feature_names : list
        Nomes das features na ordem correspondente às colunas de X.

    Retorna
    -------
    pd.DataFrame
        DataFrame onde cada coluna é uma feature e cada linha contém
        o SHAP value correspondente. Inclui colunas extras:
        - 'shap_sum': soma dos SHAP values (contribuição total)
        - 'top_feature': feature com maior impacto absoluto
    """
    print(f"[SHAP] Calculando SHAP values para {X.shape[0]} amostras...")

    # Calcula SHAP values para todo o dataset
    shap_values = explainer.shap_values(X)

    # Trata diferentes formatos de retorno
    if isinstance(shap_values, list):
        # Classificação binária — usa classe positiva
        valores = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif shap_values.ndim == 3:
        # Shape (n_samples, n_features, n_classes) — pega classe positiva
        valores = shap_values[:, :, 1]
    else:
        valores = shap_values

    valores = np.array(valores)

    # Cria DataFrame com os SHAP values
    n_cols = min(valores.shape[1], len(feature_names))
    colunas_shap = feature_names[:n_cols]

    df_shap = pd.DataFrame(
        valores[:, :n_cols],
        columns=[f"shap_{col}" for col in colunas_shap],
    )

    # Colunas extras para análise rápida
    df_shap["shap_sum"] = df_shap.sum(axis=1)
    df_shap["top_feature"] = df_shap.iloc[:, :n_cols].apply(
        lambda row: colunas_shap[np.argmax(np.abs(row.values))],
        axis=1,
    )

    # Resumo estatístico
    print(f"[SHAP] Cálculo concluído. Shape do resultado: {df_shap.shape}")
    print("\n[SHAP] Importância média global das features (top-10):")

    importancia_media = (
        df_shap.iloc[:, :n_cols]
        .abs()
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    for nome_feat, valor in importancia_media.items():
        # Remove o prefixo 'shap_' para exibição
        nome_limpo = nome_feat.replace("shap_", "")
        print(f"  {nome_limpo:<35} {valor:.4f}")

    return df_shap
