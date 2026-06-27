"""
Módulo de pré-processamento para o EduFlow Intelligence.

Constrói o pipeline de transformação de dados usando scikit-learn,
incluindo imputação, encoding categórico e normalização. Integra
a etapa de engenharia de features antes da preparação final.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import criar_features


# --- Definição das colunas por tipo ---

# Colunas numéricas contínuas: passam por imputação + normalização
COLUNAS_NUMERICAS: list[str] = [
    "dias_sem_contato",
    "n_tentativas",
    "renda_estimada",
    "score_engajamento",
    "urgencia_estimada",
]

# Colunas categóricas: passam por imputação + one-hot encoding
COLUNAS_CATEGORICAS: list[str] = [
    "canal_captacao",
    "curso_interesse",
    "faixa_etaria",
    "turno_preferencia",
]

# Colunas binárias: já estão prontas, passam direto (passthrough)
COLUNAS_BINARIAS: list[str] = [
    "respondeu_contato",
    "is_indicacao",
    "is_evento",
    "contato_rapido",
    "tentativas_ideal",
    "turno_noite",
    "curso_adulto",
]


def get_feature_names() -> tuple[list[str], list[str]]:
    """Retorna (colunas_numericas, colunas_categoricas) usadas no modelo.

    Útil para inspeção e validação do schema de dados esperado
    pelo pipeline de pré-processamento.

    Returns:
        Tupla com duas listas: nomes das colunas numéricas e
        nomes das colunas categóricas.
    """
    return COLUNAS_NUMERICAS, COLUNAS_CATEGORICAS


def criar_pipeline_preprocessamento() -> ColumnTransformer:
    """Cria o ColumnTransformer com imputação, encoding e normalização.

    Estrutura do pipeline:
        - Numéricas: SimpleImputer(mediana)  StandardScaler
        - Categóricas: SimpleImputer(mais frequente)  OneHotEncoder
        - Binárias: passthrough (sem transformação)

    Returns:
        ColumnTransformer configurado e pronto para fit/transform.
    """
    # Pipeline para variáveis numéricas contínuas
    pipeline_numerico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("normalizador", StandardScaler()),
        ]
    )

    # Pipeline para variáveis categóricas
    pipeline_categorico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # Montagem do transformador completo por tipo de coluna
    preprocessador = ColumnTransformer(
        transformers=[
            ("numericas", pipeline_numerico, COLUNAS_NUMERICAS),
            ("categoricas", pipeline_categorico, COLUNAS_CATEGORICAS),
            ("binarias", "passthrough", COLUNAS_BINARIAS),
        ]
    )

    return preprocessador


def preparar_dados(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Aplica feature engineering e retorna X, y e nomes das features.

    Executa o fluxo completo de preparação:
        1. Chama criar_features() para gerar variáveis derivadas
        2. Separa as variáveis preditoras (X) da variável alvo (y)
        3. Filtra apenas as colunas esperadas pelo pipeline

    Args:
        df: DataFrame bruto contendo todas as colunas dos leads,
            incluindo a coluna alvo 'converteu'.

    Returns:
        Tupla com:
            - X: DataFrame com as colunas de entrada do modelo
            - y: Series com a variável alvo (converteu)
            - feature_names: lista com os nomes de todas as features
    """
    # Aplica engenharia de features para criar variáveis derivadas
    df_transformado = criar_features(df)

    # Separa a variável alvo
    y = df_transformado["converteu"]

    # Seleciona apenas as colunas usadas pelo pipeline
    todas_colunas = COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS + COLUNAS_BINARIAS
    X = df_transformado[todas_colunas]

    return X, y, todas_colunas
