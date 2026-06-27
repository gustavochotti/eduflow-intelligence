"""
Módulo de engenharia de features para o EduFlow Intelligence.

Cria variáveis derivadas a partir dos dados brutos de leads,
enriquecendo o dataset para melhorar a performance do modelo
de lead scoring.
"""

import pandas as pd


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas a partir dos dados brutos de leads.

    Recebe o DataFrame com os dados brutos e retorna uma cópia
    enriquecida com novas colunas calculadas. Trata valores nulos
    antes de cada cálculo para evitar propagação de NaN.

    Args:
        df: DataFrame com as colunas brutas dos leads
            (respondeu_contato, dias_sem_contato, n_tentativas,
             canal_captacao, turno_preferencia, curso_interesse).

    Returns:
        DataFrame com as features originais mais as derivadas.
    """
    # Trabalha sobre uma cópia para não alterar o DataFrame original
    df = df.copy()

    # --- Preenche valores nulos nas colunas numéricas base ---
    dias_sem_contato = df["dias_sem_contato"].fillna(0)
    n_tentativas = df["n_tentativas"].fillna(0)
    respondeu_contato = df["respondeu_contato"].fillna(0)

    # --- Score de engajamento (0 a 1) ---
    # Combina três sinais:
    #   - 30% peso: se respondeu ao contato (binário)
    #   - 40% peso: inverso dos dias sem contato (normalizado em 60 dias)
    #   - 30% peso: número de tentativas (normalizado em 5)
    componente_resposta = 0.3 * respondeu_contato
    componente_recencia = 0.4 * (1 - dias_sem_contato / 60).clip(0, 1)
    componente_tentativas = 0.3 * (n_tentativas.clip(0, 5) / 5)
    df["score_engajamento"] = componente_resposta + componente_recencia + componente_tentativas

    # --- Flag de indicação ---
    # Leads captados por indicação tendem a converter mais
    df["is_indicacao"] = (df["canal_captacao"] == "Indicação").astype(int)

    # --- Flag de evento presencial ---
    # Leads de eventos presenciais apresentam alto engajamento inicial
    df["is_evento"] = (df["canal_captacao"] == "Evento Presencial").astype(int)

    # --- Urgência estimada ---
    # Quanto maior o valor, mais tempo sem contato com mais tentativas acumuladas
    # Sinaliza leads em risco de perda
    df["urgencia_estimada"] = dias_sem_contato * (n_tentativas + 1)

    # --- Contato rápido ---
    # Lead foi contatado em menos de 3 dias (janela ideal de resposta)
    df["contato_rapido"] = (dias_sem_contato < 3).astype(int)

    # --- Tentativas no intervalo ideal ---
    # Entre 1 e 3 tentativas é considerado o ponto ótimo de follow-up
    df["tentativas_ideal"] = ((n_tentativas >= 1) & (n_tentativas <= 3)).astype(int)

    # --- Preferência pelo turno noturno ---
    # Alunos noturnos podem ter perfil diferenciado (trabalhadores, etc.)
    df["turno_noite"] = (df["turno_preferencia"] == "Noite").astype(int)

    # --- Interesse em curso adulto ---
    # Segmentação específica para o produto principal
    df["curso_adulto"] = (df["curso_interesse"] == "Inglês Adulto").astype(int)

    return df
