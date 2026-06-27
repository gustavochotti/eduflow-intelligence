"""
Geração de dados sintéticos de leads para CRM de franquia educacional (escola de idiomas).

Este script cria um dataset realista com ~1.800 registros simulando leads captados
por diferentes canais de marketing, com variáveis demográficas, comportamentais e
uma variável-alvo de conversão baseada em lógica logística com ruído controlado.

Dependências: numpy, pandas, faker, scipy
"""

import os

import numpy as np
import pandas as pd
from faker import Faker
from scipy.stats import poisson


def gerar_dados_leads(n: int = 1800, seed: int = 42) -> pd.DataFrame:
    """
    Gera um DataFrame com dados sintéticos de leads para uma escola de idiomas.

    Parâmetros
    ----------
    n : int
        Número de registros a gerar (padrão: 1800).
    seed : int
        Semente para reprodutibilidade (padrão: 42).

    Retorna
    -------
    pd.DataFrame
        DataFrame com todas as variáveis de lead, incluindo a variável-alvo `converteu`.
    """
    # ── Fixar sementes para reprodutibilidade ──────────────────────────────
    rng = np.random.default_rng(seed)
    fake = Faker("pt_BR")
    Faker.seed(seed)

    # ── 1. Identificação ──────────────────────────────────────────────────
    lead_id = np.arange(1, n + 1)
    nome_ficticio = [fake.name() for _ in range(n)]
    municipio = [fake.city() for _ in range(n)]

    # ── 2. Canal de captação (proporções realistas) ───────────────────────
    canais = [
        "Indicação",
        "Instagram",
        "Google Ads",
        "Facebook Ads",
        "Tráfego Orgânico",
        "Evento Presencial",
    ]
    prob_canais = [0.15, 0.25, 0.20, 0.15, 0.15, 0.10]
    canal_captacao = rng.choice(canais, size=n, p=prob_canais)

    # ── 3. Curso de interesse ─────────────────────────────────────────────
    cursos = [
        "Inglês Básico",
        "Inglês Intermediário",
        "Inglês Adulto",
        "Inglês Teens",
        "Espanhol",
        "Francês",
    ]
    prob_cursos = [0.30, 0.20, 0.20, 0.15, 0.10, 0.05]
    curso_interesse = rng.choice(cursos, size=n, p=prob_cursos)

    # ── 4. Faixa etária (distribuição normal-ish centrada em 25-34) ───────
    faixas = ["18-24", "25-34", "35-44", "45-54", "55+"]
    # Pesos aproximando uma normal centrada no índice 1 (25-34)
    prob_faixas = [0.20, 0.35, 0.25, 0.13, 0.07]
    faixa_etaria = rng.choice(faixas, size=n, p=prob_faixas)

    # ── 5. Turno de preferência ───────────────────────────────────────────
    turnos = ["Manhã", "Tarde", "Noite"]
    prob_turnos = [0.25, 0.30, 0.45]
    turno_preferencia = rng.choice(turnos, size=n, p=prob_turnos)

    # ── 6. Dia da semana do primeiro contato ──────────────────────────────
    dias_semana = [
        "Segunda-feira",
        "Terça-feira",
        "Quarta-feira",
        "Quinta-feira",
        "Sexta-feira",
        "Sábado",
        "Domingo",
    ]
    # Leads chegam mais em dias úteis; fim de semana tem menos volume
    prob_dias = [0.17, 0.17, 0.17, 0.17, 0.15, 0.10, 0.07]
    dia_semana_contato = rng.choice(dias_semana, size=n, p=prob_dias)

    # ── 7. Dias sem contato (Poisson λ=7, clipado 0-60) ──────────────────
    dias_sem_contato = poisson.rvs(mu=7, size=n, random_state=seed)
    dias_sem_contato = np.clip(dias_sem_contato, 0, 60)

    # ── 8. Número de tentativas de contato (Poisson λ=2, clipado 0-8) ────
    n_tentativas = poisson.rvs(mu=2, size=n, random_state=seed + 1)
    n_tentativas = np.clip(n_tentativas, 0, 8)

    # ── 9. Respondeu ao contato (binário) ─────────────────────────────────
    # Probabilidade de responder diminui com mais dias sem contato
    prob_resposta_base = 0.45
    ajuste_dias = -0.008 * dias_sem_contato  # quanto mais dias, menor a chance
    ajuste_tentativas = np.where(n_tentativas >= 1, 0.05, -0.10)  # ao menos 1 tentativa ajuda
    prob_resposta = np.clip(prob_resposta_base + ajuste_dias + ajuste_tentativas, 0.05, 0.90)
    respondeu_contato = rng.binomial(1, prob_resposta)

    # ── 10. Renda estimada (log-normal centrada em ~R$ 3.500) ─────────────
    # ln(3500) ≈ 8.16; usamos mu=8.16 e sigma=0.55 para dispersão realista
    mu_renda = np.log(3500)
    sigma_renda = 0.55
    renda_estimada = rng.lognormal(mean=mu_renda, sigma=sigma_renda, size=n)
    renda_estimada = np.round(renda_estimada, 2)

    # ── 11. Conversão (variável-alvo com lógica logística) ────────────────
    converteu = _calcular_conversao(
        rng=rng,
        canal_captacao=canal_captacao,
        respondeu_contato=respondeu_contato,
        dias_sem_contato=dias_sem_contato,
        n_tentativas=n_tentativas,
        curso_interesse=curso_interesse,
        n=n,
    )

    # ── Montar o DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "lead_id": lead_id,
            "nome_ficticio": nome_ficticio,
            "municipio": municipio,
            "canal_captacao": canal_captacao,
            "curso_interesse": curso_interesse,
            "faixa_etaria": faixa_etaria,
            "turno_preferencia": turno_preferencia,
            "dia_semana_contato": dia_semana_contato,
            "dias_sem_contato": dias_sem_contato,
            "n_tentativas": n_tentativas,
            "respondeu_contato": respondeu_contato,
            "renda_estimada": renda_estimada,
            "converteu": converteu,
        }
    )

    # ── 12. Inserir valores ausentes (NaN) ────────────────────────────────
    df = _inserir_valores_ausentes(df, rng)

    return df


def _calcular_conversao(
    rng: np.random.Generator,
    canal_captacao: np.ndarray,
    respondeu_contato: np.ndarray,
    dias_sem_contato: np.ndarray,
    n_tentativas: np.ndarray,
    curso_interesse: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Calcula a variável-alvo `converteu` usando regressão logística com ruído.

    A lógica combina múltiplos fatores para produzir uma taxa de conversão
    global entre 5-8%, mantendo correlações realistas com as features.

    Parâmetros
    ----------
    rng : np.random.Generator
        Gerador de números aleatórios.
    canal_captacao, respondeu_contato, dias_sem_contato,
    n_tentativas, curso_interesse : np.ndarray
        Arrays com os valores das respectivas variáveis.
    n : int
        Número de registros.

    Retorna
    -------
    np.ndarray
        Array binário (0/1) indicando conversão.
    """
    # ── Intercepto (log-odds base) ────────────────────────────────────────
    # Ajustado para que a taxa global fique entre 5-8%
    logit = np.full(n, -3.2)

    # ── Efeito do canal de captação ───────────────────────────────────────
    # Taxas-alvo aproximadas por canal (em log-odds relativos):
    #   Indicação ~15%   -> forte efeito positivo
    #   Evento ~12%      -> efeito positivo
    #   Orgânico ~7%     -> efeito moderado
    #   Google Ads ~5%   -> efeito neutro (próximo da base)
    #   Instagram ~3%    -> efeito negativo
    #   Facebook Ads ~2% -> efeito negativo forte
    efeito_canal = {
        "Indicação": 1.5,
        "Evento Presencial": 1.2,
        "Tráfego Orgânico": 0.5,
        "Google Ads": 0.0,
        "Instagram": -0.5,
        "Facebook Ads": -0.8,
    }
    for canal, coef in efeito_canal.items():
        logit[canal_captacao == canal] += coef

    # ── Efeito de respondeu ao contato (dobra a probabilidade ≈ +0.7 logit)
    logit += respondeu_contato * 0.7

    # ── Efeito dos dias sem contato (quanto menos, melhor) ────────────────
    # Normalizado: cada dia a mais reduz o logit
    logit -= 0.04 * dias_sem_contato

    # ── Efeito do número de tentativas (ótimo entre 1-3) ──────────────────
    # Modela como parábola invertida com pico em ~2 tentativas
    efeito_tentativas = -0.15 * (n_tentativas - 2) ** 2 + 0.3
    logit += efeito_tentativas

    # ── Efeito do curso de interesse ──────────────────────────────────────
    efeito_curso = {
        "Inglês Adulto": 0.3,
        "Inglês Intermediário": 0.1,
        "Inglês Básico": 0.0,
        "Inglês Teens": -0.1,
        "Espanhol": -0.1,
        "Francês": -0.2,
    }
    for curso, coef in efeito_curso.items():
        logit[curso_interesse == curso] += coef

    # ── Ruído logístico para evitar relação perfeitamente determinística ──
    ruido = rng.logistic(loc=0, scale=1.0, size=n)
    logit += ruido

    # ── Converter log-odds em probabilidade via função sigmoide ───────────
    prob_conversao = 1.0 / (1.0 + np.exp(-logit))

    # ── Gerar decisão binária de conversão ────────────────────────────────
    converteu = rng.binomial(1, prob_conversao)

    return converteu


def _inserir_valores_ausentes(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Insere valores ausentes (NaN) em colunas específicas para simular
    dados reais de CRM com preenchimento incompleto.

    Proporções de NaN:
        - renda_estimada: ~12%
        - faixa_etaria: ~10%
        - turno_preferencia: ~8%

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame original sem valores ausentes.
    rng : np.random.Generator
        Gerador de números aleatórios.

    Retorna
    -------
    pd.DataFrame
        DataFrame com NaN inseridos nas colunas especificadas.
    """
    n = len(df)

    # ~12% de NaN em renda_estimada
    mask_renda = rng.random(n) < 0.12
    df.loc[mask_renda, "renda_estimada"] = np.nan

    # ~10% de NaN em faixa_etaria
    mask_faixa = rng.random(n) < 0.10
    df.loc[mask_faixa, "faixa_etaria"] = np.nan

    # ~8% de NaN em turno_preferencia
    mask_turno = rng.random(n) < 0.08
    df.loc[mask_turno, "turno_preferencia"] = np.nan

    return df


def imprimir_resumo(df: pd.DataFrame) -> None:
    """
    Imprime estatísticas descritivas do dataset gerado para validação rápida.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados de leads gerados.
    """
    print("=" * 70)
    print("  RESUMO DO DATASET DE LEADS GERADO")
    print("=" * 70)

    print(f"\n[INFO] Total de registros: {len(df)}")
    print(f"[INFO] Total de colunas:   {len(df.columns)}")

    # ── Taxa de conversão geral ───────────────────────────────────────────
    taxa_conversao = df["converteu"].mean() * 100
    total_convertidos = df["converteu"].sum()
    print(f"\n[CONVERSAO] Taxa de conversao geral: {taxa_conversao:.1f}% ({total_convertidos} leads)")

    # ── Taxa de conversão por canal ───────────────────────────────────────
    print("\n[CANAL] Taxa de conversao por canal de captacao:")
    conv_canal = (
        df.groupby("canal_captacao")["converteu"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    conv_canal["mean"] = (conv_canal["mean"] * 100).round(1)
    conv_canal.columns = ["Taxa (%)", "Total"]
    for canal, row in conv_canal.iterrows():
        print(f"   {canal:<22s} {row['Taxa (%)']:>5.1f}%  (n={int(row['Total'])})")

    # ── Valores ausentes ──────────────────────────────────────────────────
    print("\n[MISSING] Valores ausentes:")
    nulos = df.isnull().sum()
    nulos_pct = (df.isnull().mean() * 100).round(1)
    for col in df.columns:
        if nulos[col] > 0:
            print(f"   {col:<22s} {nulos[col]:>4d} ({nulos_pct[col]:.1f}%)")

    # ── Distribuição de variáveis numéricas ───────────────────────────────
    print("\n[STATS] Estatisticas de variaveis numericas:")
    numericas = ["dias_sem_contato", "n_tentativas", "renda_estimada"]
    for col in numericas:
        serie = df[col].dropna()
        print(
            f"   {col:<22s} média={serie.mean():>8.1f}  "
            f"mediana={serie.median():>8.1f}  "
            f"min={serie.min():>6.0f}  max={serie.max():>8.0f}"
        )

    # ── Distribuição de respondeu_contato ─────────────────────────────────
    pct_respondeu = df["respondeu_contato"].mean() * 100
    print(f"\n[RESPOSTA] Taxa de resposta ao contato: {pct_respondeu:.1f}%")

    print("\n" + "=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# Execução principal
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[START] Iniciando geracao de dados sinteticos de leads...\n")

    # Gerar o dataset
    df_leads = gerar_dados_leads(n=1800, seed=42)

    # Garantir que o diretorio de saida existe
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    caminho_saida = os.path.join(project_root, "data", "leads_raw.csv")
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    # Salvar em CSV com encoding compativel com Excel em portugues
    df_leads.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    print(f"[SALVO] Dataset salvo em: {caminho_saida}\n")

    # Exibir resumo
    imprimir_resumo(df_leads)
