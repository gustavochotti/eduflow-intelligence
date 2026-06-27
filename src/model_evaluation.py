"""
EduFlow Intelligence — Módulo de Avaliação de Modelos
=====================================================
Avaliação completa do modelo de lead scoring com:
  - Métricas de classificação (AUC-ROC, Precision-Recall, F1)
  - Simulação de impacto de negócio por threshold
  - Geração de gráficos diagnósticos
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para salvar figuras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Garante que o diretório raiz do projeto esteja no sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ===================================================================
# Avaliação completa do modelo
# ===================================================================

def avaliar_modelo(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Avaliação completa do pipeline treinado.

    Parâmetros
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline completo (preprocessor + classificador) já treinado.
    X_test : pd.DataFrame
        Features de teste (dados brutos após feature engineering).
    y_test : pd.Series
        Target de teste.

    Retorna
    -------
    dict
        Dicionário com todas as métricas e dados para gráficos:
        - auc_roc, avg_precision
        - classification_report (dict)
        - confusion_matrix
        - curvas ROC e Precision-Recall
        - scores (probabilidades preditas)
    """
    # Predições
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # Métricas escalares
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    # Relatório de classificação
    report = classification_report(y_test, y_pred, output_dict=True)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    # Curva ROC
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

    # Curva Precision-Recall
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
        y_test, y_proba
    )

    # Impressão resumida
    print("\n" + "=" * 60)
    print("AVALIAÇÃO DO MODELO — EduFlow Intelligence")
    print("=" * 60)
    print(f"AUC-ROC:           {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"\nMatriz de Confusão:")
    print(f"  TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
    print(f"  FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "auc_roc": auc,
        "avg_precision": ap,
        "classification_report": report,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision_curve": precision_vals,
        "recall_curve": recall_vals,
        "pr_thresholds": pr_thresholds,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


# ===================================================================
# Simulação de impacto de negócio
# ===================================================================

def simular_impacto_negocio(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_leads_mensal: int = 1200,
) -> pd.DataFrame:
    """
    Simula o impacto de diferentes thresholds de score na operação comercial.

    Para cada threshold (0.1 a 0.9, passo 0.1), calcula:
      - leads_contatados:    quantidade de leads com score >= threshold
      - matriculas_esperadas: quantos desses leads realmente convertem
      - taxa_conversao_grupo: taxa de conversão dentro do grupo selecionado
      - esforco_relativo:     proporção de leads contatados vs total

    Parâmetros
    ----------
    y_true : array-like
        Valores reais do target (0/1).
    y_scores : array-like
        Probabilidades preditas (scores do modelo).
    n_leads_mensal : int
        Volume mensal estimado de leads para projeção.

    Retorna
    -------
    pd.DataFrame
        Tabela com as métricas de impacto por threshold.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    total_leads = len(y_true)

    registros = []

    for threshold in np.arange(0.1, 1.0, 0.1):
        # Máscara: leads com score >= threshold
        mascara = y_scores >= threshold

        # Quantidade de leads que seriam contatados
        leads_contatados = int(mascara.sum())

        # Quantos desses realmente convertem (são positivos)
        matriculas_esperadas = int(y_true[mascara].sum()) if leads_contatados > 0 else 0

        # Taxa de conversão no grupo filtrado
        taxa_conversao_grupo = (
            matriculas_esperadas / leads_contatados if leads_contatados > 0 else 0.0
        )

        # Esforço relativo (proporção do total)
        esforco_relativo = leads_contatados / total_leads if total_leads > 0 else 0.0

        # Projeção mensal escalada
        fator_escala = n_leads_mensal / total_leads if total_leads > 0 else 0.0
        leads_mensal_projetado = int(leads_contatados * fator_escala)
        matriculas_mensal_projetado = int(matriculas_esperadas * fator_escala)

        registros.append({
            "threshold": round(threshold, 1),
            "leads_contatados": leads_contatados,
            "matriculas_esperadas": matriculas_esperadas,
            "taxa_conversao_grupo": round(taxa_conversao_grupo, 4),
            "esforco_relativo": round(esforco_relativo, 4),
            "leads_mensal_projetado": leads_mensal_projetado,
            "matriculas_mensal_projetado": matriculas_mensal_projetado,
        })

    df_impacto = pd.DataFrame(registros)

    # Impressão formatada
    print("\n" + "=" * 90)
    print("SIMULAÇÃO DE IMPACTO DE NEGÓCIO — EduFlow Intelligence")
    print(f"Volume mensal estimado: {n_leads_mensal} leads")
    print("=" * 90)
    print(
        f"{'Threshold':>10} │ {'Contatados':>11} │ {'Matrículas':>11} │ "
        f"{'Conversão':>10} │ {'Esforço':>8} │ "
        f"{'Proj. Mensal':>13} │ {'Matr. Mensal':>13}"
    )
    print("─" * 90)

    for _, row in df_impacto.iterrows():
        print(
            f"{row['threshold']:>10.1f} │ "
            f"{row['leads_contatados']:>11} │ "
            f"{row['matriculas_esperadas']:>11} │ "
            f"{row['taxa_conversao_grupo']:>9.1%} │ "
            f"{row['esforco_relativo']:>7.1%} │ "
            f"{row['leads_mensal_projetado']:>13} │ "
            f"{row['matriculas_mensal_projetado']:>13}"
        )

    print("─" * 90)
    return df_impacto


# ===================================================================
# Geração de gráficos
# ===================================================================

def gerar_graficos_avaliacao(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_dir: str = "models",
) -> None:
    """
    Gera e salva gráficos de avaliação do modelo:
      1. Curva ROC
      2. Curva Precision-Recall
      3. Distribuição dos scores
      4. Gráfico de impacto de negócio

    Parâmetros
    ----------
    y_true : array-like
        Valores reais do target (0/1).
    y_scores : array-like
        Probabilidades preditas (scores do modelo).
    output_dir : str
        Diretório para salvar os gráficos.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Garante que o diretório de saída exista
    output_path = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Configuração visual
    sns.set_theme(style="whitegrid", font_scale=1.1)
    cores = {"primaria": "#2563EB", "secundaria": "#10B981", "destaque": "#F59E0B"}

    # ------------------------------------------------------------------
    # Figura com 4 subplots (2×2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "EduFlow Intelligence — Avaliação do Modelo",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # 1. Curva ROC
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    ax.plot(fpr, tpr, color=cores["primaria"], lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatório")
    ax.fill_between(fpr, tpr, alpha=0.1, color=cores["primaria"])
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")

    # 2. Curva Precision-Recall
    ax = axes[0, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    ax.plot(
        recall_vals, precision_vals,
        color=cores["secundaria"], lw=2,
        label=f"PR (AP = {ap:.3f})",
    )
    ax.fill_between(recall_vals, precision_vals, alpha=0.1, color=cores["secundaria"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend(loc="upper right")

    # 3. Distribuição dos scores por classe
    ax = axes[1, 0]
    mascara_pos = y_true == 1
    mascara_neg = y_true == 0
    ax.hist(
        y_scores[mascara_neg], bins=40, alpha=0.6,
        color="#EF4444", label="Não convertido (0)", density=True,
    )
    ax.hist(
        y_scores[mascara_pos], bins=40, alpha=0.6,
        color=cores["secundaria"], label="Convertido (1)", density=True,
    )
    ax.set_xlabel("Score do Modelo")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição dos Scores")
    ax.legend()

    # 4. Gráfico de impacto de negócio
    ax = axes[1, 1]
    df_impacto = simular_impacto_negocio(y_true, y_scores)

    ax2 = ax.twinx()  # eixo secundário para taxa de conversão

    # Barras: leads contatados (esforço relativo em %)
    barras = ax.bar(
        df_impacto["threshold"].astype(str),
        df_impacto["esforco_relativo"] * 100,
        color=cores["primaria"], alpha=0.6, label="Esforço relativo (%)",
    )

    # Linha: taxa de conversão do grupo
    ax2.plot(
        df_impacto["threshold"].astype(str),
        df_impacto["taxa_conversao_grupo"] * 100,
        color=cores["destaque"], marker="o", lw=2, label="Conversão do grupo (%)",
    )

    ax.set_xlabel("Threshold de Score")
    ax.set_ylabel("Esforço Relativo (%)", color=cores["primaria"])
    ax2.set_ylabel("Taxa de Conversão (%)", color=cores["destaque"])
    ax.set_title("Impacto de Negócio por Threshold")

    # Legenda combinada
    linhas1, labels1 = ax.get_legend_handles_labels()
    linhas2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(linhas1 + linhas2, labels1 + labels2, loc="upper left", fontsize=9)

    # Salva a figura
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath = os.path.join(output_path, "avaliacao_modelo.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SALVO] Gráficos de avaliação  {filepath}")

    # ------------------------------------------------------------------
    # Salva também a tabela de impacto como CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(output_path, "impacto_negocio.csv")
    df_impacto.to_csv(csv_path, index=False)
    print(f"[SALVO] Tabela de impacto    {csv_path}")
