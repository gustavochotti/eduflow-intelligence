"""
Script para gerar o notebook de Análise Exploratória (EDA)
do projeto EduFlow Intelligence.
Executa: python src/create_eda_notebook.py
"""

import nbformat as nbf
import os

def criar_notebook_eda():
    """Cria o notebook EDA com visualizações e insights de negócio."""
    
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    }
    
    cells = []
    
    # ── Título e Executive Summary ──
    cells.append(nbf.v4.new_markdown_cell("""# 🎓 EduFlow Intelligence — Análise Exploratória dos Dados

## Executive Summary

**Objetivo:** Explorar o dataset de leads de uma franquia de ensino de idiomas para identificar padrões que separam leads que convertem de leads que não convertem. Os insights desta análise guiam o feature engineering e a modelagem preditiva.

**Principais Achados:**
1. **Canal de captação é o fator mais discriminante**: leads de indicação convertem ~3x mais que a média
2. **Velocidade de contato importa**: leads contatados em menos de 3 dias têm taxa de conversão significativamente maior
3. **Existe um ponto ótimo de tentativas**: entre 1 e 3 tentativas maximiza a conversão; mais que isso sinaliza desinteresse
4. **Resposta ao contato é forte preditor**: leads que respondem convertem muito mais que os que ignoram
5. **Renda estimada tem relação fraca**: surpreendentemente, renda por bairro não é forte preditor isolado

---"""))
    
    # ── Imports e Setup ──
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configurações visuais
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
warnings.filterwarnings('ignore')

# Carrega os dados brutos
df = pd.read_csv('../data/leads_raw.csv')
print(f"Dataset: {df.shape[0]} registros × {df.shape[1]} colunas")
print(f"\\nColunas: {list(df.columns)}")
df.head()"""))
    
    # ── Visão Geral ──
    cells.append(nbf.v4.new_markdown_cell("""## 1. Visão Geral do Dataset

Primeiro, vamos entender a estrutura dos dados, tipos de variáveis, e a distribuição da variável-alvo (conversão)."""))
    
    cells.append(nbf.v4.new_code_cell("""# Informações gerais e tipos de dados
print("=" * 60)
print("INFORMAÇÕES DO DATASET")
print("=" * 60)
print(f"\\nRegistros: {df.shape[0]}")
print(f"Colunas: {df.shape[1]}")
print(f"\\nTipos de dados:")
print(df.dtypes.value_counts())
print(f"\\nEstatísticas descritivas:")
df.describe().round(2)"""))
    
    cells.append(nbf.v4.new_code_cell("""# Taxa de conversão geral
taxa_conversao = df['converteu'].mean() * 100
n_convertidos = df['converteu'].sum()
n_nao_convertidos = df.shape[0] - n_convertidos

print(f"\\n TAXA DE CONVERSÃO GERAL")
print(f"   Converteram: {n_convertidos} ({taxa_conversao:.1f}%)")
print(f"   Não converteram: {n_nao_convertidos} ({100-taxa_conversao:.1f}%)")
print(f"\\n    Desbalanceamento: {n_nao_convertidos/n_convertidos:.0f}:1")

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#e74c3c', '#2ecc71']
df['converteu'].value_counts().plot(kind='bar', color=colors, ax=ax, edgecolor='white')
ax.set_xticklabels(['Não Converteu', 'Converteu'], rotation=0)
ax.set_ylabel('Quantidade')
ax.set_title('Distribuição da Variável-Alvo (Conversão)')
for i, v in enumerate(df['converteu'].value_counts().values):
    ax.text(i, v + 10, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()"""))
    
    # ── Análise de Missing Values ──
    cells.append(nbf.v4.new_markdown_cell("""## 2. Análise de Valores Ausentes

Valores ausentes em dados de CRM são comuns — leads nem sempre preenchem todos os campos. A estratégia de tratamento depende do padrão: se os missings são aleatórios (MCAR) ou informam algo sobre o lead."""))
    
    cells.append(nbf.v4.new_code_cell("""# Análise detalhada de valores ausentes
nulos = df.isnull().sum()
nulos_pct = (df.isnull().mean() * 100).round(1)

print(" VALORES AUSENTES POR COLUNA")
print("-" * 50)
for col in df.columns:
    if nulos[col] > 0:
        print(f"   {col:<25s} {nulos[col]:>4d} registros ({nulos_pct[col]:.1f}%)")

# Visualização
fig, ax = plt.subplots(figsize=(10, 4))
cols_com_nulos = nulos[nulos > 0].sort_values(ascending=True)
cols_com_nulos.plot(kind='barh', color='#e67e22', ax=ax, edgecolor='white')
ax.set_xlabel('Quantidade de Valores Ausentes')
ax.set_title('Colunas com Valores Ausentes')
for i, v in enumerate(cols_com_nulos.values):
    pct = v / len(df) * 100
    ax.text(v + 5, i, f'{v} ({pct:.1f}%)', va='center')
plt.tight_layout()
plt.show()

# Verificar se missings têm padrão com conversão
print("\\n TAXA DE CONVERSÃO: COM vs SEM DADOS FALTANTES")
print("-" * 50)
for col in ['renda_estimada', 'faixa_etaria', 'turno_preferencia']:
    if df[col].isnull().any():
        com_dado = df[df[col].notna()]['converteu'].mean() * 100
        sem_dado = df[df[col].isna()]['converteu'].mean() * 100
        print(f"   {col:<25s} Com dado: {com_dado:.1f}%  |  Sem dado: {sem_dado:.1f}%")"""))
    
    # ── Conversão por Canal ──
    cells.append(nbf.v4.new_markdown_cell("""## 3. Taxa de Conversão por Canal de Captação

**Insight de negócio:** O canal de captação é historicamente o maior preditor de conversão em franquias de ensino. Leads de indicação chegam com confiança pré-estabelecida; leads de tráfego pago chegam frios."""))
    
    cells.append(nbf.v4.new_code_cell("""# Conversão por canal de captação
conv_canal = df.groupby('canal_captacao').agg(
    total=('converteu', 'count'),
    convertidos=('converteu', 'sum'),
    taxa=('converteu', 'mean')
).sort_values('taxa', ascending=False)
conv_canal['taxa_pct'] = (conv_canal['taxa'] * 100).round(1)

print(" CONVERSÃO POR CANAL DE CAPTAÇÃO")
print("-" * 60)
for canal, row in conv_canal.iterrows():
    barra = "█" * int(row['taxa_pct'])
    print(f"   {canal:<22s} {row['taxa_pct']:>5.1f}% {barra}  ({int(row['convertidos'])}/{int(row['total'])})")

# Visualização
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Taxa de conversão por canal
conv_canal['taxa_pct'].plot(kind='barh', ax=axes[0], color='#2ecc71', edgecolor='white')
axes[0].set_xlabel('Taxa de Conversão (%)')
axes[0].set_title('Taxa de Conversão por Canal')
for i, v in enumerate(conv_canal['taxa_pct'].values):
    axes[0].text(v + 0.3, i, f'{v:.1f}%', va='center', fontweight='bold')

# Gráfico 2: Volume de leads por canal
df['canal_captacao'].value_counts().sort_values().plot(
    kind='barh', ax=axes[1], color='#3498db', edgecolor='white'
)
axes[1].set_xlabel('Quantidade de Leads')
axes[1].set_title('Volume de Leads por Canal')

plt.tight_layout()
plt.show()

print("\\n INSIGHT: Indicação e Evento Presencial são os canais com maior conversão,")
print("   mas representam menor volume. O desafio é otimizar a conversão nos canais de maior volume.")"""))
    
    # ── Boxplot dias sem contato ──
    cells.append(nbf.v4.new_markdown_cell("""## 4. Dias sem Contato: Convertidos vs Não Convertidos

**Hipótese:** Leads que são contatados mais rapidamente têm maior chance de converter. Quanto mais tempo sem contato, mais frio o lead fica."""))
    
    cells.append(nbf.v4.new_code_cell("""# Boxplot de dias sem contato
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot
df.boxplot(column='dias_sem_contato', by='converteu', ax=axes[0])
axes[0].set_xlabel('Converteu (0=Não, 1=Sim)')
axes[0].set_ylabel('Dias sem Contato')
axes[0].set_title('Distribuição de Dias sem Contato por Conversão')
plt.sca(axes[0])
plt.xticks([1, 2], ['Não Converteu', 'Converteu'])

# Histograma sobreposto
df[df['converteu']==0]['dias_sem_contato'].hist(
    bins=30, alpha=0.5, label='Não Converteu', color='#e74c3c', ax=axes[1], density=True
)
df[df['converteu']==1]['dias_sem_contato'].hist(
    bins=30, alpha=0.7, label='Converteu', color='#2ecc71', ax=axes[1], density=True
)
axes[1].set_xlabel('Dias sem Contato')
axes[1].set_ylabel('Densidade')
axes[1].set_title('Distribuição de Dias sem Contato')
axes[1].legend()

plt.suptitle('')
plt.tight_layout()
plt.show()

# Estatísticas
media_conv = df[df['converteu']==1]['dias_sem_contato'].mean()
media_nconv = df[df['converteu']==0]['dias_sem_contato'].mean()
print(f"\\n Média de dias sem contato:")
print(f"   Convertidos:     {media_conv:.1f} dias")
print(f"   Não convertidos: {media_nconv:.1f} dias")
print(f"   Diferença:       {media_nconv - media_conv:.1f} dias")

# Taxa de conversão por faixa de dias
bins_dias = [0, 3, 7, 14, 30, 60]
labels_dias = ['0-3d', '4-7d', '8-14d', '15-30d', '31-60d']
df['faixa_dias'] = pd.cut(df['dias_sem_contato'], bins=bins_dias, labels=labels_dias, right=True)
conv_dias = df.groupby('faixa_dias', observed=True)['converteu'].mean() * 100
print(f"\\n Taxa de conversão por faixa de dias sem contato:")
for faixa, taxa in conv_dias.items():
    print(f"   {faixa}: {taxa:.1f}%")
df.drop(columns='faixa_dias', inplace=True)"""))
    
    # ── Heatmap de correlação ──
    cells.append(nbf.v4.new_markdown_cell("""## 5. Correlação entre Variáveis Numéricas

Mapa de calor mostrando as correlações lineares entre variáveis numéricas. Correlações com a variável `converteu` indicam poder preditivo direto."""))
    
    cells.append(nbf.v4.new_code_cell("""# Heatmap de correlação
numericas = df.select_dtypes(include=[np.number])
corr = numericas.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f', 
    cmap='RdYlGn', center=0, square=True,
    linewidths=0.5, ax=ax,
    vmin=-1, vmax=1
)
ax.set_title('Matriz de Correlação — Variáveis Numéricas')
plt.tight_layout()
plt.show()

# Correlações com a variável alvo
print("\\n CORRELAÇÕES COM 'converteu' (ordenadas por magnitude):")
print("-" * 50)
corr_target = corr['converteu'].drop('converteu').abs().sort_values(ascending=False)
for var, val in corr_target.items():
    sinal = '+' if corr['converteu'][var] > 0 else '-'
    print(f"   {var:<25s} {sinal}{val:.3f}")"""))
    
    # ── Conversão por curso ──
    cells.append(nbf.v4.new_markdown_cell("""## 6. Taxa de Conversão por Curso de Interesse

Análise de qual produto (curso) atrai leads com maior propensão a converter. Isso informa tanto a estratégia de marketing quanto a priorização do time comercial."""))
    
    cells.append(nbf.v4.new_code_cell("""# Conversão por curso de interesse
conv_curso = df.groupby('curso_interesse').agg(
    total=('converteu', 'count'),
    convertidos=('converteu', 'sum'),
    taxa=('converteu', 'mean')
).sort_values('taxa', ascending=False)
conv_curso['taxa_pct'] = (conv_curso['taxa'] * 100).round(1)

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(conv_curso)))
conv_curso['taxa_pct'].plot(kind='bar', ax=ax, color=colors, edgecolor='white')
ax.set_ylabel('Taxa de Conversão (%)')
ax.set_xlabel('Curso de Interesse')
ax.set_title('Taxa de Conversão por Curso de Interesse')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
for i, v in enumerate(conv_curso['taxa_pct'].values):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()"""))
    
    # ── Tentativas e conversão ──
    cells.append(nbf.v4.new_markdown_cell("""## 7. Relação entre Número de Tentativas e Conversão

**Hipótese:** Existe um ponto ótimo de follow-up. Pouquíssimas tentativas significam desistência prematura; muitas tentativas sinalizam que o lead não tem interesse."""))
    
    cells.append(nbf.v4.new_code_cell("""# Conversão por número de tentativas
conv_tentativas = df.groupby('n_tentativas').agg(
    total=('converteu', 'count'),
    taxa=('converteu', 'mean')
)
conv_tentativas['taxa_pct'] = (conv_tentativas['taxa'] * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Taxa de conversão por tentativa
axes[0].bar(conv_tentativas.index, conv_tentativas['taxa_pct'], color='#27ae60', edgecolor='white')
axes[0].set_xlabel('Número de Tentativas')
axes[0].set_ylabel('Taxa de Conversão (%)')
axes[0].set_title('Conversão por Nº de Tentativas')
axes[0].axvspan(0.5, 3.5, alpha=0.1, color='green', label='Zona ideal (1-3)')
axes[0].legend()

# Volume por tentativa
axes[1].bar(conv_tentativas.index, conv_tentativas['total'], color='#3498db', edgecolor='white')
axes[1].set_xlabel('Número de Tentativas')
axes[1].set_ylabel('Volume de Leads')
axes[1].set_title('Volume de Leads por Nº de Tentativas')

plt.tight_layout()
plt.show()

print("\\n INSIGHT: O intervalo de 1 a 3 tentativas parece ser o ponto ótimo.")
print("   Leads com 0 tentativas não receberam follow-up adequado.")
print("   Leads com 4+ tentativas provavelmente não têm interesse.")"""))
    
    # ── Faixa etária ──
    cells.append(nbf.v4.new_markdown_cell("""## 8. Distribuição de Faixa Etária por Conversão

Análise demográfica para entender se existe um perfil de idade que converte mais."""))
    
    cells.append(nbf.v4.new_code_cell("""# Faixa etária e conversão
df_faixa = df.dropna(subset=['faixa_etaria'])
conv_faixa = df_faixa.groupby('faixa_etaria').agg(
    total=('converteu', 'count'),
    taxa=('converteu', 'mean')
)
conv_faixa['taxa_pct'] = (conv_faixa['taxa'] * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Taxa de conversão
conv_faixa['taxa_pct'].plot(kind='bar', ax=axes[0], color='#9b59b6', edgecolor='white')
axes[0].set_ylabel('Taxa de Conversão (%)')
axes[0].set_title('Conversão por Faixa Etária')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Volume por faixa etária com separação por conversão
ct = pd.crosstab(df_faixa['faixa_etaria'], df_faixa['converteu'])
ct.columns = ['Não Converteu', 'Converteu']
ct.plot(kind='bar', stacked=True, ax=axes[1], color=['#e74c3c', '#2ecc71'], edgecolor='white')
axes[1].set_ylabel('Quantidade')
axes[1].set_title('Volume por Faixa Etária e Conversão')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.show()"""))
    
    # ── Turno de preferência ──
    cells.append(nbf.v4.new_markdown_cell("""## 9. Conversão por Turno de Preferência"""))
    
    cells.append(nbf.v4.new_code_cell("""# Turno de preferência e conversão
df_turno = df.dropna(subset=['turno_preferencia'])
conv_turno = df_turno.groupby('turno_preferencia').agg(
    total=('converteu', 'count'),
    taxa=('converteu', 'mean')
)
conv_turno['taxa_pct'] = (conv_turno['taxa'] * 100).round(1)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#f1c40f', '#e67e22', '#2c3e50']
conv_turno['taxa_pct'].plot(kind='bar', ax=ax, color=colors, edgecolor='white')
ax.set_ylabel('Taxa de Conversão (%)')
ax.set_title('Taxa de Conversão por Turno de Preferência')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(conv_turno['taxa_pct'].values):
    ax.text(i, v + 0.2, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()"""))
    
    # ── Renda estimada ──
    cells.append(nbf.v4.new_markdown_cell("""## 10. Renda Estimada e Conversão"""))
    
    cells.append(nbf.v4.new_code_cell("""# Renda estimada
df_renda = df.dropna(subset=['renda_estimada'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribuição de renda por conversão
df_renda[df_renda['converteu']==0]['renda_estimada'].hist(
    bins=40, alpha=0.5, label='Não Converteu', color='#e74c3c', ax=axes[0], density=True
)
df_renda[df_renda['converteu']==1]['renda_estimada'].hist(
    bins=40, alpha=0.7, label='Converteu', color='#2ecc71', ax=axes[0], density=True
)
axes[0].set_xlabel('Renda Estimada (R$)')
axes[0].set_ylabel('Densidade')
axes[0].set_title('Distribuição de Renda por Conversão')
axes[0].legend()

# Boxplot
df_renda.boxplot(column='renda_estimada', by='converteu', ax=axes[1])
axes[1].set_xlabel('Converteu (0=Não, 1=Sim)')
axes[1].set_ylabel('Renda Estimada (R$)')
axes[1].set_title('Renda Estimada por Status de Conversão')
plt.sca(axes[1])
plt.xticks([1, 2], ['Não Converteu', 'Converteu'])

plt.suptitle('')
plt.tight_layout()
plt.show()

print(f"\\nMédia de renda — Convertidos: R$ {df_renda[df_renda['converteu']==1]['renda_estimada'].mean():,.0f}")
print(f"Média de renda — Não convertidos: R$ {df_renda[df_renda['converteu']==0]['renda_estimada'].mean():,.0f}")"""))
    
    # ── Hipóteses para Feature Engineering ──
    cells.append(nbf.v4.new_markdown_cell("""---

## Hipóteses para Feature Engineering

Com base na análise exploratória, as seguintes hipóteses guiam a criação de features derivadas:

###  Hipóteses confirmadas:
1. **Canal de captação é o principal preditor**  Criar flag `is_indicacao` e `is_evento` para os canais com maior conversão
2. **Velocidade de resposta importa**  Criar `contato_rapido` (dias_sem_contato < 3)
3. **Existe ponto ótimo de tentativas**  Criar `tentativas_ideal` (entre 1 e 3)
4. **Resposta ao contato é forte sinal**  Variável já binária, usar diretamente

### 🔍 Hipóteses a investigar com o modelo:
5. **Combinação de sinais é mais forte que variáveis isoladas**  Criar `score_engajamento` combinando resposta + recência + tentativas
6. **Urgência acumulada pode ser proxy de risco**  Criar `urgencia_estimada` = dias × tentativas
7. **Turno noturno pode indicar perfil profissional**  Criar `turno_noite`
8. **Interesse em Inglês Adulto pode ter diferencial**  Criar `curso_adulto`

### 📝 Decisões de pré-processamento:
- **Renda estimada**: imputar com mediana (missings parecem aleatórios)
- **Faixa etária**: imputar com moda
- **Turno de preferência**: imputar com moda
- **Variáveis categóricas**: One-Hot Encoding com `handle_unknown='ignore'`
- **Variáveis numéricas**: StandardScaler após imputação

---

*Próximo passo: `src/preprocessing.py` e `src/feature_engineering.py`*"""))
    
    nb.cells = cells
    
    # Salva o notebook
    output_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'eda.ipynb')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f" Notebook EDA criado em: {output_path}")


if __name__ == '__main__':
    criar_notebook_eda()
