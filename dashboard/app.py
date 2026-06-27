"""
EduFlow Intelligence — Dashboard Interativo
Dashboard Streamlit com 3 visões: Gestão, Operacional e Ação.
Consome a API FastAPI rodando localmente.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduFlow Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─────────────────────────────────────────────
# CSS customizado para visual premium
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Tipografia e cores globais */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d7377 50%, #14a085 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }
    
    /* Cards de KPI */
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #14a085;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    
    /* Score badges */
    .score-alto {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-medio {
        background: linear-gradient(135deg, #fdcb6e, #e17055);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-baixo {
        background: linear-gradient(135deg, #636e72, #b2bec3);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Lead detail card */
    .lead-detail {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
        border: 1px solid rgba(20, 160, 133, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    /* Script output */
    .script-output {
        background: linear-gradient(135deg, #0d2137 0%, #0a1628 100%);
        border-left: 4px solid #14a085;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.7;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #0f0f23;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def carregar_leads(canal=None, curso=None, score_min=0.0, score_max=1.0):
    """Carrega leads da API com filtros opcionais."""
    params = {"score_min": score_min, "score_max": score_max}
    if canal and canal != "Todos":
        params["canal"] = canal
    if curso and curso != "Todos":
        params["curso"] = curso
    
    try:
        resp = requests.get(f"{API_URL}/leads", params=params, timeout=10)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API não disponível. Execute: `uvicorn api.main:app --reload` na pasta do projeto.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar leads: {e}")
        return pd.DataFrame()


def solicitar_script(lead_id: int) -> dict:
    """Solicita geração de script de abordagem via API."""
    try:
        resp = requests.post(
            f"{API_URL}/script",
            json={"lead_id": lead_id},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"erro": "API não disponível"}
    except Exception as e:
        return {"erro": str(e)}


def classificacao_badge(classificacao: str) -> str:
    """Retorna HTML do badge de classificação."""
    cls = classificacao.lower()
    return f'<span class="score-{cls}">{classificacao}</span>'


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 EduFlow Intelligence</h1>
    <p>Sistema de Scoring Preditivo + Assistente de Abordagem para Franquias de Ensino</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar — Filtros
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filtros")
    st.markdown("---")
    
    # Carregar leads sem filtros para pegar opções
    df_all = carregar_leads()
    
    if df_all.empty:
        st.warning("Sem dados disponíveis.")
        st.stop()
    
    canais = ["Todos"] + sorted(df_all["canal_captacao"].unique().tolist())
    cursos = ["Todos"] + sorted(df_all["curso_interesse"].unique().tolist())
    
    canal_filtro = st.selectbox("Canal de Captação", canais, index=0)
    curso_filtro = st.selectbox("Curso de Interesse", cursos, index=0)
    
    st.markdown("#### 📊 Faixa de Score")
    score_range = st.slider(
        "Score de Conversão",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        format="%.2f"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Sobre")
    st.markdown(
        "Sistema de inteligência de leads que identifica os contatos "
        "com maior probabilidade de conversão e gera scripts "
        "personalizados de abordagem."
    )
    st.markdown("---")
    st.caption("EduFlow Intelligence v1.0")


# ─────────────────────────────────────────────
# Carregar dados filtrados
# ─────────────────────────────────────────────
df = carregar_leads(
    canal=canal_filtro,
    curso=curso_filtro,
    score_min=score_range[0],
    score_max=score_range[1]
)

if df.empty:
    st.warning("Nenhum lead encontrado com os filtros selecionados.")
    st.stop()


# ─────────────────────────────────────────────
# Tabs — 3 Visões
# ─────────────────────────────────────────────
tab_gestao, tab_operacional, tab_acao = st.tabs([
    "📈 Visão de Gestão",
    "📋 Visão Operacional",
    "🎯 Visão de Ação"
])


# ═══════════════════════════════════════════
# TAB 1 — VISÃO DE GESTÃO
# ═══════════════════════════════════════════
with tab_gestao:
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    total_leads = len(df)
    media_score = df["score"].mean()
    leads_alto = len(df[df["classificacao"] == "Alto"])
    pct_alto = (leads_alto / total_leads * 100) if total_leads > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{total_leads:,}</p>
            <p class="kpi-label">Total de Leads</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{media_score:.1%}</p>
            <p class="kpi-label">Score Médio</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{leads_alto}</p>
            <p class="kpi-label">Leads Prioritários</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{pct_alto:.0f}%</p>
            <p class="kpi-label">% Alta Conversão</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráficos lado a lado
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Distribuição de scores
        fig_hist = px.histogram(
            df,
            x="score",
            nbins=30,
            color_discrete_sequence=["#14a085"],
            title="Distribuição de Scores de Conversão",
            labels={"score": "Score", "count": "Quantidade"}
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title_font_size=14,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            bargap=0.05
        )
        fig_hist.add_vline(
            x=0.6, line_dash="dash", line_color="#e17055",
            annotation_text="Threshold (0.6)", annotation_position="top right"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_g2:
        # Conversão por canal
        canal_stats = df.groupby("canal_captacao").agg(
            score_medio=("score", "mean"),
            quantidade=("score", "count")
        ).reset_index().sort_values("score_medio", ascending=True)
        
        fig_canal = px.bar(
            canal_stats,
            y="canal_captacao",
            x="score_medio",
            orientation="h",
            color="score_medio",
            color_continuous_scale=["#636e72", "#14a085"],
            title="Score Médio por Canal de Captação",
            labels={"canal_captacao": "Canal", "score_medio": "Score Médio"},
            text=canal_stats["score_medio"].apply(lambda x: f"{x:.1%}")
        )
        fig_canal.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title_font_size=14,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        fig_canal.update_traces(textposition="outside")
        st.plotly_chart(fig_canal, use_container_width=True)
    
    # Segunda linha de gráficos
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        # Score por curso
        curso_stats = df.groupby("curso_interesse").agg(
            score_medio=("score", "mean"),
            quantidade=("score", "count")
        ).reset_index().sort_values("score_medio", ascending=False)
        
        fig_curso = px.bar(
            curso_stats,
            x="curso_interesse",
            y="score_medio",
            color="score_medio",
            color_continuous_scale=["#636e72", "#00b894"],
            title="Score Médio por Curso de Interesse",
            labels={"curso_interesse": "Curso", "score_medio": "Score Médio"},
            text=curso_stats["score_medio"].apply(lambda x: f"{x:.1%}")
        )
        fig_curso.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title_font_size=14,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickangle=-30),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        fig_curso.update_traces(textposition="outside")
        st.plotly_chart(fig_curso, use_container_width=True)
    
    with col_g4:
        # Distribuição por classificação (donut chart)
        class_counts = df["classificacao"].value_counts().reset_index()
        class_counts.columns = ["classificacao", "count"]
        
        color_map = {"Alto": "#00b894", "Médio": "#fdcb6e", "Baixo": "#636e72"}
        
        fig_donut = px.pie(
            class_counts,
            values="count",
            names="classificacao",
            title="Distribuição por Classificação",
            color="classificacao",
            color_discrete_map=color_map,
            hole=0.55
        )
        fig_donut.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title_font_size=14,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        fig_donut.update_traces(textinfo="percent+value", textfont_size=12)
        st.plotly_chart(fig_donut, use_container_width=True)
    
    # Simulação de impacto
    st.markdown("### 💡 Simulação de Impacto")
    st.markdown(
        "Se o time comercial priorizar apenas os leads com **score acima de 0.6**, "
        "o esforço de contato é concentrado nos leads com maior probabilidade de conversão."
    )
    
    col_imp1, col_imp2, col_imp3 = st.columns(3)
    with col_imp1:
        st.metric("Leads no Foco", f"{leads_alto}", f"{pct_alto:.0f}% do total")
    with col_imp2:
        esforco_poupado = 100 - pct_alto
        st.metric("Esforço Poupado", f"{esforco_poupado:.0f}%", "menos contatos")
    with col_imp3:
        if leads_alto > 0:
            score_medio_alto = df[df["classificacao"] == "Alto"]["score"].mean()
            st.metric("Score Médio (Foco)", f"{score_medio_alto:.1%}", "vs geral")
        else:
            st.metric("Score Médio (Foco)", "N/A", "")


# ═══════════════════════════════════════════
# TAB 2 — VISÃO OPERACIONAL
# ═══════════════════════════════════════════
with tab_operacional:
    st.markdown("### 📋 Tabela de Leads — Ordenados por Score")
    st.markdown(f"*Exibindo **{len(df)}** leads com os filtros aplicados.*")
    
    # Preparar dataframe para exibição
    df_display = df[[
        "lead_id", "nome_ficticio", "canal_captacao", "curso_interesse",
        "dias_sem_contato", "n_tentativas", "respondeu_contato",
        "score", "classificacao"
    ]].copy()
    
    df_display.columns = [
        "ID", "Nome", "Canal", "Curso",
        "Dias s/ Contato", "Tentativas", "Respondeu",
        "Score", "Classificação"
    ]
    
    # Formatar score como percentual
    df_display["Score"] = df_display["Score"].apply(lambda x: f"{x:.1%}")
    df_display["Respondeu"] = df_display["Respondeu"].apply(lambda x: "✅" if x == 1 else "❌")
    
    # Exibir tabela interativa
    st.dataframe(
        df_display,
        use_container_width=True,
        height=500,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Nome": st.column_config.TextColumn("Nome", width="medium"),
            "Canal": st.column_config.TextColumn("Canal", width="medium"),
            "Curso": st.column_config.TextColumn("Curso", width="medium"),
            "Dias s/ Contato": st.column_config.NumberColumn("Dias s/ Contato", width="small"),
            "Tentativas": st.column_config.NumberColumn("Tentativas", width="small"),
            "Respondeu": st.column_config.TextColumn("Respondeu", width="small"),
            "Score": st.column_config.TextColumn("Score", width="small"),
            "Classificação": st.column_config.TextColumn("Classificação", width="small"),
        }
    )
    
    # Estatísticas rápidas
    st.markdown("---")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown("**🏆 Top Canal (Score Médio)**")
        top_canal = df.groupby("canal_captacao")["score"].mean().idxmax()
        st.markdown(f"→ {top_canal}")
    with col_s2:
        st.markdown("**📚 Top Curso (Score Médio)**")
        top_curso = df.groupby("curso_interesse")["score"].mean().idxmax()
        st.markdown(f"→ {top_curso}")
    with col_s3:
        st.markdown("**📞 Leads com Resposta**")
        pct_resp = (df["respondeu_contato"].sum() / len(df) * 100)
        st.markdown(f"→ {pct_resp:.0f}% responderam")


# ═══════════════════════════════════════════
# TAB 3 — VISÃO DE AÇÃO
# ═══════════════════════════════════════════
with tab_acao:
    st.markdown("### 🎯 Perfil do Lead e Geração de Script")
    
    # Seleção do lead
    lead_options = df.sort_values("score", ascending=False)
    lead_choices = {
        f"{row['nome_ficticio']} — Score: {row['score']:.0%} ({row['classificacao']})": row["lead_id"]
        for _, row in lead_options.iterrows()
    }
    
    selected_lead_label = st.selectbox(
        "Selecione um lead para ver o perfil e gerar script de abordagem:",
        options=list(lead_choices.keys()),
        index=0
    )
    
    if selected_lead_label:
        selected_lead_id = lead_choices[selected_lead_label]
        lead_info = df[df["lead_id"] == selected_lead_id].iloc[0]
        
        # Card de perfil do lead
        st.markdown("---")
        
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p1:
            st.markdown(f"""
            <div class="lead-detail">
                <h3 style="margin-top:0; color: #14a085;">👤 {lead_info['nome_ficticio']}</h3>
                <table style="width:100%; color: #e0e0e0; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.5);">📚 Curso de Interesse</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 500;">{lead_info['curso_interesse']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.5);">📱 Canal de Origem</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 500;">{lead_info['canal_captacao']}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.5);">📅 Dias sem Contato</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 500;">{lead_info['dias_sem_contato']} dias</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.5);">📞 Tentativas</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 500;">{lead_info['n_tentativas']} tentativa(s)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: rgba(255,255,255,0.5);">💬 Respondeu</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 500;">{'✅ Sim' if lead_info['respondeu_contato'] == 1 else '❌ Não'}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col_p2:
            # Score gauge
            score_val = lead_info["score"]
            classificacao = lead_info["classificacao"]
            
            color_map_gauge = {"Alto": "#00b894", "Médio": "#fdcb6e", "Baixo": "#636e72"}
            gauge_color = color_map_gauge.get(classificacao, "#636e72")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_val * 100,
                number={"suffix": "%", "font": {"size": 36, "color": "white"}},
                title={"text": "Score de Conversão", "font": {"size": 14, "color": "rgba(255,255,255,0.7)"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)"},
                    "bar": {"color": gauge_color},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(99,110,114,0.2)"},
                        {"range": [30, 60], "color": "rgba(253,203,110,0.2)"},
                        {"range": [60, 100], "color": "rgba(0,184,148,0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#e17055", "width": 3},
                        "thickness": 0.8,
                        "value": 60
                    }
                }
            ))
            fig_gauge.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown(
                f'<div style="text-align:center;">{classificacao_badge(classificacao)}</div>',
                unsafe_allow_html=True
            )
        
        # Botão de geração de script
        st.markdown("---")
        st.markdown("### 💬 Script de Abordagem")
        
        col_btn, col_space = st.columns([1, 3])
        with col_btn:
            gerar = st.button(
                "🚀 Gerar Script de Abordagem",
                type="primary",
                use_container_width=True
            )
        
        if gerar:
            with st.spinner("Gerando script personalizado..."):
                resultado = solicitar_script(selected_lead_id)
            
            if "erro" in resultado:
                st.error(f"Erro: {resultado['erro']}")
            else:
                fonte_label = "🤖 Gerado por IA" if resultado.get("fonte") == "llm" else "📝 Gerado por template (sem API key)"
                st.caption(fonte_label)
                
                st.markdown(f"""
                <div class="script-output">
                    {resultado.get('script', 'Script não disponível').replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
                
                # Botão de copiar
                st.code(resultado.get("script", ""), language=None)
                st.caption("☝️ Copie o texto acima para usar no WhatsApp")
        
        # Dica quando não clicou
        elif "gerar" not in st.session_state:
            st.info(
                "👆 Clique em **Gerar Script de Abordagem** para criar uma mensagem "
                "personalizada de WhatsApp para este lead, baseada no perfil e no "
                "score de conversão."
            )
