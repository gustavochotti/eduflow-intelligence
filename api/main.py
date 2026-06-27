import sys
import os
from pathlib import Path

# Adiciona raiz do projeto ao path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import joblib
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import LeadInput, ScoreResponse, ScriptRequest, ScriptResponse, LeadResponse
from src.feature_engineering import criar_features
from src.shap_explainer import criar_explainer, explicar_lead
from src.llm_service import gerar_script_abordagem

app = FastAPI(
    title="EduFlow Intelligence API",
    description="API de scoring preditivo e geração de scripts de abordagem para leads de franquias de ensino.",
    version="1.0.0"
)

# CORS para Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global
model_pipeline = None
leads_df = None
leads_scored = None
explainer = None
metadata = None

@app.on_event("startup")
def startup():
    global model_pipeline, leads_df, leads_scored, explainer, metadata
    
    base_path = Path(__file__).parent.parent
    
    # Carrega pipeline do modelo
    model_path = base_path / 'models' / 'model.pkl'
    if not model_path.exists():
        raise RuntimeError(f"Modelo não encontrado em {model_path}. Execute model_training.py primeiro.")
    model_pipeline = joblib.load(model_path)
    
    # Carrega metadados
    meta_path = base_path / 'models' / 'metadata.json'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    
    # Carrega e pontua os leads
    data_path = base_path / 'data' / 'leads_raw.csv'
    if not data_path.exists():
        raise RuntimeError(f"Dados não encontrados em {data_path}. Execute data_generation.py primeiro.")
    
    leads_df = pd.read_csv(data_path)
    leads_with_features = criar_features(leads_df.copy())
    
    # Obtém colunas de features dos metadados
    feature_cols = metadata.get('feature_names', []) if metadata else []
    
    # Pontua todos os leads
    try:
        X = leads_with_features[feature_cols] if feature_cols else leads_with_features.drop(columns=['converteu', 'lead_id', 'nome_ficticio', 'municipio'], errors='ignore')
        scores = model_pipeline.predict_proba(X)[:, 1]
        leads_scored = leads_df.copy()
        leads_scored['score'] = scores
        leads_scored['classificacao'] = pd.cut(scores, bins=[0, 0.3, 0.6, 1.0], labels=['Baixo', 'Médio', 'Alto'])
        
        # Cria explainer SHAP
        classifier = model_pipeline.named_steps['classifier']
        preprocessor = model_pipeline.named_steps['preprocessor']
        X_transformed = preprocessor.transform(X)
        explainer = criar_explainer(classifier, X_transformed[:100])
    except Exception as e:
        print(f"Aviso: Erro ao preparar scores/SHAP: {e}")
        # Fallback: cria dataframe pontuado sem SHAP
        leads_scored = leads_df.copy()
        leads_scored['score'] = 0.5
        leads_scored['classificacao'] = 'Médio'


@app.get("/leads", response_model=list[LeadResponse])
def listar_leads(canal: str = None, curso: str = None, score_min: float = 0.0, score_max: float = 1.0):
    """Retorna tabela de leads com scores, com filtros opcionais."""
    df = leads_scored.copy()
    
    if canal:
        df = df[df['canal_captacao'] == canal]
    if curso:
        df = df[df['curso_interesse'] == curso]
    df = df[(df['score'] >= score_min) & (df['score'] <= score_max)]
    
    df = df.sort_values('score', ascending=False)
    
    result = []
    for _, row in df.iterrows():
        result.append(LeadResponse(
            lead_id=int(row['lead_id']),
            nome_ficticio=str(row['nome_ficticio']),
            canal_captacao=str(row['canal_captacao']),
            curso_interesse=str(row['curso_interesse']),
            dias_sem_contato=int(row['dias_sem_contato']),
            n_tentativas=int(row['n_tentativas']),
            respondeu_contato=int(row['respondeu_contato']),
            score=round(float(row['score']), 4),
            classificacao=str(row['classificacao'])
        ))
    return result


@app.post("/score", response_model=ScoreResponse)
def calcular_score(lead: LeadInput):
    """Recebe perfil de um lead e retorna score de conversão."""
    # Constrói DataFrame a partir da entrada
    lead_data = pd.DataFrame([lead.model_dump()])
    lead_data = criar_features(lead_data)
    
    feature_cols = metadata.get('feature_names', []) if metadata else []
    X = lead_data[feature_cols] if feature_cols else lead_data.drop(columns=['converteu', 'lead_id', 'nome_ficticio', 'municipio'], errors='ignore')
    
    score = float(model_pipeline.predict_proba(X)[:, 1][0])
    
    classificacao = 'Baixo' if score < 0.3 else ('Médio' if score < 0.6 else 'Alto')
    
    # Explicação SHAP
    shap_features = []
    if explainer is not None:
        try:
            preprocessor = model_pipeline.named_steps['preprocessor']
            X_transformed = preprocessor.transform(X)
            feature_names = metadata.get('transformed_feature_names', [f'feature_{i}' for i in range(X_transformed.shape[1])])
            shap_result = explicar_lead(explainer, X_transformed, feature_names)
            shap_features = shap_result.get('shap_top_features', [])
        except Exception:
            pass
    
    return ScoreResponse(score=round(score, 4), classificacao=classificacao, shap_top_features=shap_features)


@app.post("/script", response_model=ScriptResponse)
def gerar_script(request: ScriptRequest):
    """Gera script personalizado de abordagem para um lead."""
    if leads_scored is None:
        raise HTTPException(status_code=500, detail="Dados não carregados")
    
    lead_row = leads_scored[leads_scored['lead_id'] == request.lead_id]
    if lead_row.empty:
        raise HTTPException(status_code=404, detail=f"Lead {request.lead_id} não encontrado")
    
    lead_row = lead_row.iloc[0]
    
    # Obtém features SHAP para este lead
    shap_features = []
    if explainer is not None:
        try:
            lead_data = criar_features(pd.DataFrame([lead_row.to_dict()]))
            feature_cols = metadata.get('feature_names', []) if metadata else []
            X = lead_data[feature_cols] if feature_cols else lead_data.drop(columns=['converteu', 'lead_id', 'nome_ficticio', 'municipio', 'score', 'classificacao'], errors='ignore')
            preprocessor = model_pipeline.named_steps['preprocessor']
            X_transformed = preprocessor.transform(X)
            feature_names = metadata.get('transformed_feature_names', [f'feature_{i}' for i in range(X_transformed.shape[1])])
            shap_result = explicar_lead(explainer, X_transformed, feature_names)
            shap_features = shap_result.get('shap_top_features', [])
        except Exception:
            pass
    
    lead_context = {
        'nome_ficticio': str(lead_row['nome_ficticio']),
        'curso_interesse': str(lead_row['curso_interesse']),
        'canal_captacao': str(lead_row['canal_captacao']),
        'dias_sem_contato': int(lead_row['dias_sem_contato']),
        'n_tentativas': int(lead_row['n_tentativas']),
        'score': float(lead_row['score']),
        'shap_top_features': shap_features
    }
    
    script = gerar_script_abordagem(lead_context)
    
    # Detecta se foi LLM ou fallback
    api_key = os.getenv('OPENROUTER_API_KEY', '')
    fonte = 'llm' if api_key and api_key != 'sk-or-sua-chave-aqui' else 'fallback'
    
    return ScriptResponse(
        lead_id=request.lead_id,
        nome=str(lead_row['nome_ficticio']),
        score=round(float(lead_row['score']), 4),
        script=script,
        fonte=fonte
    )
