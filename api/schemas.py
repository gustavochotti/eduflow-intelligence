from pydantic import BaseModel, Field
from typing import Optional

class LeadInput(BaseModel):
    """Schema para entrada de um novo lead para scoring."""
    canal_captacao: str = Field(..., description="Canal de captação do lead", examples=["Indicação"])
    curso_interesse: str = Field(..., description="Curso de interesse", examples=["Inglês Adulto"])
    faixa_etaria: Optional[str] = Field(None, description="Faixa etária", examples=["25-34"])
    turno_preferencia: Optional[str] = Field(None, description="Turno de preferência", examples=["Noite"])
    dias_sem_contato: int = Field(..., ge=0, description="Dias desde o último contato")
    n_tentativas: int = Field(..., ge=0, description="Número de tentativas de contato")
    respondeu_contato: int = Field(..., ge=0, le=1, description="Se respondeu algum contato (0 ou 1)")
    renda_estimada: Optional[float] = Field(None, description="Renda estimada por bairro")
    nome_ficticio: Optional[str] = Field('Lead', description="Nome fictício do lead")

class ScoreResponse(BaseModel):
    """Schema de resposta do scoring."""
    score: float = Field(..., description="Probabilidade de conversão (0-1)")
    classificacao: str = Field(..., description="Alto/Médio/Baixo")
    shap_top_features: list = Field(..., description="Top-3 features que explicam o score")

class ScriptRequest(BaseModel):
    """Schema para requisição de script de abordagem."""
    lead_id: int = Field(..., description="ID do lead na base")

class ScriptResponse(BaseModel):
    """Schema de resposta com script gerado."""
    lead_id: int
    nome: str
    score: float
    script: str = Field(..., description="Script de abordagem personalizado")
    fonte: str = Field(..., description="'llm' ou 'fallback'")

class LeadResponse(BaseModel):
    """Schema de um lead na listagem."""
    lead_id: int
    nome_ficticio: str
    canal_captacao: str
    curso_interesse: str
    dias_sem_contato: int
    n_tentativas: int
    respondeu_contato: int
    score: float
    classificacao: str
