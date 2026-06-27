import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def carregar_template_prompt() -> str:
    """Carrega o template de prompt do arquivo externo."""
    template_path = Path(__file__).parent.parent / 'prompts' / 'script_template.txt'
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def gerar_script_abordagem(lead: dict) -> str:
    """Gera script personalizado de abordagem via LLM.
    
    Args:
        lead: dict com keys: nome_ficticio, curso_interesse, canal_captacao,
              dias_sem_contato, n_tentativas, score, shap_top_features
    
    Returns:
        Mensagem de WhatsApp personalizada.
    """
    api_key = os.getenv('OPENROUTER_API_KEY')
    model = os.getenv('LLM_MODEL', 'anthropic/claude-3.5-sonnet')
    
    if not api_key or api_key == 'sk-or-sua-chave-aqui':
        return _gerar_script_fallback(lead)
    
    template = carregar_template_prompt()
    
    # Formata features SHAP para exibição
    shap_text = _formatar_shap_features(lead.get('shap_top_features', []))
    
    prompt = template.format(
        nome=lead.get('nome_ficticio', 'Lead'),
        curso_interesse=lead.get('curso_interesse', 'Não informado'),
        canal_captacao=lead.get('canal_captacao', 'Não informado'),
        dias_sem_contato=lead.get('dias_sem_contato', 'N/A'),
        n_tentativas=lead.get('n_tentativas', 0),
        score=f"{lead.get('score', 0):.0%}",
        shap_top_features=shap_text
    )
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost:8501", # Optional
                "X-Title": "EduFlow Intelligence", # Optional
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Erro na chamada LLM: {e}")
        return _gerar_script_fallback(lead)

def _formatar_shap_features(shap_features: list) -> str:
    """Formata as top features SHAP para exibição no prompt."""
    if not shap_features:
        return 'Não disponível'
    parts = []
    for feat in shap_features[:3]:
        if isinstance(feat, dict):
            name = feat.get('feature', '')
            impact = feat.get('impact', 0)
            direction = '' if impact > 0 else ''
            parts.append(f"{name} ({direction}{abs(impact):.2f})")
        else:
            parts.append(str(feat))
    return ', '.join(parts)

def _gerar_script_fallback(lead: dict) -> str:
    """Gera um script básico quando a API LLM não está disponível.
    Usa geração baseada em template sem IA."""
    nome = lead.get('nome_ficticio', 'Lead').split()[0]
    curso = lead.get('curso_interesse', 'idiomas')
    canal = lead.get('canal_captacao', '')
    
    if canal == 'Indicação':
        intro = f"Oi {nome}!  Soube que alguém te indicou pra conhecer nosso curso de {curso}."
        corpo = "Fico feliz com a confiança! Posso te contar como funciona e tirar suas dúvidas?"
    elif lead.get('n_tentativas', 0) > 2:
        intro = f"Oi {nome}, tudo bem? Sei que já tentamos contato antes."
        corpo = f"Só queria saber se ainda tem interesse no curso de {curso} — sem compromisso, tá? "
    else:
        intro = f"Oi {nome}! Vi que você demonstrou interesse no nosso curso de {curso}."
        corpo = "Posso te contar um pouco sobre como funciona e responder suas dúvidas?"
    
    return f"{intro}\n{corpo}\nFica à vontade pra responder quando puder! "
