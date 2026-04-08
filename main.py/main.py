from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
import requests
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import re

app = FastAPI(
    title="Futebol Lineup Analysis API",
    version="2.0.0",
    description="API para análise de prováveis escalações com priorização de fontes confiáveis."
)

# =========================
# MODELOS
# =========================

class MatchLineupRequest(BaseModel):
    team_name: str = Field(..., description="Nome do time")
    opponent_name: Optional[str] = Field(None, description="Nome do adversário")

class SourceItem(BaseModel):
    source_name: str
    source_type: str
    url: str

class MatchLineupResponse(BaseModel):
    team_name: str
    opponent_name: Optional[str] = None
    probable_lineup: List[str]
    confirmed_absences: List[str]
    doubtful_players: List[str]
    likely_rest_players: List[str]
    status: Literal["forca_maxima", "misto", "reservas"]
    confidence: Literal["alta", "media", "baixa"]
    summary: str
    sources: List[SourceItem] = []

# =========================
# CONFIGURAÇÕES
# =========================

SOURCE_PRIORITY = {
    "ge.globo.com": 5,
    "globoesporte.globo.com": 5,
    "espn.com.br": 4,
    "uol.com.br": 3,
    "lance.com.br": 3,
    "cnnbrasil.com.br": 2,
}

HIGH_PRIORITY_SOURCES = {"ge.globo.com", "globoesporte.globo.com", "espn.com.br"}

# palavras que indicam sinais relevantes
REST_KEYWORDS = [
    "poupado", "poupados", "preservado", "preservados",
    "rodízio", "time misto", "misto", "reservas", "poupar"
]

FULL_STRENGTH_KEYWORDS = [
    "força máxima", "forca maxima", "titulares", "time ideal",
    "escalação ideal", "força total"
]

ABSENCE_KEYWORDS = [
    "desfalque", "desfalques", "lesionado", "lesão", "lesao",
    "suspenso", "fora", "não joga", "nao joga", "afastado"
]

DOUBT_KEYWORDS = [
    "dúvida", "duvida", "incerto", "pode ficar fora", "ainda será avaliado",
    "em observação", "transição", "transicao"
]

LIKELY_LINEUP_KEYWORDS = [
    "provável escalação", "provavel escalação", "provavel escalacao",
    "escalação provável", "escalacao provavel"
]

# alguns nomes fortes para detectar jogadores citados em títulos/snippets
COMMON_PLAYER_PREFIX = [
    "sem ", "sem o ", "sem a ", "com ", "poupa ", "poupar ", "dúvida em ",
    "duvida em ", "volta de ", "retorno de "
]

# =========================
# FUNÇÕES AUXILIARES
# =========================

def clean_html(text: str) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()

def source_weight(link: str) -> int:
    for domain, weight in SOURCE_PRIORITY.items():
        if domain in link:
            return weight
    return 1

def source_name_from_link(link: str) -> str:
    for domain in SOURCE_PRIORITY.keys():
        if domain in link:
            return domain
    return "outra_fonte"

def source_type_from_link(link: str) -> str:
    if "ge.globo.com" in link or "globoesporte.globo.com" in link or "espn.com.br" in link:
        return "midia_esportiva"
    return "outra"

def build_queries(team_name: str, opponent_name: Optional[str]) -> List[str]:
    queries = []

    base = f'"{team_name}"'
    if opponent_name:
        versus = f'"{team_name}" "{opponent_name}"'
        queries.append(f"{versus} provável escalação")
        queries.append(f"{versus} poupados desfalques")
        queries.append(f"{versus} titulares reservas")
    else:
        queries.append(f'{base} provável escalação')
        queries.append(f'{base} poupados desfalques')
        queries.append(f'{base} titulares reservas')

    # consultas com preferência implícita para ge e veículos esportivos
    if opponent_name:
        queries.append(f'"{team_name}" "{opponent_name}" ge provável escalação')
        queries.append(f'"{team_name}" "{opponent_name}" ESPN desfalques')
    else:
        queries.append(f'"{team_name}" ge provável escalação')
        queries.append(f'"{team_name}" ESPN desfalques')

    return queries

def google_news_rss(query: str, limit: int = 8) -> List[dict]:
    """
    Busca notícias no Google News RSS.
    """
    rss_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    feed = feedparser.parse(rss_url)

    items = []
    for entry in feed.entries[:limit]:
        title = entry.get("title", "")
        link = entry.get("link", "")
        summary = clean_html(entry.get("summary", ""))

        items.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": entry.get("published", ""),
            "weight": source_weight(link),
            "source_name": source_name_from_link(link),
            "source_type": source_type_from_link(link),
        })

    return items

def deduplicate_news(items: List[dict]) -> List[dict]:
    seen = set()
    result = []

    for item in sorted(items, key=lambda x: x["weight"], reverse=True):
        key = (item["title"].strip().lower(), item["link"].strip().lower())
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result

def extract_named_entities_like_players(text: str) -> List[str]:
    """
    Heurística simples: tenta encontrar palavras após gatilhos como
    'sem', 'poupa', 'dúvida em', etc.
    Não é perfeito, mas ajuda no protótipo.
    """
    found = []
    txt = text.strip()

    patterns = [
        r"(?:sem|poupa|poupar|dúvida em|duvida em|volta de|retorno de)\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç\-]+)",
        r"(?:sem o|sem a)\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç\-]+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, txt, flags=re.IGNORECASE)
        for m in matches:
            candidate = m.strip()
            if candidate and candidate.lower() not in {"time", "titular", "clássico", "jogo"}:
                found.append(candidate)

    return list(dict.fromkeys(found))

def analyze_news(items: List[dict]) -> dict:
    score_full = 0
    score_mixed = 0
    score_reserves = 0

    confirmed_absences = []
    doubtful_players = []
    likely_rest_players = []
    probable_lineup = []

    lineup_signals = []
    has_high_priority_source = False

    for item in items:
        title = item["title"]
        summary = item["summary"]
        text = f"{title}. {summary}"
        norm = normalize_text(text)
        weight = item["weight"]

        if item["source_name"] in HIGH_PRIORITY_SOURCES:
            has_high_priority_source = True

        # sinais de provável escalação
        if any(k in norm for k in LIKELY_LINEUP_KEYWORDS):
            lineup_signals.append(title)

        # força máxima
        if any(k in norm for k in FULL_STRENGTH_KEYWORDS):
            score_full += 2 * weight

        # misto / poupados
        if any(k in norm for k in REST_KEYWORDS):
            score_mixed += 2 * weight

            # possíveis nomes citados
            extracted = extract_named_entities_like_players(text)
            likely_rest_players.extend(extracted)

        # ausências
        if any(k in norm for k in ABSENCE_KEYWORDS):
            score_mixed += weight
            extracted = extract_named_entities_like_players(text)
            if extracted:
                confirmed_absences.extend(extracted)
            else:
                confirmed_absences.append(title)

        # dúvidas
        if any(k in norm for k in DOUBT_KEYWORDS):
            extracted = extract_named_entities_like_players(text)
            if extracted:
                doubtful_players.extend(extracted)
            else:
                doubtful_players.append(title)

        # reservas forte
        if "reservas" in norm or "time alternativo" in norm:
            score_reserves += 3 * weight

    confirmed_absences = list(dict.fromkeys(confirmed_absences))[:5]
    doubtful_players = list(dict.fromkeys(doubtful_players))[:5]
    likely_rest_players = list(dict.fromkeys(likely_rest_players))[:5]

    # escalação provável ainda é difícil sem scraper específico;
    # aqui trazemos os melhores sinais encontrados
    if lineup_signals:
        probable_lineup = lineup_signals[:3]
    else:
        probable_lineup = ["Sem escalação estruturada confirmada nas fontes rastreadas"]

    # decisão final
    if score_reserves >= max(score_full, score_mixed) and score_reserves >= 8:
        status = "reservas"
    elif score_mixed >= score_full or likely_rest_players or confirmed_absences:
        status = "misto"
    else:
        status = "forca_maxima"

    # confiança
    if len(items) >= 6 and has_high_priority_source:
        confidence = "alta"
    elif len(items) >= 3:
        confidence = "media"
    else:
        confidence = "baixa"

    return {
        "probable_lineup": probable_lineup,
        "confirmed_absences": confirmed_absences,
        "doubtful_players": doubtful_players,
        "likely_rest_players": likely_rest_players,
        "status": status,
        "confidence": confidence,
        "scores": {
            "full": score_full,
            "mixed": score_mixed,
            "reserves": score_reserves
        }
    }

def build_summary(
    team_name: str,
    opponent_name: Optional[str],
    analysis: dict,
    source_count: int
) -> str:
    status_text = {
        "forca_maxima": "força máxima",
        "misto": "time misto",
        "reservas": "maioria reservas"
    }[analysis["status"]]

    reasons = []
    if analysis["confirmed_absences"]:
        reasons.append("há sinais de desfalques")
    if analysis["likely_rest_players"]:
        reasons.append("existem indícios de preservação de titulares")
    if not reasons and analysis["status"] == "forca_maxima":
        reasons.append("não apareceram sinais fortes de rotação relevante")
    if not reasons:
        reasons.append("as fontes recentes sugerem ajustes na equipe")

    versus = f" contra {opponent_name}" if opponent_name else ""
    return (
        f"O {team_name}{versus} tem tendência de atuar com {status_text}. "
        f"A conclusão foi baseada em {source_count} sinais recentes, com prioridade para fontes esportivas mais confiáveis, "
        f"porque {', '.join(reasons)}."
    )

# =========================
# ENDPOINT
# =========================

@app.post("/match-lineup-analysis", response_model=MatchLineupResponse)
def analyze_match_lineup(payload: MatchLineupRequest):
    queries = build_queries(payload.team_name, payload.opponent_name)

    news_items = []
    for query in queries:
        try:
            news_items.extend(google_news_rss(query, limit=8))
        except Exception:
            continue

    news_items = deduplicate_news(news_items)[:12]

    analysis = analyze_news(news_items)

    sources = [
        SourceItem(
            source_name=item["source_name"],
            source_type=item["source_type"],
            url=item["link"]
        )
        for item in news_items[:8]
    ]

    summary = build_summary(
        team_name=payload.team_name,
        opponent_name=payload.opponent_name,
        analysis=analysis,
        source_count=len(news_items)
    )

    return MatchLineupResponse(
        team_name=payload.team_name,
        opponent_name=payload.opponent_name,
        probable_lineup=analysis["probable_lineup"],
        confirmed_absences=analysis["confirmed_absences"],
        doubtful_players=analysis["doubtful_players"],
        likely_rest_players=analysis["likely_rest_players"],
        status=analysis["status"],
        confidence=analysis["confidence"],
        summary=summary,
        sources=sources
    )