from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import re

app = FastAPI(
    title="Futebol Lineup Analysis API",
    version="3.0.0",
    description="API para análise de prováveis escalações com cruzamento de fontes confiáveis."
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
    youth_players: List[str]
    squad_composition: str
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
    "sportv.globo.com": 4,
    "uol.com.br": 3,
    "lance.com.br": 3,
    "cnnbrasil.com.br": 2,
}

HIGH_PRIORITY_SOURCES = {
    "ge.globo.com",
    "globoesporte.globo.com",
    "espn.com.br",
    "sportv.globo.com"
}

REST_KEYWORDS = [
    "poupado", "poupados", "preservado", "preservados",
    "rodízio", "rodizio", "time misto", "misto", "reservas",
    "poupar", "poupa", "gestão física", "gestao fisica"
]

FULL_STRENGTH_KEYWORDS = [
    "força máxima", "forca maxima", "titulares", "time ideal",
    "escalação ideal", "escalacao ideal", "força total", "forca total"
]

ABSENCE_KEYWORDS = [
    "desfalque", "desfalques", "lesionado", "lesão", "lesao",
    "suspenso", "fora", "não joga", "nao joga", "afastado"
]

DOUBT_KEYWORDS = [
    "dúvida", "duvida", "incerto", "pode ficar fora", "ainda será avaliado",
    "ainda sera avaliado", "em observação", "em observacao",
    "transição", "transicao"
]

LIKELY_LINEUP_KEYWORDS = [
    "provável escalação", "provavel escalação", "provavel escalacao",
    "escalação provável", "escalacao provavel"
]

YOUTH_KEYWORDS = [
    "sub-20", "sub20", "base", "jovem", "jovens", "garoto", "garotos",
    "cria", "crias", "meninos da base", "equipe alternativa"
]

POSITION_WORDS = {
    "goleiro", "lateral", "zagueiro", "volante", "meia", "atacante",
    "centroavante", "ponta", "ala"
}

STOPWORDS_NAMES = {
    "hoje", "amanhã", "amanha", "brasil", "copa", "série", "serie",
    "clube", "time", "jogo", "rodada", "ge", "espn", "sportv",
    "provável", "provavel", "escalação", "escalacao", "titulares",
    "reservas", "base", "oficial", "técnico", "tecnico", "amazonas",
    "remo", "flamengo", "palmeiras", "barcelona", "real", "madrid",
    "vila", "nova", "primavera"
}

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
    link = (link or "").lower()
    for domain, weight in SOURCE_PRIORITY.items():
        if domain in link:
            return weight
    return 1

def source_name_from_link(link: str) -> str:
    link = (link or "").lower()
    for domain in SOURCE_PRIORITY.keys():
        if domain in link:
            return domain
    return "outra_fonte"

def source_type_from_link(link: str) -> str:
    link = (link or "").lower()
    if any(domain in link for domain in HIGH_PRIORITY_SOURCES):
        return "midia_esportiva"
    if "oficial" in link:
        return "oficial"
    return "outra"

def build_queries(team_name: str, opponent_name: Optional[str]) -> List[str]:
    queries = []

    base = f'"{team_name}"'
    if opponent_name:
        versus = f'"{team_name}" "{opponent_name}"'
        queries.append(f"{versus} provável escalação")
        queries.append(f"{versus} poupados desfalques")
        queries.append(f"{versus} titulares reservas")
        queries.append(f"{versus} time misto base sub-20")
        queries.append(f"{versus} ge provável escalação")
        queries.append(f"{versus} ESPN desfalques")
        queries.append(f"{versus} sportv provável escalação")
    else:
        queries.append(f'{base} provável escalação')
        queries.append(f'{base} poupados desfalques')
        queries.append(f'{base} titulares reservas')
        queries.append(f'{base} time misto base sub-20')
        queries.append(f'{base} ge provável escalação')
        queries.append(f'{base} ESPN desfalques')
        queries.append(f'{base} sportv provável escalação')

    return queries

def google_news_rss(query: str, limit: int = 8) -> List[dict]:
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

def looks_like_player_name(token: str) -> bool:
    token = token.strip(".,:;!?()[]{}\"'")
    if len(token) < 3:
        return False
    if token.lower() in STOPWORDS_NAMES:
        return False
    if token.lower() in POSITION_WORDS:
        return False
    return token[:1].isupper()

def extract_player_candidates(text: str) -> List[str]:
    text = clean_html(text)
    tokens = re.split(r"\s+", text)
    found = []

    # nomes isolados e compostos
    for i, tok in enumerate(tokens):
        tok_clean = tok.strip(".,:;!?()[]{}\"'")
        if looks_like_player_name(tok_clean):
            found.append(tok_clean)

            if i + 1 < len(tokens):
                nxt = tokens[i + 1].strip(".,:;!?()[]{}\"'")
                if looks_like_player_name(nxt):
                    full_name = f"{tok_clean} {nxt}"
                    found.append(full_name)

    # gatilhos específicos
    trigger_patterns = [
        r"(?:sem|poupa|poupar|dúvida em|duvida em|volta de|retorno de)\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç\-]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç\-]+)?)"
    ]

    for pattern in trigger_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for m in matches:
            found.append(m.strip())

    cleaned = []
    for name in found:
        name = name.strip()
        if not name:
            continue
        low = name.lower()
        if low in STOPWORDS_NAMES or low in POSITION_WORDS:
            continue
        cleaned.append(name)

    return list(dict.fromkeys(cleaned))

# =========================
# ANÁLISE
# =========================

def analyze_news(items: List[dict]) -> dict:
    score_full = 0
    score_mixed = 0
    score_reserves = 0

    confirmed_absences = []
    doubtful_players = []
    likely_rest_players = []
    youth_players = []

    player_score = {}
    has_high_priority_source = False
    youth_signal_count = 0

    for item in items:
        title = item["title"]
        summary = item["summary"]
        text = f"{title}. {summary}"
        norm = normalize_text(text)
        weight = item["weight"]

        if item["source_name"] in HIGH_PRIORITY_SOURCES:
            has_high_priority_source = True

        players_found = extract_player_candidates(text)

        # Sinal forte de provável escalação
        if any(k in norm for k in LIKELY_LINEUP_KEYWORDS):
            for p in players_found:
                player_score[p] = player_score.get(p, 0) + (3 * weight)

        # força máxima
        if any(k in norm for k in FULL_STRENGTH_KEYWORDS):
            score_full += 2 * weight
            for p in players_found:
                player_score[p] = player_score.get(p, 0) + weight

        # misto / poupados
        if any(k in norm for k in REST_KEYWORDS):
            score_mixed += 2 * weight
            extracted = extract_player_candidates(text)
            likely_rest_players.extend(extracted)

        # ausências
        if any(k in norm for k in ABSENCE_KEYWORDS):
            score_mixed += weight
            extracted = extract_player_candidates(text)
            if extracted:
                confirmed_absences.extend(extracted)
            else:
                confirmed_absences.append(title)

        # dúvidas
        if any(k in norm for k in DOUBT_KEYWORDS):
            extracted = extract_player_candidates(text)
            if extracted:
                doubtful_players.extend(extracted)
            else:
                doubtful_players.append(title)

        # reservas forte
        if "reservas" in norm or "time alternativo" in norm:
            score_reserves += 3 * weight

        # base / sub-20
        if any(k in norm for k in YOUTH_KEYWORDS):
            youth_signal_count += 1
            extracted = extract_player_candidates(text)
            if extracted:
                youth_players.extend(extracted)

        # reforça score geral dos jogadores citados
        for p in players_found:
            player_score[p] = player_score.get(p, 0) + weight

    confirmed_absences = list(dict.fromkeys(confirmed_absences))[:6]
    doubtful_players = list(dict.fromkeys(doubtful_players))[:6]
    likely_rest_players = list(dict.fromkeys(likely_rest_players))[:6]
    youth_players = list(dict.fromkeys(youth_players))[:6]

    blocked = {x.lower() for x in confirmed_absences + likely_rest_players}
    ranked_players = sorted(player_score.items(), key=lambda x: x[1], reverse=True)

    probable_lineup = []
    for name, _pts in ranked_players:
        if name.lower() not in blocked:
            probable_lineup.append(name)
        if len(probable_lineup) >= 11:
            break

    if not probable_lineup:
        probable_lineup = ["Sem escalação nominal consistente nas fontes rastreadas"]

    # decisão final
    if score_reserves >= max(score_full, score_mixed) and score_reserves >= 8:
        status = "reservas"
    elif score_mixed >= score_full or likely_rest_players or confirmed_absences:
        status = "misto"
    else:
        status = "forca_maxima"

    # composição do elenco
    if status == "forca_maxima":
        squad_composition = "predominância de titulares"
    elif youth_signal_count >= 2 and likely_rest_players:
        squad_composition = "time misto com reservas e uso de jogadores da base"
    elif youth_signal_count >= 2:
        squad_composition = "equipe com presença relevante de jogadores da base"
    elif likely_rest_players:
        squad_composition = "time misto com titulares e reservas"
    elif status == "reservas":
        squad_composition = "maioria reservas"
    else:
        squad_composition = "misto"

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
        "youth_players": youth_players,
        "squad_composition": squad_composition,
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
    if analysis["youth_players"]:
        reasons.append("aparecem sinais de utilização de jogadores da base")
    if not reasons and analysis["status"] == "forca_maxima":
        reasons.append("não apareceram sinais fortes de rotação relevante")
    if not reasons:
        reasons.append("as fontes recentes sugerem ajustes na equipe")

    versus = f" contra {opponent_name}" if opponent_name else ""
    return (
        f"O {team_name}{versus} tem tendência de atuar com {status_text}. "
        f"A composição mais provável é: {analysis['squad_composition']}. "
        f"A conclusão foi baseada em {source_count} sinais recentes, com prioridade para fontes esportivas confiáveis, "
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

    news_items = deduplicate_news(news_items)[:15]

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
        youth_players=analysis["youth_players"],
        squad_composition=analysis["squad_composition"],
        status=analysis["status"],
        confidence=analysis["confidence"],
        summary=summary,
        sources=sources
    )