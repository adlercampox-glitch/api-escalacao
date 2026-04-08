from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Tuple
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import requests
import re
from collections import Counter

app = FastAPI(
    title="Futebol Lineup Analysis API",
    version="4.0.0",
    description="API para análise de prováveis escalações com priorização de fontes confiáveis e classificação estrutural do time."
)

# =========================================================
# MODELOS
# =========================================================

class MatchLineupRequest(BaseModel):
    team_name: str = Field(..., description="Nome do time")
    opponent_name: Optional[str] = Field(None, description="Nome do adversário")

class SourceItem(BaseModel):
    source_name: str
    source_type: str
    url: str
    title: Optional[str] = None
    published: Optional[str] = None
    weight: int = 1

class MatchLineupResponse(BaseModel):
    team_name: str
    opponent_name: Optional[str] = None

    probable_lineup: List[str]
    lineup_source_quality: Literal["direta_forte", "direta_media", "indireta", "fraca"]

    confirmed_absences: List[str]
    doubtful_players: List[str]
    likely_rest_players: List[str]
    youth_players: List[str]

    habitual_starters_detected: List[str]
    reserve_players_detected: List[str]

    squad_composition: str
    status: Literal["forca_maxima", "misto", "reservas", "base"]
    team_strength_weight: Literal["muito_forte", "forte", "medio", "fraco", "muito_fraco"]
    confidence: Literal["alta", "media", "baixa"]

    summary: str
    sources: List[SourceItem] = []

# =========================================================
# CONFIGURAÇÕES
# =========================================================

SOURCE_PRIORITY = {
    "ge.globo.com": 10,
    "globoesporte.globo.com": 10,
    "espn.com.br": 8,
    "sportv.globo.com": 8,
    "uol.com.br": 6,
    "lance.com.br": 6,
    "terra.com.br": 4,
    "cnnbrasil.com.br": 3,
}

HIGH_PRIORITY_SOURCES = {
    "ge.globo.com",
    "globoesporte.globo.com",
    "espn.com.br",
    "sportv.globo.com",
}

OFFICIAL_HINTS = [
    "oficial", "site oficial", "clube", ".com.br", ".com"
]

REST_KEYWORDS = [
    "poupado", "poupados", "preservado", "preservados",
    "rodízio", "rodizio", "time misto", "misto", "reservas",
    "poupar", "poupa", "gestão física", "gestao fisica",
    "alternativo", "deve ser preservado", "será preservado", "sera preservado"
]

FULL_STRENGTH_KEYWORDS = [
    "força máxima", "forca maxima", "titulares", "time ideal",
    "escalação ideal", "escalacao ideal", "força total", "forca total",
    "deve repetir a equipe", "mesma equipe", "base titular", "time principal"
]

ABSENCE_KEYWORDS = [
    "desfalque", "desfalques", "lesionado", "lesão", "lesao",
    "suspenso", "fora", "não joga", "nao joga", "afastado",
    "vetado", "não foi relacionado", "nao foi relacionado"
]

DOUBT_KEYWORDS = [
    "dúvida", "duvida", "incerto", "pode ficar fora", "ainda será avaliado",
    "ainda sera avaliado", "em observação", "em observacao",
    "transição", "transicao", "depende", "será reavaliado", "sera reavaliado"
]

LIKELY_LINEUP_PATTERNS = [
    r"provável escalação",
    r"provavel escalacao",
    r"provável time",
    r"provavel time",
    r"time provável",
    r"time provavel",
]

YOUTH_KEYWORDS = [
    "sub-20", "sub20", "sub-17", "sub17", "base", "jovem", "jovens",
    "garoto", "garotos", "cria", "crias", "meninos da base",
    "categoria de base", "equipe alternativa", "garotos da base"
]

RESERVE_KEYWORDS = [
    "reserva", "reservas", "alternativo", "time alternativo", "equipe alternativa"
]

BENCH_KEYWORDS = [
    "banco", "opção no banco", "opcao no banco", "fica no banco", "deve começar no banco",
    "deve comecar no banco"
]

POSITION_WORDS = {
    "goleiro", "lateral", "laterais", "zagueiro", "zagueiros", "volante",
    "volantes", "meia", "meias", "atacante", "atacantes", "centroavante",
    "ponta", "ponta-esquerda", "ponta-direita", "ala", "alas"
}

STOPWORDS_NAMES = {
    "hoje", "amanhã", "amanha", "brasil", "copa", "série", "serie",
    "clube", "time", "jogo", "rodada", "ge", "espn", "sportv",
    "provável", "provavel", "escalação", "escalacao", "titulares",
    "reservas", "base", "oficial", "técnico", "tecnico", "auxiliar",
    "contra", "versus", "x", "vs"
}

REQUEST_TIMEOUT = 8

# =========================================================
# FUNÇÕES AUXILIARES GERAIS
# =========================================================

def clean_html(text: str) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def normalize_text(text: str) -> str:
    return normalize_spaces(text).lower()

def source_weight(link: str) -> int:
    link = (link or "").lower()
    for domain, weight in SOURCE_PRIORITY.items():
        if domain in link:
            return weight
    return 2

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

    if any(h in link for h in ["fc", "clube", "oficial"]):
        return "oficial"

    return "outra"

def fetch_url_text(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)
        return normalize_spaces(text)
    except Exception:
        return ""

def deduplicate_news(items: List[dict]) -> List[dict]:
    seen = set()
    result = []

    for item in sorted(items, key=lambda x: (x["weight"], x.get("published", "")), reverse=True):
        key = (
            normalize_text(item.get("title", "")),
            normalize_text(item.get("link", ""))
        )
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result

def build_queries(team_name: str, opponent_name: Optional[str]) -> List[str]:
    base = f'"{team_name}"'
    queries = []

    if opponent_name:
        pair = f'"{team_name}" "{opponent_name}"'
        queries.extend([
            f'{pair} provável escalação',
            f'{pair} provável time',
            f'{pair} poupados desfalques',
            f'{pair} titulares reservas',
            f'{pair} base sub-20 time misto',
            f'{pair} ge provável escalação',
            f'{pair} ESPN provável time',
        ])
    else:
        queries.extend([
            f'{base} provável escalação',
            f'{base} provável time',
            f'{base} poupados desfalques',
            f'{base} titulares reservas',
            f'{base} base sub-20 time misto',
            f'{base} ge provável escalação',
            f'{base} ESPN provável time',
        ])

    return queries

def google_news_rss(query: str, limit: int = 8) -> List[dict]:
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    )
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

# =========================================================
# EXTRAÇÃO DE NOMES
# =========================================================

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

    for i, tok in enumerate(tokens):
        tok_clean = tok.strip(".,:;!?()[]{}\"'")
        if looks_like_player_name(tok_clean):
            found.append(tok_clean)

            if i + 1 < len(tokens):
                nxt = tokens[i + 1].strip(".,:;!?()[]{}\"'")
                if looks_like_player_name(nxt):
                    found.append(f"{tok_clean} {nxt}")

    cleaned = []
    for name in found:
        name = normalize_spaces(name)
        if not name:
            continue

        low = name.lower()
        if low in STOPWORDS_NAMES or low in POSITION_WORDS:
            continue

        # evita frases grandes e ruído
        if len(name.split()) > 3:
            continue

        cleaned.append(name)

    return list(dict.fromkeys(cleaned))

def split_lineup_names(raw: str) -> List[str]:
    raw = raw.strip(" .:-")
    raw = re.sub(r"\([^)]*\)", "", raw)
    raw = raw.replace(";", ",")
    raw = raw.replace(" e ", ", ")
    parts = [p.strip(" .:-") for p in raw.split(",")]

    names = []
    for part in parts:
        if not part:
            continue

        # remove posições escritas antes do nome
        part = re.sub(
            r"^(goleiro|lateral(?: direito| esquerdo)?|zagueiro|volante|meia|atacante|centroavante|ponta|ala)\s+",
            "",
            part,
            flags=re.IGNORECASE
        )

        if len(part.split()) <= 4 and any(ch.isalpha() for ch in part):
            names.append(normalize_spaces(part))

    # filtra itens ruins
    final = []
    for n in names:
        low = n.lower()
        if low in STOPWORDS_NAMES:
            continue
        if len(n) < 2:
            continue
        final.append(n)

    return list(dict.fromkeys(final))

# =========================================================
# EXTRAÇÃO DE ESCALAÇÃO DIRETA
# =========================================================

def try_extract_lineup_from_text(team_name: str, text: str) -> List[str]:
    if not text:
        return []

    patterns = [
        r"(?:provável escalação|provavel escalacao|provável time|provavel time|time provável|time provavel)\s*[:\-]\s*(.+?)(?:\.\s|desfalques|dúvidas|duvidas|quem está fora|tecnico|técnico|$)",
        rf"{re.escape(team_name)}\s*[:\-]\s*(.+?)(?:\.\s|desfalques|dúvidas|duvidas|tecnico|técnico|$)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        for match in matches:
            lineup = split_lineup_names(match)
            if 8 <= len(lineup) <= 15:
                return lineup[:11]

    return []

def choose_best_direct_lineup(team_name: str, items: List[dict]) -> Tuple[List[str], str, List[dict]]:
    """
    Tenta extrair a provável escalação diretamente das fontes.
    Retorna: lineup, quality, supporting_items
    """
    candidates = []

    for item in items:
        combined = f"{item.get('title', '')}. {item.get('summary', '')}"
        from_snippet = try_extract_lineup_from_text(team_name, combined)

        if from_snippet:
            score = item["weight"] + 3
            if item["source_name"] in HIGH_PRIORITY_SOURCES:
                score += 4

            candidates.append({
                "lineup": from_snippet,
                "score": score,
                "quality": "direta_media" if item["source_name"] in HIGH_PRIORITY_SOURCES else "indireta",
                "item": item
            })

        page_text = fetch_url_text(item.get("link", ""))
        from_page = try_extract_lineup_from_text(team_name, page_text)

        if from_page:
            score = item["weight"] + 8
            if item["source_name"] in HIGH_PRIORITY_SOURCES:
                score += 6

            quality = "direta_forte" if item["source_name"] in HIGH_PRIORITY_SOURCES else "direta_media"

            candidates.append({
                "lineup": from_page,
                "score": score,
                "quality": quality,
                "item": item,
                "page_text": page_text
            })

    if not candidates:
        return [], "fraca", []

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    return best["lineup"], best["quality"], [best["item"]]

# =========================================================
# EXTRAÇÃO DE CONTEXTO
# =========================================================

def extract_names_near_keywords(text: str, keywords: List[str], window: int = 180) -> List[str]:
    text_clean = normalize_spaces(text)
    low = text_clean.lower()

    collected = []

    for kw in keywords:
        start = 0
        while True:
            idx = low.find(kw.lower(), start)
            if idx == -1:
                break

            snippet = text_clean[max(0, idx - 40): idx + window]
            collected.extend(extract_player_candidates(snippet))
            start = idx + len(kw)

    return list(dict.fromkeys(collected))

def extract_context_signals(items: List[dict]) -> dict:
    confirmed_absences = []
    doubtful_players = []
    likely_rest_players = []
    youth_players = []
    reserve_players = []

    score_full = 0
    score_mixed = 0
    score_reserves = 0
    score_base = 0

    has_high_priority_source = False

    for item in items:
        base_text = f"{item.get('title', '')}. {item.get('summary', '')}"
        page_text = fetch_url_text(item.get("link", ""))
        text = f"{base_text}. {page_text}"
        norm = normalize_text(text)
        weight = item["weight"]

        if item["source_name"] in HIGH_PRIORITY_SOURCES:
            has_high_priority_source = True

        if any(k in norm for k in FULL_STRENGTH_KEYWORDS):
            score_full += 3 * weight

        if any(k in norm for k in REST_KEYWORDS):
            score_mixed += 3 * weight
            likely_rest_players.extend(extract_names_near_keywords(text, REST_KEYWORDS))

        if any(k in norm for k in ABSENCE_KEYWORDS):
            score_mixed += 1 * weight
            confirmed_absences.extend(extract_names_near_keywords(text, ABSENCE_KEYWORDS))

        if any(k in norm for k in DOUBT_KEYWORDS):
            doubtful_players.extend(extract_names_near_keywords(text, DOUBT_KEYWORDS))

        if any(k in norm for k in RESERVE_KEYWORDS):
            score_reserves += 3 * weight
            reserve_players.extend(extract_names_near_keywords(text, RESERVE_KEYWORDS + BENCH_KEYWORDS))

        if any(k in norm for k in YOUTH_KEYWORDS):
            score_base += 4 * weight
            youth_players.extend(extract_names_near_keywords(text, YOUTH_KEYWORDS))

    return {
        "confirmed_absences": list(dict.fromkeys(confirmed_absences))[:10],
        "doubtful_players": list(dict.fromkeys(doubtful_players))[:10],
        "likely_rest_players": list(dict.fromkeys(likely_rest_players))[:10],
        "youth_players": list(dict.fromkeys(youth_players))[:10],
        "reserve_players_detected": list(dict.fromkeys(reserve_players))[:10],
        "score_full": score_full,
        "score_mixed": score_mixed,
        "score_reserves": score_reserves,
        "score_base": score_base,
        "has_high_priority_source": has_high_priority_source
    }

# =========================================================
# CLASSIFICAÇÃO DO TIME
# =========================================================

def infer_habitual_starters(lineup: List[str], context: dict) -> List[str]:
    blocked = {x.lower() for x in context["likely_rest_players"] + context["confirmed_absences"] + context["youth_players"]}
    starters = []

    for player in lineup:
        if player.lower() not in blocked:
            starters.append(player)

    return starters[:11]

def classify_team_status(
    lineup: List[str],
    lineup_source_quality: str,
    context: dict
) -> Tuple[str, str, List[str], List[str]]:
    """
    Retorna:
    status, squad_composition, habitual_starters_detected, reserve_players_detected
    """

    youth_in_lineup = [p for p in lineup if p in context["youth_players"]]
    rested_in_lineup = [p for p in lineup if p in context["likely_rest_players"]]
    starters_detected = infer_habitual_starters(lineup, context)
    reserve_detected = [p for p in lineup if p in context["reserve_players_detected"]]

    lineup_count = len(lineup)
    youth_ratio = len(youth_in_lineup) / lineup_count if lineup_count else 0
    reserve_ratio = len(reserve_detected) / lineup_count if lineup_count else 0
    starter_ratio = len(starters_detected) / lineup_count if lineup_count else 0

    score_full = context["score_full"]
    score_mixed = context["score_mixed"]
    score_reserves = context["score_reserves"]
    score_base = context["score_base"]

    # Regras mais estruturadas
    if youth_ratio >= 0.45 or score_base >= max(score_full, score_mixed, score_reserves) + 6:
        status = "base"
        squad = "equipe com presença forte de jogadores da base / sub-20"

    elif reserve_ratio >= 0.45 or score_reserves >= max(score_full, score_mixed) + 6:
        status = "reservas"
        squad = "maioria de reservas e rotação forte"

    elif starter_ratio >= 0.72 and not context["likely_rest_players"] and score_full >= score_mixed:
        status = "forca_maxima"
        squad = "predominância clara de titulares habituais"

    else:
        status = "misto"
        if youth_ratio >= 0.25:
            squad = "time misto com titulares, reservas e uso de jogadores da base"
        elif context["likely_rest_players"]:
            squad = "time misto com preservação de alguns titulares"
        else:
            squad = "time misto entre titulares habituais e peças de rotação"

    # ajuste fino com fonte direta forte
    if lineup_source_quality == "direta_forte" and lineup_count >= 10:
        if starter_ratio >= 0.8 and status in ["misto", "reservas"]:
            status = "forca_maxima"
            squad = "predominância clara de titulares habituais"
        elif youth_ratio >= 0.4 and status != "base":
            status = "base"
            squad = "equipe com presença forte de jogadores da base / sub-20"

    return status, squad, starters_detected, reserve_detected

def classify_team_strength_weight(
    status: str,
    lineup: List[str],
    habitual_starters_detected: List[str],
    reserve_players_detected: List[str],
    youth_players: List[str]
) -> str:
    if not lineup:
        return "medio"

    total = len(lineup)
    starter_ratio = len(habitual_starters_detected) / total if total else 0
    reserve_ratio = len([p for p in lineup if p in reserve_players_detected]) / total if total else 0
    youth_ratio = len([p for p in lineup if p in youth_players]) / total if total else 0

    if status == "forca_maxima":
        if starter_ratio >= 0.82:
            return "muito_forte"
        return "forte"

    if status == "misto":
        if starter_ratio >= 0.60:
            return "forte"
        if youth_ratio >= 0.25 or reserve_ratio >= 0.30:
            return "medio"
        return "medio"

    if status == "reservas":
        if reserve_ratio >= 0.55:
            return "fraco"
        return "medio"

    # status == "base"
    if youth_ratio >= 0.50:
        return "muito_fraco"
    return "fraco"

def classify_confidence(
    items: List[dict],
    lineup_source_quality: str,
    context: dict,
    lineup: List[str]
) -> str:
    if lineup_source_quality == "direta_forte" and len(lineup) >= 10:
        return "alta"

    if len(items) >= 5 and context["has_high_priority_source"]:
        return "media"

    if len(items) >= 3:
        return "media"

    return "baixa"

# =========================================================
# FALLBACK HEURÍSTICO
# =========================================================

def build_fallback_lineup(items: List[dict], context: dict) -> List[str]:
    """
    Só usa isso se não houver escalação direta extraída.
    """
    player_score: Dict[str, int] = Counter()

    blocked = {
        x.lower() for x in (
            context["confirmed_absences"] +
            context["likely_rest_players"]
        )
    }

    for item in items:
        text = f"{item.get('title', '')}. {item.get('summary', '')}"
        page_text = fetch_url_text(item.get("link", ""))
        full_text = f"{text}. {page_text}"

        names = extract_player_candidates(full_text)

        for name in names:
            if name.lower() in blocked:
                continue

            pts = item["weight"]
            if item["source_name"] in HIGH_PRIORITY_SOURCES:
                pts += 2

            player_score[name] += pts

    probable = [name for name, _ in player_score.most_common(11)]
    return probable

# =========================================================
# SUMÁRIO
# =========================================================

def status_to_text(status: str) -> str:
    mapping = {
        "forca_maxima": "força máxima",
        "misto": "time misto",
        "reservas": "maioria reservas",
        "base": "equipe com forte uso da base"
    }
    return mapping[status]

def strength_to_text(weight: str) -> str:
    mapping = {
        "muito_forte": "muito forte",
        "forte": "forte",
        "medio": "médio",
        "fraco": "fraco",
        "muito_fraco": "muito fraco"
    }
    return mapping[weight]

def build_summary(
    team_name: str,
    opponent_name: Optional[str],
    status: str,
    squad_composition: str,
    weight: str,
    confidence: str,
    lineup_source_quality: str,
    context: dict,
    lineup: List[str]
) -> str:
    versus = f" contra {opponent_name}" if opponent_name else ""
    reasons = []

    if lineup_source_quality in ["direta_forte", "direta_media"]:
        reasons.append("há extração direta de provável escalação em fonte relevante")
    if context["confirmed_absences"]:
        reasons.append("existem desfalques detectados")
    if context["likely_rest_players"]:
        reasons.append("há sinais de preservação de titulares")
    if context["youth_players"]:
        reasons.append("aparecem indícios de utilização da base")
    if not reasons:
        reasons.append("o conjunto de fontes sugere manutenção da estrutura recente")

    return (
        f"O {team_name}{versus} tem tendência de atuar com {status_to_text(status)}. "
        f"A composição esperada indica {squad_composition}. "
        f"O peso competitivo projetado para a equipe é {strength_to_text(weight)} "
        f"em relação ao padrão recente do time. "
        f"A confiança da leitura é {confidence}, porque {', '.join(reasons)}."
    )

# =========================================================
# ENDPOINT
# =========================================================

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

    # 1) tenta extrair a escalação diretamente da matéria
    direct_lineup, lineup_source_quality, direct_support = choose_best_direct_lineup(
        payload.team_name,
        news_items
    )

    # 2) extrai contexto global
    context = extract_context_signals(news_items)

    # 3) fallback só se não tiver escalação direta
    probable_lineup = direct_lineup
    if not probable_lineup:
        probable_lineup = build_fallback_lineup(news_items, context)
        lineup_source_quality = "indireta" if probable_lineup else "fraca"

    # 4) se ainda assim não tiver nada, devolve frase padrão
    if not probable_lineup:
        probable_lineup = ["Sem escalação nominal consistente nas fontes rastreadas"]

    # 5) classifica status
    status, squad_composition, habitual_starters_detected, reserve_players_detected = classify_team_status(
        probable_lineup if probable_lineup[0] != "Sem escalação nominal consistente nas fontes rastreadas" else [],
        lineup_source_quality,
        context
    )

    # 6) classifica peso competitivo
    team_strength_weight = classify_team_strength_weight(
        status,
        probable_lineup if probable_lineup[0] != "Sem escalação nominal consistente nas fontes rastreadas" else [],
        habitual_starters_detected,
        reserve_players_detected,
        context["youth_players"]
    )

    # 7) confiança
    confidence = classify_confidence(
        news_items,
        lineup_source_quality,
        context,
        probable_lineup if probable_lineup[0] != "Sem escalação nominal consistente nas fontes rastreadas" else []
    )

    # 8) fontes
    sources = [
        SourceItem(
            source_name=item["source_name"],
            source_type=item["source_type"],
            url=item["link"],
            title=item.get("title"),
            published=item.get("published"),
            weight=item["weight"]
        )
        for item in news_items[:8]
    ]

    # 9) sumário
    summary = build_summary(
        team_name=payload.team_name,
        opponent_name=payload.opponent_name,
        status=status,
        squad_composition=squad_composition,
        weight=team_strength_weight,
        confidence=confidence,
        lineup_source_quality=lineup_source_quality,
        context=context,
        lineup=probable_lineup if probable_lineup[0] != "Sem escalação nominal consistente nas fontes rastreadas" else []
    )

    return MatchLineupResponse(
        team_name=payload.team_name,
        opponent_name=payload.opponent_name,
        probable_lineup=probable_lineup,
        lineup_source_quality=lineup_source_quality,

        confirmed_absences=context["confirmed_absences"],
        doubtful_players=context["doubtful_players"],
        likely_rest_players=context["likely_rest_players"],
        youth_players=context["youth_players"],

        habitual_starters_detected=habitual_starters_detected,
        reserve_players_detected=reserve_players_detected,

        squad_composition=squad_composition,
        status=status,
        team_strength_weight=team_strength_weight,
        confidence=confidence,

        summary=summary,
        sources=sources
    )