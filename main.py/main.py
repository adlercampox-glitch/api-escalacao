from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Tuple
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import requests
import re
from collections import Counter, defaultdict

app = FastAPI(
    title="Futebol Lineup Analysis API",
    version="4.1.0",
    description="API para análise de prováveis escalações com cruzamento de fontes confiáveis."
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
    "ge.globo.com": 12,
    "globoesporte.globo.com": 12,
    "espn.com.br": 10,
    "sportv.globo.com": 10,
    "uol.com.br": 7,
    "lance.com.br": 7,
    "terra.com.br": 4,
    "cnnbrasil.com.br": 3,
}

HIGH_PRIORITY_SOURCES = {
    "ge.globo.com",
    "globoesporte.globo.com",
    "espn.com.br",
    "sportv.globo.com",
}

REST_KEYWORDS = [
    "poupado", "poupados", "preservado", "preservados",
    "rodízio", "rodizio", "time misto", "misto", "reservas",
    "poupar", "poupa", "gestão física", "gestao fisica",
    "alternativo", "será preservado", "sera preservado"
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
    "transição", "transicao", "será reavaliado", "sera reavaliado"
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
    "categoria de base", "garotos da base"
]

RESERVE_KEYWORDS = [
    "reserva", "reservas", "alternativo", "time alternativo", "equipe alternativa"
]

BENCH_KEYWORDS = [
    "banco", "fica no banco", "deve começar no banco", "deve comecar no banco"
]

POSITION_WORDS = {
    "goleiro", "goleiros", "lateral", "laterais", "zagueiro", "zagueiros",
    "volante", "volantes", "meia", "meias", "atacante", "atacantes",
    "centroavante", "ponta", "ala", "alas"
}

STOPWORDS_NAMES = {
    "hoje", "amanhã", "amanha", "brasil", "copa", "série", "serie",
    "clube", "time", "jogo", "rodada", "ge", "espn", "sportv",
    "provável", "provavel", "escalação", "escalacao", "titulares",
    "reservas", "base", "oficial", "técnico", "tecnico", "auxiliar",
    "contra", "versus", "x", "vs", "partida", "confronto", "elenco",
    "comissão", "comissao", "treinador", "amazonas", "remo", "primavera",
    "vila", "nova", "barcelona", "real", "madrid", "flamengo", "palmeiras"
}

REQUEST_TIMEOUT = 8

# =========================================================
# FUNÇÕES AUXILIARES
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
    return "outra"

def fetch_url_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return normalize_spaces(soup.get_text(" ", strip=True))
    except Exception:
        return ""

def deduplicate_news(items: List[dict]) -> List[dict]:
    seen = set()
    result = []
    for item in sorted(items, key=lambda x: (x["weight"], x.get("published", "")), reverse=True):
        key = (normalize_text(item.get("title", "")), normalize_text(item.get("link", "")))
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
            f'{pair} ge provável escalação',
            f'{pair} ESPN provável time',
            f'{pair} sportv provável escalação',
            f'{pair} desfalques dúvidas',
            f'{pair} poupados titulares reservas',
            f'{pair} base sub-20 time misto',
        ])
    else:
        queries.extend([
            f'{base} provável escalação',
            f'{base} provável time',
            f'{base} ge provável escalação',
            f'{base} ESPN provável time',
            f'{base} sportv provável escalação',
            f'{base} desfalques dúvidas',
            f'{base} poupados titulares reservas',
            f'{base} base sub-20 time misto',
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
        items.append({
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "summary": clean_html(entry.get("summary", "")),
            "published": entry.get("published", ""),
            "weight": source_weight(entry.get("link", "")),
            "source_name": source_name_from_link(entry.get("link", "")),
            "source_type": source_type_from_link(entry.get("link", "")),
        })
    return items

# =========================================================
# EXTRAÇÃO DE NOMES
# =========================================================

def looks_like_player_name(token: str) -> bool:
    token = token.strip(".,:;!?()[]{}\"'")
    if len(token) < 3:
        return False
    low = token.lower()
    if low in STOPWORDS_NAMES or low in POSITION_WORDS:
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
        if len(name.split()) > 3:
            continue
        low = name.lower()
        if low in STOPWORDS_NAMES or low in POSITION_WORDS:
            continue
        cleaned.append(name)

    return list(dict.fromkeys(cleaned))

def split_lineup_names(raw: str) -> List[str]:
    raw = raw.strip(" .:-")
    raw = re.sub(r"\([^)]*\)", "", raw)
    raw = raw.replace(";", ",").replace(" e ", ", ")
    parts = [p.strip(" .:-") for p in raw.split(",")]

    names = []
    for part in parts:
        if not part:
            continue
        part = re.sub(
            r"^(goleiro|lateral(?: direito| esquerdo)?|zagueiro|volante|meia|atacante|centroavante|ponta|ala)\s+",
            "",
            part,
            flags=re.IGNORECASE
        )
        if len(part.split()) <= 4 and any(ch.isalpha() for ch in part):
            names.append(normalize_spaces(part))

    final = []
    for n in names:
        low = n.lower()
        if low in STOPWORDS_NAMES or len(n) < 2:
            continue
        final.append(n)

    return list(dict.fromkeys(final))

# =========================================================
# ESCALAÇÃO DIRETA
# =========================================================

def try_extract_lineup_from_text(team_name: str, text: str) -> List[str]:
    if not text:
        return []

    patterns = [
        r"(?:provável escalação|provavel escalacao|provável time|provavel time|time provável|time provavel)\s*[:\-]\s*(.+?)(?:\.\s|desfalques|dúvidas|duvidas|quem está fora|quem esta fora|técnico|tecnico|$)",
        rf"{re.escape(team_name)}\s*[:\-]\s*(.+?)(?:\.\s|desfalques|dúvidas|duvidas|quem está fora|quem esta fora|técnico|tecnico|$)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        for match in matches:
            lineup = split_lineup_names(match)
            valid_names = [
                n for n in lineup
                if len(n.split()) <= 3 and n.lower() not in STOPWORDS_NAMES
            ]
            if 10 <= len(valid_names) <= 11:
                return valid_names[:11]
    return []

def choose_best_direct_lineup(team_name: str, items: List[dict]) -> Tuple[List[str], str, List[dict]]:
    candidates = []

    for item in items:
        combined = f"{item.get('title', '')}. {item.get('summary', '')}"
        from_snippet = try_extract_lineup_from_text(team_name, combined)
        if from_snippet:
            score = item["weight"] + 4
            if item["source_name"] in HIGH_PRIORITY_SOURCES:
                score += 5
            candidates.append({
                "lineup": from_snippet,
                "score": score,
                "quality": "direta_media" if item["source_name"] in HIGH_PRIORITY_SOURCES else "indireta",
                "item": item
            })

        page_text = fetch_url_text(item.get("link", ""))
        from_page = try_extract_lineup_from_text(team_name, page_text)
        if from_page:
            score = item["weight"] + 10
            if item["source_name"] in HIGH_PRIORITY_SOURCES:
                score += 8
            candidates.append({
                "lineup": from_page,
                "score": score,
                "quality": "direta_forte" if item["source_name"] in HIGH_PRIORITY_SOURCES else "direta_media",
                "item": item
            })

    if not candidates:
        return [], "fraca", []

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    return best["lineup"], best["quality"], [best["item"]]

# =========================================================
# CONTEXTO
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

    player_penalties = defaultdict(int)
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
            names = extract_names_near_keywords(text, REST_KEYWORDS)
            likely_rest_players.extend(names)
            for n in names:
                player_penalties[n.lower()] += 6

        if any(k in norm for k in ABSENCE_KEYWORDS):
            score_mixed += weight
            names = extract_names_near_keywords(text, ABSENCE_KEYWORDS)
            confirmed_absences.extend(names)
            for n in names:
                player_penalties[n.lower()] += 8

        if any(k in norm for k in DOUBT_KEYWORDS):
            names = extract_names_near_keywords(text, DOUBT_KEYWORDS)
            doubtful_players.extend(names)
            for n in names:
                player_penalties[n.lower()] += 4

        if any(k in norm for k in RESERVE_KEYWORDS):
            score_reserves += 3 * weight
            reserve_players.extend(extract_names_near_keywords(text, RESERVE_KEYWORDS + BENCH_KEY