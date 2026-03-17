"""
EXOFIT - Chargeurs de donnees reelles.
Parse des cas cliniques depuis des fichiers JSON, CSV, TXT ou DOCX.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


PATIENT_FIELD_ALIASES = {
    "age": "age",
    "sexe": "sexe",
    "sex": "sexe",
    "genre": "sexe",
    "taille": "taille_cm",
    "taille_cm": "taille_cm",
    "height_cm": "taille_cm",
    "poids": "poids_kg",
    "poids_kg": "poids_kg",
    "weight_kg": "poids_kg",
    "symptomes": "symptomes",
    "symptomes_codes": "symptomes",
    "symptoms": "symptomes",
    "symptome_libre": "symptome_libre",
    "description": "symptome_libre",
    "plainte_principale": "symptome_libre",
    "histoire": "symptome_libre",
    "tension_systolique": "tension_systolique",
    "ta_sys": "tension_systolique",
    "systolic_bp": "tension_systolique",
    "tension_diastolique": "tension_diastolique",
    "ta_dia": "tension_diastolique",
    "diastolic_bp": "tension_diastolique",
    "frequence_cardiaque": "frequence_cardiaque",
    "fc": "frequence_cardiaque",
    "heart_rate": "frequence_cardiaque",
    "temperature": "temperature",
    "temp": "temperature",
    "spo2": "spo2",
    "sao2": "spo2",
    "antecedents": "antecedents",
    "history": "antecedents",
    "traitements_en_cours": "traitements_en_cours",
    "traitements": "traitements_en_cours",
    "medications": "traitements_en_cours",
    "examens_realises": "examens_realises",
    "examens": "examens_realises",
    "tests_done": "examens_realises",
    "resultats_examens": "resultats_examens",
    "resultats": "resultats_examens",
    "exam_results": "resultats_examens",
}

LIST_FIELDS = {"symptomes", "antecedents", "traitements_en_cours", "examens_realises"}
NUMERIC_INT_FIELDS = {
    "age",
    "taille_cm",
    "tension_systolique",
    "tension_diastolique",
    "frequence_cardiaque",
    "spo2",
}
NUMERIC_FLOAT_FIELDS = {"poids_kg", "temperature"}


def load_patient_file(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """Parse un fichier patient reel dans un format supporte."""
    suffix = Path(filename).suffix.lower()
    text = file_bytes.decode("utf-8-sig", errors="replace")

    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            if not payload:
                raise ValueError("Le fichier JSON est vide")
            payload = payload[0]
        if not isinstance(payload, dict):
            raise ValueError("Le fichier JSON doit contenir un objet patient")
        if isinstance(payload.get("patient"), dict):
            payload = payload["patient"]
        return normalize_patient_data(payload)

    if suffix in {".csv", ".tsv"}:
        return _load_patient_from_delimited(text)

    if suffix == ".docx":
        return _load_patient_from_text(_extract_docx_text(file_bytes))

    if suffix in {".txt", ".md"}:
        return _load_patient_from_text(text)

    raise ValueError("Format non supporte. Utilisez JSON, CSV, TSV, TXT, MD ou DOCX.")


def normalize_patient_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Mappe des noms de champs heterogenes vers le schema API attendu."""
    patient: dict[str, Any] = {
        "age": None,
        "sexe": "",
        "taille_cm": None,
        "poids_kg": None,
        "symptomes": [],
        "symptome_libre": None,
        "tension_systolique": None,
        "tension_diastolique": None,
        "frequence_cardiaque": None,
        "temperature": None,
        "spo2": None,
        "antecedents": [],
        "traitements_en_cours": [],
        "examens_realises": [],
        "resultats_examens": {},
    }

    for key, value in raw.items():
        normalized_key = PATIENT_FIELD_ALIASES.get(_canonical_key(key))
        if not normalized_key:
            continue

        if normalized_key in LIST_FIELDS:
            patient[normalized_key] = _normalize_list(value)
        elif normalized_key == "resultats_examens":
            patient[normalized_key] = _normalize_results(value)
        elif normalized_key in NUMERIC_INT_FIELDS:
            patient[normalized_key] = _safe_int(value)
        elif normalized_key in NUMERIC_FLOAT_FIELDS:
            patient[normalized_key] = _safe_float(value)
        elif normalized_key == "sexe":
            patient[normalized_key] = _normalize_sex(value)
        else:
            patient[normalized_key] = _normalize_scalar(value)

    if patient["age"] is None:
        raise ValueError("Champ age manquant ou invalide")
    if not patient["sexe"]:
        raise ValueError("Champ sexe manquant ou invalide")

    return patient


def _load_patient_from_delimited(text: str) -> dict[str, Any]:
    delimiter = _detect_delimiter(text)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        raise ValueError("Le fichier tabulaire ne contient aucune ligne")
    return normalize_patient_data(rows[0])


def _load_patient_from_text(text: str) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    resultats: dict[str, str] = {}
    current_multiline_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            current_multiline_key = None
            continue

        match = re.match(r"^([A-Za-z0-9_][A-Za-z0-9_ /-]*)\s*:\s*(.*)$", line)
        if match:
            key = _canonical_key(match.group(1))
            value = match.group(2).strip()
            current_multiline_key = key
            if key in {"resultats", "resultats_examens", "exam_results"}:
                if value:
                    resultats.update(_parse_key_value_block(value))
            else:
                if value:
                    raw[key] = value
                elif key in LIST_FIELDS:
                    raw[key] = raw.get(key, [])
                else:
                    raw[key] = raw.get(key, "")
            continue

        if line.startswith("-") and current_multiline_key:
            item = line[1:].strip()
            if current_multiline_key in LIST_FIELDS:
                raw.setdefault(current_multiline_key, [])
                raw[current_multiline_key].append(item)
            elif current_multiline_key in {"resultats", "resultats_examens", "exam_results"}:
                resultats.update(_parse_key_value_block(item))
            else:
                previous = str(raw.get(current_multiline_key, "")).strip()
                raw[current_multiline_key] = f"{previous}\n{item}".strip()

    if resultats:
        raw["resultats_examens"] = resultats
    return normalize_patient_data(raw)


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[;\n|,]+", text)
    return [p.strip() for p in parts if p.strip()]


def _normalize_results(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip()}
    if isinstance(value, list):
        merged: dict[str, str] = {}
        for item in value:
            merged.update(_parse_key_value_block(str(item)))
        return merged
    return _parse_key_value_block(str(value))


def _parse_key_value_block(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    chunks = re.split(r"[;\n|]+", text)
    for chunk in chunks:
        if ":" in chunk:
            key, value = chunk.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key:
                result[key] = value
    return result


def _normalize_scalar(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_sex(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"m", "masculin", "male", "homme"}:
        return "M"
    if text in {"f", "feminin", "female", "femme"}:
        return "F"
    return ""


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _canonical_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _detect_delimiter(text: str) -> str:
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    candidates = [",", ";", "\t", "|"]
    counts = {candidate: first_line.count(candidate) for candidate in candidates}
    return max(counts, key=counts.get) if any(counts.values()) else ","


def _extract_docx_text(file_bytes: bytes) -> str:
    paragraphs: list[str] = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
        with archive.open("word/document.xml") as handle:
            xml_bytes = handle.read()

    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", namespace):
        runs = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        text = "".join(runs).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)
