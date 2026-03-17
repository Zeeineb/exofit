"""
EXOFIT - Chargeur ECG reel.
Accepte des exports CSV/TSV avec en-tetes, colonne temps et separateurs variables.
"""

from __future__ import annotations

import csv
import io

import numpy as np

from ecg_pipeline import bandpass_filter, extract_features, notch_filter


LEAD_NAMES = {"i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"}
TIME_COLUMN_NAMES = {"time", "temps", "timestamp", "sample", "index"}


def preprocess_real_ecg(file_bytes: bytes) -> dict:
    parsed = read_real_ecg_table(file_bytes)
    raw = parsed["signal"]
    filtered = bandpass_filter(raw)
    filtered = notch_filter(filtered)
    features = extract_features(filtered)
    return {
        "features": features,
        "signal": filtered,
        "metadata": parsed["metadata"],
    }


def read_real_ecg_table(file_bytes: bytes) -> dict:
    content = file_bytes.decode("utf-8-sig", errors="replace")
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Fichier ECG vide")

    delimiter = _detect_delimiter(lines)
    reader = csv.reader(io.StringIO(content), delimiter=delimiter)
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("Aucune donnee ECG lisible")

    header = rows[0]
    signal_rows = rows
    selected_indexes = None
    selected_names = []

    if _looks_like_header(header):
        selected_indexes = _select_signal_columns(header)
        selected_names = [
            _normalize_column_name(header[idx])
            for idx in selected_indexes
            if idx < len(header)
        ]
        signal_rows = rows[1:]

    samples = []
    expected_len = None
    for row in signal_rows:
        values = _extract_row_values(row, selected_indexes)
        if not values:
            continue
        if expected_len is None:
            expected_len = len(values)
        if len(values) >= expected_len:
            samples.append(values[:expected_len])

    if not samples:
        raise ValueError("Impossible de parser le signal ECG depuis le fichier fourni")

    arr = np.array(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    original_lead_count = int(arr.shape[1])
    detected_leads = [
        name.upper()
        for name in selected_names
        if name in LEAD_NAMES
    ]
    if arr.shape[1] < 12:
        arr = np.tile(arr, (1, 12 // arr.shape[1] + 1))[:, :12]
    elif arr.shape[1] > 12:
        arr = arr[:, :12]

    metadata = {
        "original_lead_count": original_lead_count,
        "detected_leads": detected_leads,
        "is_twelve_lead": original_lead_count >= 12 and _has_all_standard_leads(detected_leads),
    }

    return {"signal": arr, "metadata": metadata}


def _extract_row_values(row: list[str], selected_indexes: list[int] | None) -> list[float]:
    if selected_indexes is None:
        values = [_safe_float(cell) for cell in row]
    else:
        values = [_safe_float(row[idx]) for idx in selected_indexes if idx < len(row)]
    return [value for value in values if value is not None]


def _detect_delimiter(lines: list[str]) -> str:
    candidates = [",", ";", "\t", "|"]
    sample = lines[:5]
    scores = {candidate: sum(line.count(candidate) for line in sample) for candidate in candidates}
    return max(scores, key=scores.get) if any(scores.values()) else ","


def _looks_like_header(row: list[str]) -> bool:
    normalized = [_normalize_column_name(cell) for cell in row]
    if any(name in LEAD_NAMES or name in TIME_COLUMN_NAMES for name in normalized):
        return True
    return any(_safe_float(cell) is None for cell in row)


def _select_signal_columns(header: list[str]) -> list[int]:
    normalized = [_normalize_column_name(cell) for cell in header]
    lead_indexes = [idx for idx, name in enumerate(normalized) if name in LEAD_NAMES]
    if lead_indexes:
        return lead_indexes
    return [idx for idx, name in enumerate(normalized) if name not in TIME_COLUMN_NAMES]


def _normalize_column_name(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _safe_float(value: str) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text.replace(",", "."))
    except (TypeError, ValueError):
        return None


def _has_all_standard_leads(detected_leads: list[str]) -> bool:
    if not detected_leads:
        return False
    expected = {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"}
    return expected.issubset(set(detected_leads))
