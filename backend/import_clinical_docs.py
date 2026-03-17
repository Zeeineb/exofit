"""
Convertit les documents Word de cas cliniques en fichiers JSON exploitables.

Usage:
    python import_clinical_docs.py
    python import_clinical_docs.py --input-dir ../data/cas_cliniques/raw --output-dir ../data/cas_cliniques/parsed
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


CASE_PATTERN = re.compile(r"(?=Cas clinique\s+\d+\s*:?)", re.IGNORECASE)
SYMPTOM_KEYWORDS = {
    "douleur_thoracique": ["douleur thoracique", "oppression thoracique", "douleurs thoraciques"],
    "dyspnee": ["dyspnee", "essoufflement"],
    "palpitations": ["palpitations"],
    "syncope": ["syncope", "perte de connaissance", "malaise"],
    "fievre": ["fievre", "38", "39", "temperature"],
    "toux": ["toux"],
    "cephalees": ["cephalee", "maux de tete"],
    "douleur_abdominale": ["douleur abdominale", "douleurs abdominales", "epigastrique", "bas-ventre"],
    "nausees_vomissements": ["nausee", "nausees", "vomissement", "vomissements"],
    "asthenie": ["asthenie", "fatigue"],
    "oedemes_membres": ["oedeme", "oedemes"],
    "vertiges": ["vertige", "vertiges"],
}
SECTION_HEADERS = {
    "presentation": {"presentation du patient", "informations generales", "patient", "patiente", "constantes"},
    "motif": {"motif de consultation", "symptomes actuels", "histoire de la maladie"},
    "symptomes": {"symptomes", "signes cliniques"},
    "antecedents": {"antecedents", "antecedents medicaux"},
    "traitements": {"traitements actuels", "traitements"},
    "examen_clinique": {"examen clinique", "resultats de l'examen clinique", "examen neurologique", "inspection", "palpation abdominale", "examen des voies respiratoires", "examen pulmonaire", "examen cardiovasculaire", "examen abdominal"},
    "examens_complementaires": {"examens complementaires", "biologie sanguine", "investigations biologiques", "biologie", "echographie abdominale", "radiographie thoracique", "gaz du sang arteriel", "analyse urinaire rapide"},
    "diagnostic": {"diagnostic", "diagnostic retenu", "hypothese diagnostique"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(Path("..") / "data" / "cas_cliniques" / "raw"))
    parser.add_argument("--output-dir", default=str(Path("..") / "data" / "cas_cliniques" / "parsed"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = []
    for docx_path in sorted(input_dir.glob("*.docx")):
        extracted = parse_docx_cases(docx_path)
        all_cases.extend(extracted)

    for case in all_cases:
        case_path = output_dir / f"{case['id']}.json"
        with case_path.open("w", encoding="utf-8") as handle:
            json.dump(case, handle, indent=2, ensure_ascii=False)

    index_path = output_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(all_cases, handle, indent=2, ensure_ascii=False)

    print(f"{len(all_cases)} cas exportes dans {output_dir}")
    print(f"Index global: {index_path}")


def parse_docx_cases(path: Path) -> list[dict]:
    text = extract_docx_text(path.read_bytes())
    blocks = [block.strip() for block in CASE_PATTERN.split(text) if block.strip()]
    cases = []
    for block in blocks:
        if not block.lower().startswith("cas clinique"):
            continue
        case = parse_case_block(block, path)
        if case:
            cases.append(case)
    return cases


def parse_case_block(block: str, source_path: Path) -> dict | None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None

    match = re.search(r"Cas clinique\s+(\d+)", lines[0], re.IGNORECASE)
    if not match:
        return None
    case_number = int(match.group(1))

    sections = collect_sections(lines[1:])
    merged_text = "\n".join(lines)
    age = extract_age(merged_text)
    sexe = extract_sex(merged_text)

    patient = {
        "age": age or 0,
        "sexe": sexe or "",
        "taille_cm": None,
        "poids_kg": None,
        "symptomes": infer_symptoms(merged_text),
        "symptome_libre": build_free_text(sections),
        "tension_systolique": extract_bp(merged_text)[0],
        "tension_diastolique": extract_bp(merged_text)[1],
        "frequence_cardiaque": extract_first_int(merged_text, [r"\bFC\s*:\s*(\d+)", r"frequence cardiaque\s*:\s*(\d+)"]),
        "temperature": extract_first_float(merged_text, [r"temperature\s*:\s*(\d+[.,]?\d*)", r"fievre a\s*(\d+[.,]?\d*)"]),
        "spo2": extract_first_int(merged_text, [r"saturation(?: en oxygene| en o2| o2| spo2)?\s*[:=]?\s*(\d+)", r"SpO.\s*[:=]?\s*(\d+)"]),
        "antecedents": bullet_or_sentences(sections.get("antecedents", [])),
        "traitements_en_cours": bullet_or_sentences(sections.get("traitements", [])),
        "examens_realises": infer_exams(sections),
        "resultats_examens": build_exam_results(sections),
    }

    case = {
        "id": make_case_id(source_path.stem, case_number),
        "source_document": source_path.name,
        "case_number": case_number,
        "patient": patient,
        "diagnostic_reference": extract_diagnostic(sections),
        "raw_text": block,
    }
    return case


def collect_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"general": []}
    current = "general"

    for line in lines:
        inline_header = split_inline_header(line)
        if inline_header:
            header, content = inline_header
            current = canonical_section(header)
            sections.setdefault(current, [])
            if content:
                sections[current].append(content)
            continue

        normalized = normalize_heading(line)
        if is_known_heading(normalized):
            current = canonical_section(normalized)
            sections.setdefault(current, [])
            continue

        sections.setdefault(current, []).append(line)

    return sections


def split_inline_header(line: str) -> tuple[str, str] | None:
    match = re.match(r"^([A-Za-zÀ-ÿ0-9'() /-]+)\s*:\s*(.+)$", line)
    if not match:
        return None
    header = match.group(1).strip()
    content = match.group(2).strip()
    if header.lower().startswith("cas clinique"):
        return None
    if not is_known_heading(normalize_heading(header)):
        return None
    return header, content


def canonical_section(header: str) -> str:
    normalized = normalize_heading(header)
    for section_name, aliases in SECTION_HEADERS.items():
        if normalized in aliases:
            return section_name
    return "general"


def is_known_heading(normalized_header: str) -> bool:
    return any(normalized_header in aliases for aliases in SECTION_HEADERS.values())


def normalize_heading(value: str) -> str:
    text = value.lower().strip().replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    replacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a",
        "î": "i", "ï": "i",
        "ô": "o", "ö": "o",
        "ù": "u", "û": "u", "ü": "u",
        "ç": "c",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.rstrip(":")


def extract_age(text: str) -> int | None:
    match = re.search(r"(\d{1,3})\s*ans", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_sex(text: str) -> str | None:
    lowered = normalize_heading(text)
    if "sexe : masculin" in lowered or "homme" in lowered or "patient :" in lowered and "femme" not in lowered:
        return "M"
    if "sexe : feminin" in lowered or "femme" in lowered or "patiente" in lowered or "adolescente" in lowered:
        return "F"
    return None


def infer_symptoms(text: str) -> list[str]:
    lowered = normalize_heading(text)
    matches = []
    for code, keywords in SYMPTOM_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            matches.append(code)
    return matches


def build_free_text(sections: dict[str, list[str]]) -> str | None:
    ordered = []
    for key in ["motif", "symptomes", "examen_clinique", "examens_complementaires", "general"]:
        ordered.extend(sections.get(key, []))
    text = "\n".join(ordered).strip()
    return text or None


def extract_bp(text: str) -> tuple[int | None, int | None]:
    patterns = [
        r"\bTA\s*:\s*(\d+)\s*/\s*(\d+)",
        r"tension arterielle\s*:\s*(\d+)\s*/\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalize_heading(text), re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def extract_first_int(text: str, patterns: Iterable[str]) -> int | None:
    lowered = normalize_heading(text)
    for pattern in patterns:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_first_float(text: str, patterns: Iterable[str]) -> float | None:
    lowered = normalize_heading(text)
    for pattern in patterns:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def bullet_or_sentences(lines: list[str]) -> list[str]:
    cleaned = [line.lstrip("- ").strip() for line in lines if line.strip()]
    if len(cleaned) <= 1 and cleaned:
        return [part.strip() for part in re.split(r"[.;]\s+", cleaned[0]) if part.strip()]
    return cleaned


def infer_exams(sections: dict[str, list[str]]) -> list[str]:
    exams = []
    exam_text = "\n".join(sections.get("examens_complementaires", []) + sections.get("examen_clinique", []))
    lowered = normalize_heading(exam_text)
    mapping = {
        "ecg": ["ecg"],
        "biologie": ["troponine", "crp", "creatinine", "nfs", "hba1c", "bilirubine", "alat", "asat", "albumine"],
        "radiographie": ["radiographie", "thoracique"],
        "echographie": ["echographie"],
        "gaz_du_sang": ["pao2", "paco2", "gaz du sang"],
        "analyse_urinaire": ["leucocytes", "nitrites", "analyse urinaire"],
    }
    for exam, keywords in mapping.items():
        if any(keyword in lowered for keyword in keywords):
            exams.append(exam)
    return exams


def build_exam_results(sections: dict[str, list[str]]) -> dict[str, str]:
    results = {}
    for section in ["examen_clinique", "examens_complementaires"]:
        for line in sections.get(section, []):
            if ":" in line:
                key, value = line.split(":", 1)
                results[key.strip()] = value.strip()
    return results


def extract_diagnostic(sections: dict[str, list[str]]) -> str | None:
    diagnostic_lines = sections.get("diagnostic", [])
    if diagnostic_lines:
        return " ".join(diagnostic_lines).strip()
    return None


def make_case_id(source_stem: str, case_number: int) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", normalize_heading(source_stem)).strip("_")
    return f"{normalized}_cas_{case_number:03d}"


def extract_docx_text(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            xml_content = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("Fichier DOCX invalide ou corrompu") from exc

    root = ET.fromstring(xml_content)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        chunks = []
        for text_node in paragraph.findall(".//w:t", namespace):
            if text_node.text:
                chunks.append(text_node.text)
        line = "".join(chunks).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


if __name__ == "__main__":
    main()
