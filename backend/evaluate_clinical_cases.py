"""
Evalue les cas cliniques parses contre le moteur de prompt.

Usage:
    python evaluate_clinical_cases.py
    python evaluate_clinical_cases.py --parsed-dir ../data/cas_cliniques/parsed
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from prompt_engine import build_prompt, call_llm


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-dir", default=str(Path("..") / "data" / "cas_cliniques" / "parsed"))
    parser.add_argument("--output", default=str(Path("..") / "data" / "cas_cliniques" / "evaluation_results.json"))
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir).resolve()
    output_path = Path(args.output).resolve()

    case_files = sorted(path for path in parsed_dir.glob("*.json") if path.name != "index.json")
    if not case_files:
        raise SystemExit(f"Aucun cas parse trouve dans {parsed_dir}")

    results = []
    for path in case_files:
        case = json.loads(path.read_text(encoding="utf-8"))
        patient = case["patient"]
        prompt = build_prompt(patient)
        try:
            response = await call_llm(prompt)
        except Exception as exc:
            response = {
                "diagnostic_preliminaire": f"ERREUR: {exc}",
                "diagnostics_differentiels": [],
                "questions_complementaires": [],
                "examens_proposes": [],
                "niveau_urgence": "modere",
                "confiance": 0.0,
                "modele_utilise": "error",
            }

        results.append(
            {
                "id": case["id"],
                "source_document": case["source_document"],
                "diagnostic_reference": case.get("diagnostic_reference"),
                "patient": patient,
                "llm_response": response,
            }
        )

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(results)} cas evalues.")
    print(f"Resultats sauvegardes dans {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
