"""
Construit un jeu de donnees ECG a partir de df_meta.pkl et des CSV reels.

Usage:
    python build_ecg_dataset.py
    python build_ecg_dataset.py --meta ../data/df_meta.pkl --ecg-dir ../data/ecg/raw --output ../data/ecg/dataset_index.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=str(Path("..") / "data" / "df_meta.pkl"))
    parser.add_argument("--ecg-dir", default=str(Path("..") / "data" / "ecg" / "raw"))
    parser.add_argument("--output", default=str(Path("..") / "data" / "ecg" / "dataset_index.json"))
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas est requis. Installez-le avec: pip install pandas") from exc

    meta_path = Path(args.meta).resolve()
    ecg_dir = Path(args.ecg_dir).resolve()
    output_path = Path(args.output).resolve()

    df_meta = pd.read_pickle(meta_path)
    path_column = find_path_column(list(df_meta.columns))
    if not path_column:
        raise SystemExit("Impossible de detecter la colonne de chemin ECG dans df_meta.pkl")

    label_columns = find_label_columns(list(df_meta.columns))
    dataset_rows = []
    missing = 0

    for row in df_meta.to_dict(orient="records"):
        resolved = resolve_ecg_path(str(row.get(path_column, "")), meta_path.parent, ecg_dir)
        if not resolved.exists():
            missing += 1
            continue

        item = {
            "ecg_path": str(resolved),
            "source_path_value": row.get(path_column),
        }
        for column in label_columns:
            item[column] = normalize_value(row.get(column))
        dataset_rows.append(item)

    output = {
        "meta_path": str(meta_path),
        "ecg_dir": str(ecg_dir),
        "path_column": path_column,
        "label_columns": label_columns,
        "total_rows": int(len(df_meta)),
        "matched_rows": len(dataset_rows),
        "missing_rows": missing,
        "rows": dataset_rows,
    }

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Index dataset ECG sauvegarde dans {output_path}")
    print(f"Lignes totales: {len(df_meta)}")
    print(f"Lignes associees a un CSV present: {len(dataset_rows)}")
    print(f"Lignes sans CSV resolu: {missing}")
    print("Colonnes labels candidates:")
    for column in label_columns:
        print(f"- {column}")


def find_path_column(columns: list[str]) -> str | None:
    for candidate in ["ecg_file_path", "file_path", "path", "csv_path"]:
        if candidate in columns:
            return candidate
    for column in columns:
        lowered = column.lower()
        if "path" in lowered and "ecg" in lowered:
            return column
    return None


def find_label_columns(columns: list[str]) -> list[str]:
    tokens = ("diag", "label", "target", "pathology", "machine", "cardio")
    return [column for column in columns if any(token in column.lower() for token in tokens)]


def resolve_ecg_path(value: str, metadata_dir: Path, ecg_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = (metadata_dir / path).resolve()
    if direct.exists():
        return direct
    return (ecg_dir / path.name).resolve()


def normalize_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


if __name__ == "__main__":
    main()
