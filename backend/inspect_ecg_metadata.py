"""
Inspecte df_meta.pkl et verifie la coherence avec les CSV ECG.

Usage:
    python inspect_ecg_metadata.py
    python inspect_ecg_metadata.py --meta ../data/df_meta.pkl --ecg-dir ../data/ecg/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=str(Path("..") / "data" / "df_meta.pkl"))
    parser.add_argument("--ecg-dir", default=str(Path("..") / "data" / "ecg" / "raw"))
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas est requis pour lire df_meta.pkl. Installez-le avec: pip install pandas"
        ) from exc

    meta_path = Path(args.meta).resolve()
    ecg_dir = Path(args.ecg_dir).resolve()

    df_meta = pd.read_pickle(meta_path)

    print(f"Fichier metadata : {meta_path}")
    print(f"Type : {type(df_meta)}")
    print(f"Shape : {getattr(df_meta, 'shape', None)}")
    print()

    columns = list(df_meta.columns)
    print("Colonnes disponibles :")
    for name in columns:
        print(f"- {name}")
    print()

    path_column = find_path_column(columns)
    if not path_column:
        print("Aucune colonne de chemin ECG evidente trouvee.")
        return

    print(f"Colonne chemin detectee : {path_column}")
    print()

    series = df_meta[path_column].dropna().astype(str)
    total_rows = len(df_meta)
    non_empty_paths = len(series)
    print(f"Lignes totales : {total_rows}")
    print(f"Lignes avec chemin ECG : {non_empty_paths}")

    resolved = series.apply(lambda value: resolve_ecg_path(value, meta_path.parent, ecg_dir))
    exists = resolved.apply(Path.exists)
    existing_count = int(exists.sum())
    missing_count = len(resolved) - existing_count

    print(f"CSV trouves sur disque : {existing_count}")
    print(f"CSV manquants : {missing_count}")
    print()

    if existing_count:
        sample_existing = resolved[exists].head(5)
        print("Exemples de CSV resolus :")
        for path in sample_existing:
            print(f"- {path}")
        print()

    if missing_count:
        sample_missing = resolved[~exists].head(10)
        print("Exemples de CSV manquants :")
        for path in sample_missing:
            print(f"- {path}")
        print()

    diagnosis_columns = [
        col for col in columns
        if any(token in col.lower() for token in ["diag", "label", "target", "pathology"])
    ]
    if diagnosis_columns:
        print("Colonnes potentiellement utiles pour les labels :")
        for name in diagnosis_columns:
            print(f"- {name}")
        print()
        for name in diagnosis_columns[:3]:
            counts = df_meta[name].astype(str).value_counts(dropna=False).head(10)
            print(f"Top valeurs pour {name} :")
            print(counts.to_string())
            print()

    print("Apercu des 3 premieres lignes :")
    print(df_meta.head(3).to_string())


def find_path_column(columns: list[str]) -> str | None:
    for candidate in ["ecg_file_path", "file_path", "path", "csv_path"]:
        if candidate in columns:
            return candidate
    for column in columns:
        lowered = column.lower()
        if "path" in lowered and "ecg" in lowered:
            return column
    return None


def resolve_ecg_path(value: str, metadata_dir: Path, ecg_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = (metadata_dir / path).resolve()
    if direct.exists():
        return direct
    return (ecg_dir / path.name).resolve()


if __name__ == "__main__":
    main()
