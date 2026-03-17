"""
Entraine et evalue le pipeline ECG a partir de dataset_index.json.

Usage:
    python train_eval_ecg_dataset.py
    python train_eval_ecg_dataset.py --dataset ../data/ecg/dataset_index.json --label-column diagnostic_col
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecg_pipeline import PATHOLOGIES, WINDOW_SAMPLES, N_LEADS, build_cnn_model, diagnose_by_rules
from evaluation import compute_metrics
from real_ecg import preprocess_real_ecg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(Path("..") / "data" / "ecg" / "dataset_index.json"))
    parser.add_argument("--label-column", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=str(Path("..") / "data" / "ecg" / "training_report.json"))
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("Le dataset ECG est vide")

    label_column = args.label_column or pick_label_column(payload.get("label_columns", []), rows)
    if not label_column:
        raise SystemExit("Aucune colonne de label exploitable detectee. Passez --label-column.")

    X, y, kept_paths = load_dataset(rows, label_column)
    if len(X) < 4:
        raise SystemExit("Pas assez d'ECG valides pour entrainer/evaluer")

    split_idx = max(1, int(len(X) * args.train_ratio))
    split_idx = min(split_idx, len(X) - 1)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    y_test_labels = [PATHOLOGIES[int(index)] for index in y_test]

    report = {
        "dataset_path": str(dataset_path),
        "label_column": label_column,
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "paths_sample": kept_paths[:5],
    }

    rule_predictions = predict_by_rules(X_test)
    report["rules_metrics"] = compute_metrics(y_test_labels, rule_predictions, PATHOLOGIES)

    model = build_cnn_model()
    if model is None:
        report["cnn"] = {"status": "tensorflow_non_installe"}
    else:
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2 if len(X_train) >= 5 else 0.0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
        )
        probs = model.predict(X_test, verbose=0)
        y_pred = [PATHOLOGIES[int(np.argmax(row))] for row in probs]
        report["cnn"] = {
            "status": "trained",
            "final_loss": float(history.history["loss"][-1]),
            "metrics": compute_metrics(y_test_labels, y_pred, PATHOLOGIES),
        }

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport sauvegarde dans {output_path}")
    print(f"Label utilise: {label_column}")
    print(f"Echantillons valides: {len(X)}")


def pick_label_column(label_columns: list[str], rows: list[dict]) -> str | None:
    preferred = [
        "diagnostic",
        "diagnosis",
        "pathology",
        "label",
        "machine_diagnosis",
        "cardiologist_diagnosis",
    ]
    lowered_map = {column.lower(): column for column in label_columns}
    for name in preferred:
        if name in lowered_map:
            return lowered_map[name]
    for column in label_columns:
        if any(normalize_label(row.get(column)) in PATHOLOGIES for row in rows[:50]):
            return column
    return label_columns[0] if label_columns else None


def load_dataset(rows: list[dict], label_column: str):
    signals = []
    labels = []
    kept_paths = []

    for row in rows:
        normalized_label = normalize_label(row.get(label_column))
        if normalized_label not in PATHOLOGIES:
            continue

        path = Path(row["ecg_path"])
        if not path.exists():
            continue

        try:
            preprocessed = preprocess_real_ecg(path.read_bytes())
        except Exception:
            continue

        signal = preprocessed["signal"]
        window = signal[:WINDOW_SAMPLES]
        if window.shape[0] < WINDOW_SAMPLES:
            pad = np.zeros((WINDOW_SAMPLES - window.shape[0], window.shape[1]))
            window = np.vstack([window, pad])
        if window.shape[1] > N_LEADS:
            window = window[:, :N_LEADS]
        elif window.shape[1] < N_LEADS:
            window = np.tile(window, (1, N_LEADS // window.shape[1] + 1))[:, :N_LEADS]

        signals.append(window.astype(np.float32))
        labels.append(PATHOLOGIES.index(normalized_label))
        kept_paths.append(str(path))

    return np.array(signals), np.array(labels), kept_paths


def predict_by_rules(X_test: np.ndarray) -> list[str]:
    predictions = []
    for signal in X_test:
        features = {
            "features": {},
            "signal": signal,
        }
        from ecg_pipeline import extract_features
        extracted = extract_features(signal)
        result = diagnose_by_rules(extracted)
        predictions.append(result["pathologie"])
    return predictions


def normalize_label(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    replacements = {
        " ": "_",
        "-": "_",
        "é": "e",
        "è": "e",
        "ê": "e",
        "à": "a",
        "ô": "o",
        "î": "i",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    aliases = {
        "fibrillation_auriculaire": "fibrillation_auriculaire",
        "fa": "fibrillation_auriculaire",
        "tachycardie_ventriculaire": "tachycardie_ventriculaire",
        "tv": "tachycardie_ventriculaire",
        "bradycardie": "bradycardie",
        "bloc_auriculo_ventriculaire": "bloc_auriculo_ventriculaire",
        "bav": "bloc_auriculo_ventriculaire",
        "hypertrophie_ventriculaire_gauche": "hypertrophie_ventriculaire_gauche",
        "hvg": "hypertrophie_ventriculaire_gauche",
        "ischemie_myocardique": "ischemie_myocardique",
        "infarctus_du_myocarde": "infarctus_du_myocarde",
        "idm": "infarctus_du_myocarde",
        "normal": "normal",
    }
    return aliases.get(text)


if __name__ == "__main__":
    main()
