"""
EXOFIT — Backend FastAPI
Partie 1 : Diagnostic IA (infirmière + LLM + télémédecin)
Partie 2 : Analyse ECG automatique
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import json
import os

from prompt_engine import build_prompt, call_llm
from ecg_pipeline import predict_pathology
from data_ingestion import load_patient_file
from real_ecg import preprocess_real_ecg

app = FastAPI(title="EXOFIT API", version="1.0.0")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARSED_CASES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cas_cliniques", "parsed"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODÈLES DE DONNÉES ────────────────────────────────────────────────────

class PatientData(BaseModel):
    age: int
    sexe: str                        # "M" | "F"
    taille_cm: Optional[int] = None
    poids_kg: Optional[float] = None
    symptomes: list[str]             # ex: ["douleur_thoracique", "dyspnee"]
    symptome_libre: Optional[str] = None
    tension_systolique: Optional[int] = None
    tension_diastolique: Optional[int] = None
    frequence_cardiaque: Optional[int] = None
    temperature: Optional[float] = None
    spo2: Optional[int] = None
    antecedents: list[str] = Field(default_factory=list)
    traitements_en_cours: list[str] = Field(default_factory=list)
    examens_realises: list[str] = Field(default_factory=list)  # ex: ["ecg", "nfs", "crp"]
    resultats_examens: dict = Field(default_factory=dict)      # ex: {"crp": "45 mg/L", "nfs": "Hb 10g/dL"}

class DiagnosticResponse(BaseModel):
    diagnostic_preliminaire: str
    diagnostics_differentiels: list[str]
    questions_complementaires: list[str]
    examens_proposes: list[str]
    niveau_urgence: str              # "faible" | "modere" | "eleve" | "critique"
    confiance: float                 # 0.0 à 1.0
    modele_utilise: str

class ValidationMedecin(BaseModel):
    cas_id: str
    diagnostic_valide: bool
    diagnostic_corrige: Optional[str] = None
    commentaire: Optional[str] = None
    medecin_id: str

class ECGDiagnosticResponse(BaseModel):
    pathologie_detectee: str
    probabilites: dict[str, float]   # {"fibrillation_auriculaire": 0.87, ...}
    features_extraites: dict
    confiance: float
    recommandation: str

# ─── ROUTES PARTIE 1 : DIAGNOSTIC IA ─────────────────────────────────────

@app.post("/api/diagnostic", response_model=DiagnosticResponse)
async def obtenir_diagnostic(patient: PatientData):
    """
    Reçoit les données patient de l'infirmière,
    génère un prompt structuré, appelle le LLM,
    retourne le diagnostic préliminaire.
    """
    try:
        prompt = build_prompt(patient.dict())
        resultat = await call_llm(prompt)
        return DiagnosticResponse(**resultat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/diagnostic/from-file", response_model=DiagnosticResponse)
async def obtenir_diagnostic_depuis_fichier(file: UploadFile = File(...)):
    """
    Charge un cas clinique reel depuis un fichier JSON, CSV, TSV, TXT ou MD,
    puis applique le meme moteur de diagnostic que la saisie manuelle.
    """
    contenu = await file.read()
    try:
        patient = load_patient_file(contenu, file.filename or "")
        prompt = build_prompt(patient)
        resultat = await call_llm(prompt)
        return DiagnosticResponse(**resultat)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clinical-cases")
async def list_clinical_cases():
    """
    Liste les cas cliniques parses disponibles sur disque.
    """
    if not os.path.isdir(PARSED_CASES_DIR):
        return {"cases": [], "count": 0, "directory": PARSED_CASES_DIR}

    cases = []
    for name in sorted(os.listdir(PARSED_CASES_DIR)):
        if not name.endswith(".json") or name == "index.json":
            continue
        cases.append(
            {
                "id": name[:-5],
                "filename": name,
                "path": os.path.join(PARSED_CASES_DIR, name),
            }
        )

    return {"cases": cases, "count": len(cases), "directory": PARSED_CASES_DIR}


@app.get("/api/clinical-cases/{case_id}")
async def get_clinical_case(case_id: str):
    """
    Retourne un cas clinique parse depuis data/cas_cliniques/parsed.
    """
    path = os.path.join(PARSED_CASES_DIR, f"{case_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Cas clinique introuvable")

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@app.post("/api/validation")
async def valider_diagnostic(validation: ValidationMedecin):
    """
    Le télémédecin valide ou corrige le diagnostic.
    Enregistre pour l'évaluation de fiabilité.
    """
    # En production : sauvegarder en base de données
    # Pour la démo : on log dans un fichier JSON
    log_path = "validations.json"
    validations = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            validations = json.load(f)
    validations.append(validation.dict())
    with open(log_path, "w") as f:
        json.dump(validations, f, indent=2, ensure_ascii=False)

    return {"status": "ok", "message": "Validation enregistrée"}


@app.get("/api/evaluation")
async def get_evaluation():
    """
    Calcule les métriques de fiabilité à partir des validations enregistrées.
    """
    log_path = "validations.json"
    if not os.path.exists(log_path):
        return {"message": "Aucune validation enregistrée"}

    with open(log_path) as f:
        validations = json.load(f)

    total = len(validations)
    corrects = sum(1 for v in validations if v["diagnostic_valide"])
    accuracy = corrects / total if total > 0 else 0

    return {
        "total_cas": total,
        "diagnostics_valides": corrects,
        "accuracy": round(accuracy, 3),
        "note": "Pour précision/rappel/F1, brancher sur les cas tests labellisés"
    }


# ─── ROUTES PARTIE 2 : ECG ───────────────────────────────────────────────

@app.post("/api/ecg/analyser", response_model=ECGDiagnosticResponse)
async def analyser_ecg(file: UploadFile = File(...)):
    """
    Reçoit un fichier ECG au format CSV (12 dérivations),
    applique le pipeline de prétraitement,
    retourne le diagnostic de pathologie.
    """
    if not (file.filename or "").lower().endswith((".csv", ".tsv")):
        raise HTTPException(status_code=400, detail="Format CSV ou TSV requis")

    contenu = await file.read()
    try:
        features = preprocess_real_ecg(contenu)
        resultat = predict_pathology(features)
        return ECGDiagnosticResponse(**resultat)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ecg/pathologies")
async def get_pathologies():
    """Liste des 7 pathologies détectables."""
    return {
        "pathologies": [
            {"id": "FA",  "nom": "Fibrillation auriculaire",      "critere_ecg": "Absence onde P, rythme irrégulier"},
            {"id": "TV",  "nom": "Tachycardie ventriculaire",     "critere_ecg": "QRS larges > 0.12s, FC > 100"},
            {"id": "BR",  "nom": "Bradycardie",                   "critere_ecg": "FC < 60 bpm"},
            {"id": "BAV", "nom": "Bloc auriculo-ventriculaire",   "critere_ecg": "PR > 0.20s ou dissociation P-QRS"},
            {"id": "HVG", "nom": "Hypertrophie ventriculaire G.", "critere_ecg": "Sokolow-Lyon > 35mm"},
            {"id": "ISC", "nom": "Ischémie myocardique",          "critere_ecg": "Sous-décalage ST > 1mm"},
            {"id": "IDM", "nom": "Infarctus du myocarde",         "critere_ecg": "Sus-décalage ST > 2mm, onde Q"},
        ]
    }


# ─── HEALTH CHECK ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "EXOFIT API opérationnelle", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
