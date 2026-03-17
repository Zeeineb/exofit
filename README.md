# EXOFIT - Assistance Medicale IA
**Centrale Lille · MIAS M2 · Mars 2026**

> Les ingenieurs peuvent-ils reduire la carence en medecins ?

---

## Structure du projet

```text
exofit/
|-- backend/
|   |-- main.py              # API FastAPI
|   |-- prompt_engine.py     # Construction des prompts + appel LLM
|   |-- ecg_pipeline.py      # Features ECG + regles + architecture CNN
|   |-- real_ecg.py          # Import ECG reels CSV/TSV
|   |-- data_ingestion.py    # Import cas cliniques reels JSON/CSV/TXT
|   |-- evaluation.py        # Metriques de performance
|   `-- requirements.txt
|-- frontend/
|   `-- index.html           # Application web standalone
`-- README.md
```

---

## Installation

```bash
cd backend
pip install -r requirements.txt
```

Optionnel pour le CNN :

```bash
pip install tensorflow
```

---

## Configuration des cles API

Creez un fichier `.env` dans `backend/` :

```env
ACTIVE_LLM=anthropic

ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

Sans cle API, l'application reste utilisable en mode demo.

---

## Lancement

```bash
# Terminal 1 : backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 : frontend
cd frontend
python -m http.server 3000
```

Puis ouvrir `http://localhost:3000`.

---

## Utilisation

### Diagnostic clinique

Deux modes sont disponibles :

1. saisie manuelle du patient dans l'interface
2. import d'un dossier clinique via un fichier `JSON`, `CSV`, `TSV`, `TXT` ou `MD`

Le frontend appelle :

- `POST /api/diagnostic`
- `POST /api/diagnostic/from-file`

### Analyse ECG

Le frontend accepte maintenant des ECG reels en `CSV` ou `TSV` :

- avec ou sans en-tete
- avec colonne temps facultative (`time`, `temps`, `timestamp`, `sample`, `index`)
- avec 6 ou 12 derivations

Le frontend appelle :

- `POST /api/ecg/analyser`

---

## Formats de donnees supportes

### Cas cliniques

Champs reconnus automatiquement :

- `age`
- `sexe`
- `taille_cm`, `poids_kg`
- `symptomes`, `symptome_libre`
- `tension_systolique`, `tension_diastolique`
- `frequence_cardiaque`, `temperature`, `spo2`
- `antecedents`
- `traitements_en_cours`
- `examens_realises`
- `resultats_examens`

Exemple `TXT` :

```txt
age: 58
sexe: M
symptomes: douleur_thoracique; dyspnee
symptome_libre: douleur oppressive irradiant dans le bras gauche
antecedents:
- HTA
- diabete type 2
traitements_en_cours:
- metformine 1g
- amlodipine 5mg
examens_realises: ecg
resultats_examens:
- ecg: sus-decalage ST en V1-V4
```

Exemple `CSV` :

```csv
age,sexe,symptomes,symptome_libre,antecedents,examens_realises,resultats_examens
58,M,"douleur_thoracique;dyspnee","douleur oppressive","HTA;diabete type 2","ecg","ecg: sus-decalage ST en V1-V4"
```

### ECG

Exemple `CSV` :

```csv
time,I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6
0.000,0.12,0.25,0.08,-0.03,0.01,0.04,0.11,0.09,0.05,0.03,0.15,0.12
0.002,0.13,0.27,0.09,-0.03,0.01,0.04,0.12,0.10,0.05,0.04,0.16,0.12
```

Le chargeur ignore les colonnes de temps et selectionne automatiquement les colonnes de signal.

---

## Evaluation

Les cas cliniques sans diagnostic servent a l'inference, pas a l'evaluation.
Pour evaluer correctement la partie 1, il faut ajouter :

- un diagnostic de reference
- ou une validation/correction medecin

Pour l'ECG, le pipeline fonctionne deja en inference sur fichiers reels, mais le CNN doit toujours etre entraine sur des ECG labellises.

---

## Tests rapides

```bash
cd backend
python prompt_engine.py
python ecg_pipeline.py
python evaluation.py
```

---

## Notes

- Ce projet est une preuve de concept academique, pas un dispositif medical certifie.
- Les donnees de sante doivent etre traitees dans un cadre RGPD adapte.
- En contexte medical, le rappel est plus important que l'accuracy seule.
