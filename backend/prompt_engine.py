"""
EXOFIT — Prompt Engine
Construit le prompt structuré à 3 blocs et appelle l'API LLM.
Testable sans données réelles grâce aux cas synthétiques.
"""

import json
import httpx
import os
from typing import Any
from pathlib import Path


def _load_local_env() -> None:
    """
    Charge backend/.env dans os.environ si present.
    Priorise le .env local pour eviter les variables stale du shell.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


_load_local_env()

# ─── CONFIGURATION ────────────────────────────────────────────────────────

LLM_CONFIGS = {
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
        "header_key": "Authorization",
        "header_prefix": "Bearer ",
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-opus-4-6",
        "header_key": "x-api-key",
        "header_prefix": "",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "model": "gemini-1.5-flash",
        "header_key": "x-goog-api-key",
        "header_prefix": "",
        "env_key": "GOOGLE_API_KEY",
    },
    "huggingface": {
        "url": "https://router.huggingface.co/v1/chat/completions",
        "model": os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct-1M"),
        "header_key": "Authorization",
        "header_prefix": "Bearer ",
        "env_key": "HUGGINGFACE_API_KEY",
    },
}

ACTIVE_LLM = os.getenv("ACTIVE_LLM", "anthropic").strip().lower()

# ─── LISTE DES SYMPTÔMES STANDARDISÉS ─────────────────────────────────────

SYMPTOMES_LABELS = {
    "douleur_thoracique":     "Douleur thoracique",
    "dyspnee":                "Dyspnée (essoufflement)",
    "palpitations":           "Palpitations",
    "syncope":                "Syncope / perte de connaissance",
    "fievre":                 "Fièvre",
    "toux":                   "Toux",
    "cephalees":              "Céphalées",
    "douleur_abdominale":     "Douleur abdominale",
    "nausees_vomissements":   "Nausées / vomissements",
    "asthenie":               "Asthénie (fatigue intense)",
    "oedemes_membres":        "Œdèmes des membres inférieurs",
    "vertiges":               "Vertiges",
}

# ─── CONSTRUCTION DU PROMPT ───────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant médical expert utilisé dans un contexte de télémédecine.
Tu assistes une infirmière qui ne dispose pas de médecin sur place.
Tu as accès au matériel suivant : tensiomètre, thermomètre, stéthoscope, ECG 6 dérivations, analyseur sanguin portable.
Ton rôle est de proposer un diagnostic différentiel structuré, des questions complémentaires à poser au patient, et des examens à envisager.
Tu ne poses PAS de diagnostic définitif — ce rôle revient au médecin téléconsultant.
Si tu détectes un signe de gravité immédiate (douleur thoracique + dyspnée + ECG anormal, trouble de conscience, etc.), indique-le clairement avec le niveau d'urgence CRITIQUE.
Réponds UNIQUEMENT en JSON valide selon le format demandé. Pas de texte avant ou après le JSON."""

def build_prompt(patient: dict) -> dict:
    """
    Construit le prompt structuré en 3 blocs :
    1. Contexte (rôle IA, matériel, objectif)
    2. Informations patient (données structurées)
    3. Demandes précises à l'IA
    """
    symptomes_str = "\n".join([
        f"  - {SYMPTOMES_LABELS.get(s, s)}"
        for s in patient.get("symptomes", [])
    ])
    if patient.get("symptome_libre"):
        symptomes_str += f"\n  - (Libre) {patient['symptome_libre']}"

    antecedents_str = "\n".join([f"  - {a}" for a in patient.get("antecedents", [])]) or "  - Aucun renseigné"
    traitements_str = "\n".join([f"  - {t}" for t in patient.get("traitements_en_cours", [])]) or "  - Aucun"
    examens_str = "\n".join([f"  - {e}" for e in patient.get("examens_realises", [])]) or "  - Aucun réalisé"

    resultats_str = ""
    if patient.get("resultats_examens"):
        resultats_str = "\n".join([f"  - {k}: {v}" for k, v in patient["resultats_examens"].items()])
    else:
        resultats_str = "  - Non disponibles"

    constantes = []
    if patient.get("tension_systolique") and patient.get("tension_diastolique"):
        constantes.append(f"TA : {patient['tension_systolique']}/{patient['tension_diastolique']} mmHg")
    if patient.get("frequence_cardiaque"):
        constantes.append(f"FC : {patient['frequence_cardiaque']} bpm")
    if patient.get("temperature"):
        constantes.append(f"T° : {patient['temperature']} °C")
    if patient.get("spo2"):
        constantes.append(f"SpO2 : {patient['spo2']} %")
    constantes_str = "\n".join([f"  - {c}" for c in constantes]) or "  - Non mesurées"

    bmi = None
    if patient.get("taille_cm") and patient.get("poids_kg"):
        bmi = round(patient["poids_kg"] / (patient["taille_cm"] / 100) ** 2, 1)

    user_content = f"""## INFORMATIONS PATIENT

- Âge : {patient['age']} ans
- Sexe : {patient['sexe']}
- Taille / Poids : {patient.get('taille_cm', 'NC')} cm / {patient.get('poids_kg', 'NC')} kg{f' (IMC {bmi})' if bmi else ''}

### Symptômes rapportés
{symptomes_str if symptomes_str else '  - Aucun symptôme renseigné'}

### Constantes vitales
{constantes_str}

### Antécédents médicaux
{antecedents_str}

### Traitements en cours
{traitements_str}

### Examens réalisés
{examens_str}

### Résultats d'examens disponibles
{resultats_str}

## DEMANDES

Fournis une réponse JSON STRICTEMENT dans ce format :
{{
  "diagnostic_preliminaire": "string — hypothèse diagnostique principale",
  "diagnostics_differentiels": ["string", "string", "string"],
  "questions_complementaires": ["string", "string", "string"],
  "examens_proposes": ["string", "string"],
  "niveau_urgence": "faible|modere|eleve|critique",
  "confiance": 0.0,
  "modele_utilise": "string"
}}

Contraintes :
- Si des examens sont manquants (ex: ECG non réalisé), mentionne-le dans les examens_proposes
- Le niveau_urgence doit être "critique" si tu détectes un signe de gravité immédiate
- La confiance est entre 0.0 et 1.0 selon la complétude des données
- Reste concis et cliniquement pertinent"""

    return {
        "system": SYSTEM_PROMPT,
        "user": user_content
    }


# ─── APPEL LLM ────────────────────────────────────────────────────────────

async def call_llm(prompt: dict, provider: str = ACTIVE_LLM) -> dict:
    """
    Appelle l'API LLM configurée et parse la réponse JSON.
    Supporte OpenAI, Anthropic, Google et Hugging Face.
    """
    provider = (provider or "").strip().lower()
    if provider not in LLM_CONFIGS:
        allowed = ", ".join(sorted(LLM_CONFIGS.keys()))
        raise RuntimeError(f"Provider LLM invalide: '{provider}'. Valeurs supportees: {allowed}")
    config = LLM_CONFIGS[provider]
    api_key = os.getenv(config["env_key"])

    if not api_key:
        # MODE DÉMO : retourne un diagnostic synthétique sans API
        return _demo_response(provider)

    headers = {
        "Content-Type": "application/json",
        config["header_key"]: f"{config['header_prefix']}{api_key}"
    }
    params = None

    # Formater le body selon le provider
    if provider == "openai":
        body = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
    elif provider == "anthropic":
        body = {
            "model": config["model"],
            "max_tokens": 1024,
            "system": prompt["system"],
            "messages": [{"role": "user", "content": prompt["user"]}]
        }
    elif provider == "google":
        # Gemini accepte aussi la cle API en query param ; plus robuste selon les configs.
        params = {"key": api_key}
        body = {
            "contents": [{"parts": [{"text": prompt["system"] + "\n\n" + prompt["user"]}]}]
        }
    else:  # huggingface
        body = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(config["url"], headers=headers, params=params, json=body)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Renvoyer le message API detaille (utile pour diagnostiquer les 400 Google).
            error_payload = ""
            try:
                error_payload = response.text
            except Exception:
                error_payload = str(exc)
            raise RuntimeError(
                f"{provider} API error ({response.status_code}): {error_payload}"
            ) from exc
        data = response.json()

    # Extraire le texte de la réponse selon le provider
    if provider == "openai":
        text = data["choices"][0]["message"]["content"]
    elif provider == "anthropic":
        text = data["content"][0]["text"]
    elif provider == "google":
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        text = _extract_hf_text(data)

    # Parser le JSON
    result = json.loads(text)
    result["modele_utilise"] = f"{provider}/{config['model']}"
    return result


def _demo_response(provider: str) -> dict:
    """Réponse de démonstration quand aucune API key n'est configurée."""
    return {
        "diagnostic_preliminaire": "MODE DÉMO — Configurez une clé API dans le fichier .env",
        "diagnostics_differentiels": [
            "Douleur thoracique d'origine coronarienne",
            "Syndrome coronarien aigu",
            "Péricardite aiguë"
        ],
        "questions_complementaires": [
            "La douleur irradie-t-elle dans le bras gauche ou la mâchoire ?",
            "Avez-vous déjà eu ce type de douleur ?",
            "La douleur s'aggrave-t-elle à l'effort ?"
        ],
        "examens_proposes": [
            "ECG 12 dérivations en urgence",
            "Troponine (si analyseur disponible)",
            "Saturation en oxygène"
        ],
        "niveau_urgence": "eleve",
        "confiance": 0.0,
        "modele_utilise": f"{provider}/demo"
    }


def _extract_hf_text(payload: Any) -> str:
    """
    Extrait le texte de reponse depuis les formats possibles de l'API HF.
    """
    if isinstance(payload, dict):
        if "choices" in payload and isinstance(payload["choices"], list) and payload["choices"]:
            choice0 = payload["choices"][0]
            if isinstance(choice0, dict):
                message = choice0.get("message", {})
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(choice0.get("text"), str):
                    return choice0["text"]
        if "generated_text" in payload and isinstance(payload["generated_text"], str):
            return payload["generated_text"]
        if "error" in payload:
            raise RuntimeError(f"huggingface API error: {payload['error']}")
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
    raise RuntimeError(f"huggingface response format inattendu: {json.dumps(payload)[:500]}")


# ─── CAS TESTS SYNTHÉTIQUES ───────────────────────────────────────────────

CAS_TESTS = [
    {
        "id": "CAS_001",
        "description": "Infarctus du myocarde typique",
        "patient": {
            "age": 58, "sexe": "M", "taille_cm": 175, "poids_kg": 85,
            "symptomes": ["douleur_thoracique", "dyspnee", "nausees_vomissements"],
            "symptome_libre": "Douleur oppressive irradiant dans le bras gauche depuis 45 min",
            "tension_systolique": 145, "tension_diastolique": 92,
            "frequence_cardiaque": 98, "temperature": 37.1, "spo2": 94,
            "antecedents": ["HTA", "tabagisme 20 PA", "diabète type 2"],
            "traitements_en_cours": ["metformine 1g", "amlodipine 5mg"],
            "examens_realises": ["ecg"],
            "resultats_examens": {"ecg": "Sus-décalage ST en V1-V4, onde Q en V1-V2"}
        },
        "diagnostic_attendu": "Infarctus du myocarde antérieur",
        "urgence_attendue": "critique"
    },
    {
        "id": "CAS_002",
        "description": "Fibrillation auriculaire",
        "patient": {
            "age": 72, "sexe": "F", "taille_cm": 162, "poids_kg": 68,
            "symptomes": ["palpitations", "asthenie", "dyspnee"],
            "symptome_libre": "Palpitations irrégulières depuis hier soir",
            "tension_systolique": 132, "tension_diastolique": 78,
            "frequence_cardiaque": 118, "temperature": 36.8, "spo2": 97,
            "antecedents": ["HTA", "hypothyroïdie"],
            "traitements_en_cours": ["levothyroxine 75µg", "ramipril 5mg"],
            "examens_realises": ["ecg"],
            "resultats_examens": {"ecg": "Rythme irrégulier, absence d'onde P, trémulations de la ligne isoélectrique"}
        },
        "diagnostic_attendu": "Fibrillation auriculaire",
        "urgence_attendue": "eleve"
    },
    {
        "id": "CAS_003",
        "description": "Angine bactérienne simple",
        "patient": {
            "age": 24, "sexe": "F", "taille_cm": 168, "poids_kg": 60,
            "symptomes": ["fievre", "cephalees", "asthenie"],
            "symptome_libre": "Gorge très douloureuse depuis 2 jours, difficulté à avaler",
            "tension_systolique": 118, "tension_diastolique": 72,
            "frequence_cardiaque": 88, "temperature": 38.9, "spo2": 99,
            "antecedents": [],
            "traitements_en_cours": [],
            "examens_realises": [],
            "resultats_examens": {}
        },
        "diagnostic_attendu": "Angine bactérienne (streptococcique)",
        "urgence_attendue": "faible"
    }
]


if __name__ == "__main__":
    import asyncio

    async def test():
        print("=== Test du prompt engine ===\n")
        for cas in CAS_TESTS:
            print(f"--- {cas['id']} : {cas['description']} ---")
            prompt = build_prompt(cas["patient"])
            print("PROMPT USER (extrait) :")
            print(prompt["user"][:300], "...\n")
            print(f"Diagnostic attendu : {cas['diagnostic_attendu']}")
            print(f"Urgence attendue   : {cas['urgence_attendue']}\n")

    asyncio.run(test())
