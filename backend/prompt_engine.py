"""
EXOFIT — Prompt Engine
Construit le prompt structuré à 3 blocs et appelle l'API LLM.
Testable sans données réelles grâce aux cas synthétiques.
"""

import json
import httpx
import os
import re
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


def _google_generate_url() -> str:
    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

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
        "url": _google_generate_url(),
        "model": os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        "header_key": "x-goog-api-key",
        "header_prefix": "",
        "env_key": "GOOGLE_API_KEY",
    },
    "huggingface": {
        "url": "https://router.huggingface.co/v1/chat/completions",
        "model": os.getenv("HUGGINGFACE_MODEL", "deepseek-ai/DeepSeek-V3-0324"),
        "header_key": "Authorization",
        "header_prefix": "Bearer ",
        "env_key": "HUGGINGFACE_API_KEY",
    },
    "qwen": {
        "url": "https://router.huggingface.co/v1/chat/completions",
        "model": os.getenv("QW_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "header_key": "Authorization",
        "header_prefix": "Bearer ",
        "env_key": "QW_TOKEN",
    },
    "deepseek": {
        "url": "https://router.huggingface.co/v1/chat/completions",
        "model": os.getenv("DS_MODEL", "deepseek-ai/DeepSeek-V3-0324"),
        "header_key": "Authorization",
        "header_prefix": "Bearer ",
        "env_key": "DS_TOKEN",
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        "header_key": "Authorization",
        "header_prefix": "Bearer ",
        "env_key": "MISTRAL_API_KEY",
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

SYSTEM_PROMPT = """Tu es un assistant medical expert utilise dans un contexte de telemedecine.
Le resultat est destine a orienter la prise en charge immediate d'un patient, a partir des informations recueillies par une infirmiere.
Tu as acces au materiel suivant : tensiometre, thermometre, stethoscope, ECG 6 derivations, analyseur sanguin portable.
Ton role est de proposer une hypothese diagnostique, des diagnostics differentiels, des questions utiles, des examens a envisager, et une conduite a tenir immediate orientee patient.
Tu ne poses PAS de diagnostic definitif.
Tu dois etre prudent dans l'estimation des pourcentages:
- diagnostic_credibilite = plausibilite clinique du diagnostic principal a partir des seules donnees presentes
- confiance = fiabilite globale de la sortie, donc depend de la qualite, coherence et completude des donnees
- en cas de donnees incompletes, ambiguës ou peu objectives, garde des scores bas et prudents
- n'utilise des scores eleves que si plusieurs donnees concordantes et objectives soutiennent l'hypothese
Si tu detectes un signe de gravite immediate (douleur thoracique + dyspnee + ECG anormal, trouble de conscience, deficit neurologique progressif, etc.), indique-le clairement avec le niveau d'urgence CRITIQUE.
Reponds UNIQUEMENT en JSON valide selon le format demande. Pas de texte avant ou apres le JSON."""

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
  "diagnostic_credibilite": 0.0,
  "diagnostic_justification": "string â€” pourquoi cette hypothese est retenue",
  "diagnostics_differentiels": [
    {{
      "diagnostic": "string",
      "credibilite": 0.0,
      "justification": "string â€” pourquoi cette hypothese reste plausible"
    }}
  ],
  "questions_complementaires": ["string", "string", "string"],
  "examens_proposes": ["string", "string"],
  "traitements_proposes": ["string", "string"],
  "niveau_urgence": "faible|modere|eleve|critique",
  "confiance": 0.0,
  "modele_utilise": "string"
}}

Contraintes :
- Si des examens sont manquants (ex: ECG non réalisé), mentionne-le dans les examens_proposes
- Le champ traitements_proposes correspond a une conduite a tenir immediate orientee patient, pas a une recommandation entre professionnels
- Ecris des actions concretes et directement utiles : repos, surveillance, eviter de conduire, appeler le 15, aller aux urgences, consultation rapide, etc.
- N'ecris pas "evaluation par un medecin", "avis specialise", "bilan par neurologue" ou formulations similaires dans traitements_proposes
- Propose uniquement des traitements/initiations compatibles avec un contexte infirmier/telemedecine, sans prescription definitive
- Si les donnees objectives sont insuffisantes, formule l'hypothese de maniere prudente: "suspicion de", "compatible avec", "a confirmer"
- diagnostic_credibilite represente la plausibilite clinique de l'hypothese principale, pas une certitude
- confiance represente la fiabilite globale de la sortie et doit rester proche de la qualite des donnees disponibles
- N'attribue pas une diagnostic_credibilite > 0.50 sans constantes ET au moins un examen objectif; n'attribue pas une diagnostic_credibilite > 0.35 si constantes et examens sont absents
- N'attribue pas une confiance > 0.60 sans constantes ou examens objectifs; n'attribue pas une confiance > 0.45 si constantes et examens sont absents
- La confiance ne doit pas depasser la diagnostic_credibilite de plus de 0.10
- Les diagnostics differenciels doivent etre de vraies hypotheses alternatives, pas des reformulations du diagnostic principal
- Chaque diagnostic differentiel doit avoir une credibilite strictement inferieure a celle du diagnostic principal, avec un ecart minimal de 0.05
- La somme des credibilites des diagnostics differenciels ne doit pas depasser 0.60
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
            raise RuntimeError(f"{provider} API error ({response.status_code}): {response.text}") from exc
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
    result = _parse_llm_json(text)
    result = _normalize_diagnostic_explanations(result)
    result = _enforce_diagnostic_complementarity(result)
    result = _apply_result_guards(result, prompt)
    result = _calibrate_diagnostic_credibility(result, prompt)
    result = _normalize_action_plan(result)
    result["modele_utilise"] = f"{provider}/{config['model']}"
    return result


def _demo_response(provider: str) -> dict:
    """Réponse de démonstration quand aucune API key n'est configurée."""
    return {
        "diagnostic_preliminaire": "MODE DÉMO — Configurez une clé API dans le fichier .env",
        "diagnostic_credibilite": 0.35,
        "diagnostic_justification": "Hypothese de demonstration generee sans interpretation clinique reelle par un modele actif.",
        "diagnostics_differentiels": [
            {
                "diagnostic": "Douleur thoracique d'origine coronarienne",
                "credibilite": 0.30,
                "justification": "Hypothese frequente a eliminer si douleur thoracique rapportee."
            },
            {
                "diagnostic": "Syndrome coronarien aigu",
                "credibilite": 0.22,
                "justification": "Differentiel important mais non confirme sans donnees objectives."
            },
            {
                "diagnostic": "Pericardite aigue",
                "credibilite": 0.15,
                "justification": "Peut expliquer une douleur thoracique atypique ou inflammatoire."
            }
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
        "traitements_proposes": [
            "Surveillance clinique rapprochee",
            "Mise au repos et reevaluation reguliere",
            "Orientation rapide selon le niveau d'urgence et les signes de gravite"
        ],
        "niveau_urgence": "eleve",
        "confiance": 0.0,
        "modele_utilise": f"{provider}/demo"
    }


def _extract_hf_text(payload: Any) -> str:
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


def _parse_llm_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Le modele a retourne une reponse vide")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"Reponse du modele non JSON: {text[:300]}")


def _apply_result_guards(result: dict, prompt: dict) -> dict:
    user_text = (prompt.get("user") or "").lower()
    no_vitals = "non mesur" in user_text
    no_results = "non disponibles" in user_text
    no_exams = "aucun r" in user_text

    confidence = float(result.get("confiance", 0.0) or 0.0)
    if no_vitals and no_results:
        confidence = min(confidence, 0.45)
    elif no_vitals or no_results or no_exams:
        confidence = min(confidence, 0.60)
    result["confiance"] = max(0.0, min(confidence, 1.0))

    diagnosis = str(result.get("diagnostic_preliminaire", "") or "").strip()
    cautious_markers = ("suspicion", "probable", "compatible", "a confirmer", "à confirmer")
    if diagnosis and result["confiance"] <= 0.60 and not any(marker in diagnosis.lower() for marker in cautious_markers):
        result["diagnostic_preliminaire"] = f"Suspicion de {diagnosis[0].lower() + diagnosis[1:]}" if len(diagnosis) > 1 else diagnosis

    return result


def _normalize_diagnostic_explanations(result: dict) -> dict:
    result["diagnostic_credibilite"] = max(
        0.0,
        min(float(result.get("diagnostic_credibilite", result.get("confiance", 0.0)) or 0.0), 1.0),
    )
    result["diagnostic_justification"] = str(
        result.get("diagnostic_justification")
        or "Hypothese retenue sur la base des donnees cliniques actuellement disponibles."
    ).strip()

    normalized_differentials = []
    for item in result.get("diagnostics_differentiels") or []:
        if isinstance(item, dict):
            diagnostic = str(item.get("diagnostic") or "").strip()
            credibilite = max(0.0, min(float(item.get("credibilite", 0.0) or 0.0), 1.0))
            justification = str(item.get("justification") or "Hypothese differencielle a discuter selon les donnees disponibles.").strip()
        else:
            diagnostic = str(item).strip()
            credibilite = 0.0
            justification = "Hypothese differencielle a discuter selon les donnees disponibles."

        if diagnostic:
            normalized_differentials.append(
                {
                    "diagnostic": diagnostic,
                    "credibilite": credibilite,
                    "justification": justification,
                }
            )

    result["diagnostics_differentiels"] = normalized_differentials[:3]
    return result


def _canonical_diagnostic_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _enforce_diagnostic_complementarity(result: dict) -> dict:
    main = str(result.get("diagnostic_preliminaire") or "").strip()
    main_key = _canonical_diagnostic_label(main)
    seen = {main_key} if main_key else set()

    filtered = []
    for item in result.get("diagnostics_differentiels") or []:
        diagnostic = str(item.get("diagnostic") or "").strip()
        if not diagnostic:
            continue
        key = _canonical_diagnostic_label(diagnostic)
        if not key or key in seen:
            continue
        seen.add(key)
        filtered.append(item)

    filtered.sort(key=lambda x: float(x.get("credibilite", 0.0) or 0.0), reverse=True)
    result["diagnostics_differentiels"] = filtered[:3]
    return result


def _estimate_credibility_cap(prompt: dict) -> float:
    user_text = str(prompt.get("user") or "")
    lower = user_text.lower()
    points = 0.0

    if "aucun sympt" not in lower:
        points += 1.0
    if "non mesur" not in lower:
        points += 1.5
    if "aucun r" not in lower:
        points += 0.5
    if "non disponibles" not in lower:
        points += 2.0
    if "aucun renseign" not in lower:
        points += 0.5

    if any(token in lower for token in ["ecg", "troponine", "crp", "nfs", "inr", "creatin", "créatin", "ta :", "fc :", "spo2", "temp"]):
        points += 0.5

    cap = min(0.82, 0.12 + points * 0.10)

    no_vitals = "non mesur" in lower
    no_results = "non disponibles" in lower
    no_exams = "aucun r" in lower
    if no_vitals and no_results and no_exams:
        cap = min(cap, 0.28)
    elif no_vitals and no_results:
        cap = min(cap, 0.35)
    elif no_vitals or no_results:
        cap = min(cap, 0.50)

    return round(max(0.12, cap), 3)


def _calibrate_diagnostic_credibility(result: dict, prompt: dict) -> dict:
    cap = _estimate_credibility_cap(prompt)
    confidence = max(0.0, min(float(result.get("confiance", 0.0) or 0.0), 1.0))
    requested = max(0.0, min(float(result.get("diagnostic_credibilite", confidence) or 0.0), 1.0))

    lower_bound = max(0.12, min(cap, confidence - 0.08))
    upper_bound = min(cap, confidence + 0.10)
    calibrated_main = min(requested, upper_bound)
    calibrated_main = max(lower_bound, calibrated_main)
    result["diagnostic_credibilite"] = round(calibrated_main, 3)

    calibrated_confidence = min(confidence, result["diagnostic_credibilite"] + 0.10, cap + 0.08)
    result["confiance"] = round(max(0.0, calibrated_confidence), 3)

    current = result.get("diagnostics_differentiels") or []
    calibrated = []
    previous_cap = result["diagnostic_credibilite"]
    remaining_total = 0.60
    for index, item in enumerate(current):
        item_cap = max(
            0.05,
            min(
                cap - 0.12 - index * 0.08,
                previous_cap - 0.06,
                result["diagnostic_credibilite"] - 0.06,
                remaining_total,
            ),
        )
        requested_diff = max(0.0, min(float(item.get("credibilite", 0.0) or 0.0), 1.0))
        cred = min(requested_diff, item_cap)
        calibrated_item = dict(item)
        calibrated_item["credibilite"] = round(max(0.05, cred), 3)
        calibrated.append(calibrated_item)
        previous_cap = calibrated_item["credibilite"]
        remaining_total = round(max(0.0, remaining_total - calibrated_item["credibilite"]), 3)

    result["diagnostics_differentiels"] = calibrated
    return result


def _normalize_action_plan(result: dict) -> dict:
    items = result.get("traitements_proposes") or []
    urgence = str(result.get("niveau_urgence", "") or "").lower()
    normalized: list[str] = []

    for item in items:
        text = str(item).strip()
        lower = text.lower()
        if not text:
            continue

        if any(token in lower for token in [
            "par un medecin",
            "par un médecin",
            "consulter un medecin",
            "consulter un médecin",
            "voir un medecin",
            "voir un médecin",
            "confirmation et traitement",
            "avis medical",
            "avis médical",
            "avis specialise",
            "avis spécialisé",
            "neurolog",
            "cardiolog",
            "evaluation urgente",
            "évaluation urgente",
        ]):
            if urgence in {"critique", "eleve"}:
                normalized.append("Orientation immediate vers les urgences ou appel au 15")
            elif urgence == "modere":
                normalized.append("Poursuivre l'evaluation clinique sur place et organiser une reevaluation medicale le jour meme")
            else:
                normalized.append("Surveillance clinique et consultation medicale programmee rapidement")
            continue

        if "urgences" in lower or "appel au 15" in lower:
            normalized.append("Orientation immediate vers les urgences ou appel au 15")
            continue

        normalized.append(text)

    if not normalized:
        normalized = ["Surveillance clinique rapprochee", "Reevaluation medicale rapide si aggravation"]

    seen = set()
    deduped = []
    for item in normalized:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    result["traitements_proposes"] = deduped[:3]
    return result


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
