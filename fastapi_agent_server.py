# /home/piyush/hackathon/AI-part/fastapi_agent_server.py
"""
Bhasha Setu - Agentic AI Pedagogical Tutor Microservice
Handles:
1. Anti-hallucination verified Bollywood song drills
2. Clause-scoped pronoun-verb register evaluation
3. Akshara-level Devanagari Levenshtein scoring
4. Autonomous 4-skill learner model state
5. Sub-second response fallback for live judging
"""
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Bhasha Setu AI Tutor", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 1. Linguistic Logic & Indic Akshara Tokenizer
# =====================================================================
TOKEN = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")
FORMAL_VERBS = {"हैं", "कीजिए", "बताइए", "आइए", "बैठिए", "दीजिए", "लीजिए", "पधारिए"}
INFORMAL_2P = {"हो", "है", "रहा", "रही", "करो", "जाओ", "बताओ", "आओ", "कर", "जा", "बता", "आ"}

AKSHARA_PATTERN = re.compile(
    r"[\u0904-\u0914\u0960\u0961]|"
    r"[\u0915-\u0939\u0958-\u095F](?:[\u094D][\u0915-\u0939\u0958-\u095F])*(?:[\u093E-\u094C\u0962\u0963]|[\u0901-\u0903])*"
)


def normalize_devanagari(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[।॥.,!?\u200c\u200d]", "", s).strip()
    return s.replace("\u0901", "\u0902")


def tokenize_aksharas(s: str) -> list[str]:
    norm_s = normalize_devanagari(s)
    matches = AKSHARA_PATTERN.findall(norm_s)
    return matches if matches else list(norm_s)


def akshara_distance(a: str, b: str) -> int:
    x, y = tokenize_aksharas(a), tokenize_aksharas(b)
    prev = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        cur = [i]
        for j, cy in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (cx != cy)))
        prev = cur
    return prev[-1]


def evaluate_register(text: str) -> dict:
    """Evaluates adherence to standard polite formal register on elders dialogues."""
    clauses = re.split(r"[?!।,.]", text)
    status = "NEUTRAL"
    for clause in clauses:
        tokens = set(TOKEN.findall(clause))
        if not tokens:
            continue

        # 1. Direct informal second-person pronoun with elder
        if tokens & {"तू"}:
            return {
                "status": "INFORMAL_FAUX_PAS",
                "score": 0.2,
                "feedback": "संकेत: बड़ों से बात करते समय 'आप' और आदरार्थक क्रिया ('हैं') का प्रयोग मानक माना जाता है।",
            }

        # 2. Check clause containing 'आप'
        if "आप" in tokens:
            if tokens & FORMAL_VERBS:
                status = "RESPECTFUL_PERFECT"
            elif tokens & INFORMAL_2P:
                return {
                    "status": "MISMATCHED_AGREEMENT",
                    "score": 0.5,
                    "feedback": "व्याकरण ध्यान: 'आप' के साथ 'हो' या 'है' के बजाय 'हैं' का प्रयोग होता है (जैसे 'आप कैसी हैं')।",
                }

    scores = {"RESPECTFUL_PERFECT": 1.0, "NEUTRAL": 0.6}
    feedback_map = {
        "RESPECTFUL_PERFECT": "उत्कृष्ट! शिष्ट और मानक आदरार्थक भाषा।",
        "NEUTRAL": "सटीक प्रयास! बड़ों के संवाद में 'जी' और 'आप' का प्रयोग भाषा में और मिठास लाता है।",
    }
    return {"status": status, "score": scores[status], "feedback": feedback_map[status]}


# =====================================================================
# 2. Autonomous Learner Model
# =====================================================================
@dataclass
class LearnerSkillModel:
    user_id: str
    skills: dict[str, float] = field(
        default_factory=lambda: {
            "akshara_orthography": 0.5,  # Devanagari typing & matra accuracy
            "honorific_register": 0.5,  # Formal pronoun-verb agreement with elders
            "speed_listening": 0.5,  # Street food rapid audio comprehension
            "vocal_confidence": 0.5,  # Speaking aloud cadence
        }
    )
    last_practiced: dict[str, float] = field(default_factory=dict)

    def select_next_goal(self) -> str:
        """Autonomously selects the next drill based on the lowest scoring skill."""
        now = time.time()
        effective_scores = {}
        for skill, score in self.skills.items():
            hours_since = (now - self.last_practiced.get(skill, now)) / 3600.0
            decay = min(0.3, hours_since * 0.01)
            effective_scores[skill] = score - decay
        weakest_skill = min(effective_scores, key=effective_scores.get)
        goal_mapping = {
            "akshara_orthography": "DRILL_BOLLYWOOD_LYRIC",
            "honorific_register": "DRILL_NANI_ETIQUETTE",
            "speed_listening": "DRILL_STREET_FOOD_RAPID",
            "vocal_confidence": "DRILL_CRICKET_COMMENTARY",
        }
        return goal_mapping[weakest_skill]

    def record_attempt(self, skill: str, success: bool, weight: float = 0.1):
        delta = weight if success else -weight
        self.skills[skill] = max(0.1, min(1.0, round(self.skills[skill] + delta, 2)))
        self.last_practiced[skill] = time.time()


LEARNER_MODELS: dict[str, LearnerSkillModel] = {}


def get_learner_model(user_id: str) -> LearnerSkillModel:
    if user_id not in LEARNER_MODELS:
        LEARNER_MODELS[user_id] = LearnerSkillModel(user_id=user_id)
    return LEARNER_MODELS[user_id]


# =====================================================================
# 3. Curated Anti-Hallucination Song Knowledge Base
# =====================================================================
CURATED_SONGS = {
    "दिल": {
        "song_id": "kabhi_kabhie_dil",
        "target_word": "दिल",
        "akshara_count": 2,
        "song_title": "Kabhi Kabhie Mere Dil Mein (1976)",
        "film": "Kabhi Kabhie",
        "singers": ["Mukesh", "Lata Mangeshkar"],
        "lyricist": "Sahir Ludhianvi",
        "canonical_line": "कभी कभी मेरे दिल में ख़याल आता है",
        "blanked_line": "कभी कभी मेरे [ ____ ] में ख़याल आता है",
        "translation": "Sometimes, in my heart, a thought arises",
        "phonetic_rhyme_hint": "Rhymes with English 'bill'; starts with 'द' (da)",
        "cultural_etymology": "In Hindustani poetry, 'दिल' denotes the seat of conscience and intuition.",
        "xp_reward": 20,
    },
    "तीखा": {
        "song_id": "street_food_teekha",
        "target_word": "तीखा",
        "akshara_count": 2,
        "song_title": "Pani Puri Counter Dialogue",
        "film": "Old Delhi Street Kitchen",
        "singers": ["Vendor Dialogue"],
        "lyricist": "Traditional",
        "canonical_line": "भैया, पानी पूरी थोड़ा तीखा और कम मीठा बनाना।",
        "blanked_line": "भैया, पानी पूरी थोड़ा [ ____ ] और कम मीठा बनाना।",
        "translation": "Brother, make the pani puri a little spicy and less sweet.",
        "phonetic_rhyme_hint": "Rhymes with 'Seekha'; starts with 'त' (ta)",
        "cultural_etymology": "In Indian culinary parlance, 'तीखा' specifically refers to chili heat, distinct from 'मसालेदार' (spiced).",
        "xp_reward": 40,
    },
}

# =====================================================================
# 4. Request / Response Pydantic Models
# =====================================================================
class EvaluateRequest(BaseModel):
    drill_type: str
    user_text: str
    expected_text: Optional[str] = None
    attempt_count: Optional[int] = 1
    user_id: Optional[str] = "usr_zen_pioneer_01"


class SongTutorRequest(BaseModel):
    target_word: str
    category: Optional[str] = "BOLLYWOOD"


class TutorTurnRequest(BaseModel):
    session_id: Optional[str] = "session_default"
    user_text: str
    context: Optional[str] = "DIWALI_NANI_BOSS"
    user_id: Optional[str] = "usr_zen_pioneer_01"


class LearnerGoalRequest(BaseModel):
    user_id: str


class LearnerAttemptRequest(BaseModel):
    user_id: str
    skill: str
    success: bool
    weight: Optional[float] = 0.1


# =====================================================================
# 5. API Endpoints
# =====================================================================
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "bhasha-setu-ai-tutor", "version": "3.2.0"}


@app.post("/api/agent/evaluate")
def evaluate_answer(req: EvaluateRequest):
    expected = req.expected_text or ("दिल" if "BOLLYWOOD" in req.drill_type.upper() else "तीखा")
    norm_user = normalize_devanagari(req.user_text)
    norm_exp = normalize_devanagari(expected)
    dist = akshara_distance(norm_user, norm_exp)
    is_correct = dist == 0

    # Anti-leak hint ladder logic
    if is_correct:
        feedback_hi = f"शानदार! बिल्कुल सही उत्तर: '{norm_exp}'।"
        feedback_en = "Splendid! Exact active recall match."
        awarded_xp = 30
    else:
        awarded_xp = 0
        if req.attempt_count == 1:
            feedback_hi = "करीब हैं! तुकांत संकेत: यह शब्द 'Bill' से मिलता-जुलता है।"
            feedback_en = "Close! Rhyme hint: Rhymes with English 'bill'."
        elif req.attempt_count == 2:
            feedback_hi = "दूसरा संकेत: यह भावना और अंतरात्मा का प्रतीक है।"
            feedback_en = "Second hint: Meaning relates to heart and intuition."
        else:
            feedback_hi = f"सही उत्तर '{norm_exp}' था। आइए दोबारा अभ्यास करें!"
            feedback_en = f"The target word was '{norm_exp}'."

    # Update autonomous learner model
    if req.user_id:
        model = get_learner_model(req.user_id)
        model.record_attempt("akshara_orthography", is_correct)

    return {
        "is_correct": is_correct,
        "score": 100 if is_correct else max(0, 100 - dist * 40),
        "akshara_distance": dist,
        "normalized_user_text": norm_user,
        "feedback_hindi": feedback_hi,
        "feedback_english": feedback_en,
        "xp_awarded": awarded_xp,
    }


@app.post("/api/agent/song-tutor")
def song_tutor(req: SongTutorRequest):
    record = CURATED_SONGS.get(req.target_word, CURATED_SONGS["दिल"])
    return record


@app.post("/api/agent/tutor/turn")
def tutor_turn(req: TutorTurnRequest):
    reg = evaluate_register(req.user_text)
    is_perfect = reg["status"] == "RESPECTFUL_PERFECT"

    if is_perfect:
        nani_reply = "जीते रहो बेटा! खूब खुश रहो। तुम्हारी पसंद के बेसन के लड्डू तैयार हैं! 🪔"
        nani_en = "Live long my child! Stay blessed. Your favorite besan laddus are ready!"
        xp = 50
        unlock = True
    elif reg["status"] == "MISMATCHED_AGREEMENT":
        nani_reply = "बेटा, 'आप' के साथ 'हैं' कहते हैं। लेकिन तुम खुश रहो और जल्दी घर आओ!"
        nani_en = "Child, we use 'Hain' with 'Aap'. But stay blessed and come home soon!"
        xp = 20
        unlock = False
    elif reg["status"] == "INFORMAL_FAUX_PAS":
        nani_reply = "बेटा, बड़ों को 'तू' नहीं, 'आप' कहना चाहिए। कोई बात नहीं, सीख जाओगे।"
        nani_en = "Child, we address elders as 'Aap', not 'Tu'. Never mind, you will learn."
        xp = 10
        unlock = False
    else:
        nani_reply = "जीते रहो बेटा! बड़ों से बात करते समय 'आप' और 'जी' कहना और अच्छा लगता है।"
        nani_en = "Stay blessed child! It sounds wonderful when you use 'Aap' and 'Ji' with elders."
        xp = 25
        unlock = False

    # Update autonomous learner model
    if req.user_id:
        model = get_learner_model(req.user_id)
        model.record_attempt("honorific_register", is_perfect)

    return {
        "status": reg["status"],
        "score": reg["score"],
        "register_feedback": reg["feedback"],
        "nani_reply_hindi": nani_reply,
        "nani_reply_english": nani_en,
        "xp_awarded": xp,
        "unlock_golden_diya": unlock,
    }


@app.post("/api/agent/learner/next-goal")
def learner_next_goal(req: LearnerGoalRequest):
    model = get_learner_model(req.user_id)
    goal = model.select_next_goal()
    return {
        "user_id": req.user_id,
        "next_goal": goal,
        "skill_scores": model.skills,
    }


@app.post("/api/agent/learner/attempt")
def learner_attempt(req: LearnerAttemptRequest):
    model = get_learner_model(req.user_id)
    model.record_attempt(req.skill, req.success, req.weight or 0.1)
    return {
        "user_id": req.user_id,
        "updated_skills": model.skills,
        "next_recommended_goal": model.select_next_goal(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_agent_server:app", host="0.0.0.0", port=8000, reload=True)

