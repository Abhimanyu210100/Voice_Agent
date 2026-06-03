import requests

_OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
_MODEL = "llama3.1:8b"


def _llm(system: str, user: str, max_tokens: int = 800) -> str:
    resp = requests.post(
        _OLLAMA_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def build_interview_prompt(
    job_title: str,
    job_description: str,
    interviewer_title: str,
    interviewer_company: str,
    interviewer_bio: str,
    interview_type: str,
) -> "tuple[str, str]":
    """
    Returns (system_prompt, greeting).
    Makes two LLM Gateway calls: one for questions, one for the greeting.
    """
    questions_raw = _llm(
        system=(
            "You are an expert interviewer. Generate a numbered list of exactly 7 "
            "relevant interview questions for the given role and interview type. "
            "Return ONLY the numbered list — no intro, no commentary."
        ),
        user=(
            f"Interview type: {interview_type}\n"
            f"Job title: {job_title}\n"
            f"Job description:\n{job_description[:600]}"
        ),
        max_tokens=600,
    )

    persona_line = interviewer_title
    if interviewer_company:
        persona_line += f" at {interviewer_company}"

    greeting = _llm(
        system=(
            "Write a natural, brief (1–2 sentence) opening line that a real interviewer "
            "would say to kick off the interview. Return ONLY the greeting text."
        ),
        user=(
            f"Interviewer: {persona_line}. "
            f"Interview type: {interview_type}. "
            f"Role being interviewed for: {job_title}."
        ),
        max_tokens=80,
    )

    bio_section = f"\nBackground: {interviewer_bio}" if interviewer_bio else ""

    system_prompt = f"""You are conducting a mock {interview_type} interview for the role of {job_title}.

INTERVIEWER PERSONA:
You are {persona_line}.{bio_section}
Stay in character throughout the entire conversation. Do not break character or reveal you are an AI.

ROLE BEING INTERVIEWED FOR:
{job_description[:800]}

PREPARED QUESTIONS (use as a guide — adapt naturally based on the candidate's responses):
{questions_raw}

HOW TO CONDUCT THE INTERVIEW:
- Ask exactly ONE question at a time. Wait for the candidate's full response before proceeding.
- After EACH candidate answer, give brief constructive feedback before moving on: acknowledge one specific strength in their answer, then name one concrete improvement. Keep feedback to 2–3 sentences. Start with "Good answer —" or "Thanks for that —".
- React naturally to answers — ask a brief follow-up if something is interesting or unclear, then give feedback before the next question.
- Be professional and encouraging, but hold realistic standards.
- After all questions have been covered, deliver an overall interview summary: 3 key strengths observed, 2 areas to improve, and a clear hiring signal ("I'd recommend moving forward", "It's a borderline call", or "I'd pass at this stage").
- Keep your spoken responses concise and conversational — you are talking, not writing.
"""

    return system_prompt, greeting
