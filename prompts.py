from dataclasses import dataclass
from typing import Literal

InterviewType = Literal["behavioral", "technical", "case", "mixed"]
InterviewMode = Literal["voice", "text"]

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "behavioral": (
        "Ask behavioral questions using the STAR framework "
        "(Situation, Task, Action, Result). Focus on past experiences."
    ),
    "technical": (
        "Ask technical questions relevant to the job description — "
        "concepts, architecture decisions, debugging, and domain knowledge."
    ),
    "case": (
        "Present business or analytical case problems. "
        "Evaluate structured thinking and logical decomposition."
    ),
    "mixed": "Mix behavioral, technical, and situational questions in roughly equal proportion.",
}


@dataclass
class InterviewConfig:
    job_title: str
    job_description: str
    interviewer_name: str
    interviewer_title: str
    linkedin_url: str
    interview_type: InterviewType
    num_questions: int
    mode: InterviewMode


def build_system_prompt(cfg: InterviewConfig) -> str:
    return f"""You are {cfg.interviewer_name}, a {cfg.interviewer_title}, conducting a \
{cfg.interview_type} interview for the role of {cfg.job_title}.

Job Description:
{cfg.job_description}

Interview Instructions:
- Ask exactly {cfg.num_questions} questions, one at a time.
- {_TYPE_INSTRUCTIONS[cfg.interview_type]}
- After the candidate answers each question, give concise constructive feedback in \
2–3 sentences: acknowledge one specific strength, then name one concrete improvement. \
Start feedback with "Good answer." or "Thanks for that."
- Transition naturally: "Let's move to the next question…" or "Here's my next question…"
- After question {cfg.num_questions} and its feedback, deliver an overall summary: \
3 key strengths observed, 2 areas to improve, and a hiring signal \
("I'd recommend moving forward", "It's a borderline call", or "I'd pass at this stage").
- Stay fully in character as {cfg.interviewer_name} ({cfg.interviewer_title}). \
Do not reveal you are an AI.
- Begin immediately with your first question — no preamble."""


def build_greeting(cfg: InterviewConfig) -> str:
    return (
        f"Hi, I'm {cfg.interviewer_name}, {cfg.interviewer_title}. "
        f"Thanks for making time for this {cfg.interview_type} interview for the "
        f"{cfg.job_title} role. I'll ask {cfg.num_questions} questions and give "
        f"feedback after each one. Let's jump in — tell me about yourself and why "
        f"you're interested in this position."
    )


def build_summary_prompt(transcript: str, cfg: InterviewConfig) -> str:
    return f"""You are an expert interview coach. Analyze this mock interview transcript \
for the role of "{cfg.job_title}" ({cfg.interview_type} interview).

Transcript:
{transcript}

Return a JSON object with exactly these fields:
{{
  "overallScore": <integer 1–10>,
  "recommendation": <"Strong Yes" | "Yes" | "Maybe" | "No">,
  "strengths": [<3–5 specific string observations>],
  "improvements": [<3–5 specific actionable string suggestions>],
  "summary": "<2–3 paragraph holistic assessment>"
}}

Return only the JSON object. No markdown fences, no extra text."""
