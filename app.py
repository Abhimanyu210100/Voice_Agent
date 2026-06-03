import json
import os
import re
import time

import requests
import streamlit as st
from dotenv import load_dotenv

from prompt_builder import build_interview_prompt
from text_agent import chat
from voice_agent import VoiceInterviewSession

load_dotenv()

_LLM_URL = "https://llm-gateway.assemblyai.com/v1/chat/completions"


def generate_summary(messages: list[dict], job_title: str, interview_type: str) -> dict:
    transcript = "\n".join(
        f"{'Interviewer' if m['role'] == 'interviewer' else 'Candidate'}: {m['content']}"
        for m in messages
    )
    prompt = f"""You are an expert interview coach. Analyze this mock interview transcript \
for the role of "{job_title}" ({interview_type} interview).

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

    resp = requests.post(
        _LLM_URL,
        headers={
            "Authorization": os.environ["ASSEMBLYAI_API_KEY"],
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
        },
        timeout=45,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return json.loads(match.group())
    return {"summary": raw, "overallScore": None, "recommendation": None,
            "strengths": [], "improvements": []}

st.set_page_config(page_title="Mock Interviewer", page_icon="🎙", layout="centered")


# ── session state defaults ────────────────────────────────────────────────────

def _init() -> None:
    defaults: dict = {
        "stage": "setup",
        "system_prompt": "",
        "greeting": "",
        "mode": "text",
        "job_title": "",
        "interview_type": "",
        # display: [{"role": "interviewer"|"candidate", "content": str}]
        "messages": [],
        # LLM context for text mode: [{"role": "user"|"assistant", "content": str}]
        "text_history": [],
        "voice_session": None,
        "voice_active": False,
        "partial": "",           # in-progress candidate partial from voice
        "agent_speaking": False, # True while interviewer audio is playing
        "voice_first_agent": True,  # used to skip greeting duplication in voice
        "summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init()


# ── helpers ───────────────────────────────────────────────────────────────────

def reset_to_setup() -> None:
    if st.session_state.voice_session:
        st.session_state.voice_session.stop()
    for key in [
        "system_prompt", "greeting", "messages", "text_history",
        "voice_session", "voice_active", "partial", "voice_first_agent",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.stage = "setup"


def render_messages() -> None:
    for msg in st.session_state.messages:
        avatar = "🤵" if msg["role"] == "interviewer" else "🙋"
        with st.chat_message(msg["role"] if msg["role"] == "assistant" else "user",
                             avatar=avatar):
            st.markdown(msg["content"])


# ── SETUP PAGE ────────────────────────────────────────────────────────────────

if st.session_state.stage == "setup":
    st.title("Mock Interviewer")
    st.caption("Get interview-ready with a realistic AI interviewer tailored to your target role.")

    with st.form("setup_form"):
        st.subheader("The Role")
        job_title = st.text_input(
            "Job title *",
            placeholder="e.g. Senior Software Engineer",
        )
        job_description = st.text_area(
            "Job description *",
            height=160,
            placeholder="Paste the full job description here…",
        )

        st.subheader("The Interviewer")
        col1, col2 = st.columns(2)
        with col1:
            interviewer_title = st.text_input(
                "Interviewer's job title *",
                placeholder="e.g. Engineering Manager",
            )
        with col2:
            interviewer_company = st.text_input(
                "Company (optional)",
                placeholder="e.g. Acme Corp",
            )
        interviewer_bio = st.text_area(
            "Interviewer's background / LinkedIn summary (optional)",
            height=90,
            placeholder="Paste a short bio to shape the interviewer's personality and focus areas…",
        )

        st.subheader("Interview Type")
        interview_type = st.selectbox(
            "Select interview type",
            ["Behavioral", "Technical", "System Design", "HR Screen", "Case Study"],
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "Prepare Interview →", use_container_width=True, type="primary"
        )

    if submitted:
        if not job_title.strip() or not job_description.strip() or not interviewer_title.strip():
            st.error("Please fill in job title, job description, and interviewer title.")
        else:
            with st.spinner("Building your interview questions and persona…"):
                try:
                    system_prompt, greeting = build_interview_prompt(
                        job_title=job_title.strip(),
                        job_description=job_description.strip(),
                        interviewer_title=interviewer_title.strip(),
                        interviewer_company=interviewer_company.strip(),
                        interviewer_bio=interviewer_bio.strip(),
                        interview_type=interview_type,
                    )
                    st.session_state.system_prompt = system_prompt
                    st.session_state.greeting = greeting
                    st.session_state.job_title = job_title.strip()
                    st.session_state.interview_type = interview_type
                    st.session_state.stage = "interview"
                    st.session_state.messages = [
                        {"role": "interviewer", "content": greeting}
                    ]
                    st.session_state.text_history = [
                        {"role": "assistant", "content": greeting}
                    ]
                    st.session_state.voice_first_agent = True
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to build interview: {exc}")


# ── INTERVIEW PAGE ────────────────────────────────────────────────────────────

elif st.session_state.stage == "interview":
    # Header row
    col_title, col_reset = st.columns([5, 1])
    with col_title:
        st.title("Mock Interview")
    with col_reset:
        st.write("")
        if st.button("↩ Restart", use_container_width=True):
            reset_to_setup()
            st.rerun()

    # Mode toggle + End Interview
    toggle_col, end_col = st.columns([3, 1])
    with toggle_col:
        mode = st.radio(
            "Mode",
            ["Text", "Voice"],
            horizontal=True,
            index=0 if st.session_state.mode == "text" else 1,
        )
        st.session_state.mode = mode.lower()
    with end_col:
        st.write("")
        if st.button("📋 Get Feedback", use_container_width=True, type="primary",
                     disabled=len(st.session_state.messages) < 2):
            if st.session_state.voice_active and st.session_state.voice_session:
                st.session_state.voice_session.stop()
                st.session_state.voice_active = False
            with st.spinner("Generating your feedback…"):
                try:
                    st.session_state.summary = generate_summary(
                        st.session_state.messages,
                        st.session_state.job_title,
                        st.session_state.interview_type,
                    )
                except Exception as exc:
                    st.error(f"Failed to generate summary: {exc}")
                    st.session_state.summary = None
            if st.session_state.summary:
                st.session_state.stage = "summary"
                st.rerun()

    # Stop voice if switching to text
    if st.session_state.mode == "text" and st.session_state.voice_active:
        st.session_state.voice_session.stop()
        st.session_state.voice_active = False
        st.session_state.partial = ""
        st.session_state.agent_speaking = False

    st.divider()

    # Conversation transcript
    render_messages()

    # Live indicators: candidate partial and agent speaking (voice only)
    live_slot = st.empty()
    if st.session_state.agent_speaking:
        with live_slot.container():
            with st.chat_message("assistant", avatar="🤵"):
                st.markdown("*…*")
    elif st.session_state.partial:
        with live_slot.container():
            with st.chat_message("user", avatar="🙋"):
                st.markdown(f"*{st.session_state.partial}*")

    # ── TEXT MODE ─────────────────────────────────────────────────────────────
    if st.session_state.mode == "text":
        user_input = st.chat_input("Your answer…")
        if user_input:
            with st.spinner(""):
                try:
                    reply = chat(
                        system_prompt=st.session_state.system_prompt,
                        history=st.session_state.text_history,
                        user_message=user_input,
                    )
                except Exception as exc:
                    reply = f"[Error contacting LLM: {exc}]"

            st.session_state.messages.append({"role": "candidate", "content": user_input})
            st.session_state.messages.append({"role": "interviewer", "content": reply})
            st.session_state.text_history.append({"role": "user", "content": user_input})
            st.session_state.text_history.append({"role": "assistant", "content": reply})
            st.rerun()

    # ── VOICE MODE ────────────────────────────────────────────────────────────
    else:
        voice_col, status_col = st.columns([1, 2])

        with voice_col:
            if not st.session_state.voice_active:
                if st.button("🎙 Start", use_container_width=True, type="primary"):
                    session = VoiceInterviewSession(
                        system_prompt=st.session_state.system_prompt,
                        greeting=st.session_state.greeting,
                    )
                    session.start()
                    st.session_state.voice_session = session
                    st.session_state.voice_active = True
                    st.session_state.voice_first_agent = True
                    st.rerun()
            else:
                if st.button("⏹ Stop", use_container_width=True):
                    st.session_state.voice_session.stop()
                    st.session_state.voice_active = False
                    st.session_state.partial = ""
                    st.session_state.agent_speaking = False
                    st.rerun()

        with status_col:
            if st.session_state.voice_active:
                st.info("🔴 Listening — speak naturally, the interviewer will respond.")
            else:
                st.info("Press Start to begin the voice interview.")

        # Drain the events queue and update conversation
        if st.session_state.voice_active and st.session_state.voice_session:
            session = st.session_state.voice_session
            changed = False

            while not session.events.empty():
                ev = session.events.get_nowait()
                etype = ev["type"]

                if etype == "user":
                    if ev["final"]:
                        if ev["text"].strip():
                            st.session_state.messages.append(
                                {"role": "candidate", "content": ev["text"]}
                            )
                        st.session_state.partial = ""
                    else:
                        st.session_state.partial = ev["text"]
                    changed = True

                elif etype == "agent_speaking":
                    st.session_state.agent_speaking = True
                    changed = True

                elif etype == "agent":
                    text = ev["text"].strip()
                    st.session_state.agent_speaking = False
                    if not text:
                        continue
                    # Skip the first agent transcript — it's the greeting we already display
                    if st.session_state.voice_first_agent:
                        st.session_state.voice_first_agent = False
                        continue
                    st.session_state.messages.append(
                        {"role": "interviewer", "content": text}
                    )
                    changed = True

                elif etype == "error":
                    st.error(f"Voice session error: {ev['text']}")
                    st.session_state.voice_active = False
                    st.session_state.agent_speaking = False
                    changed = True

                elif etype == "done":
                    st.session_state.voice_active = False
                    st.session_state.agent_speaking = False
                    changed = True

            # Keep polling while active; rerun immediately if something changed
            if st.session_state.voice_active:
                time.sleep(0.3)
                st.rerun()
            elif changed:
                st.rerun()


# ── SUMMARY PAGE ──────────────────────────────────────────────────────────────

elif st.session_state.stage == "summary":
    data: dict = st.session_state.summary or {}

    col_title, col_back = st.columns([5, 1])
    with col_title:
        st.title("Interview Feedback")
        st.caption(f"{st.session_state.job_title} · {st.session_state.interview_type}")
    with col_back:
        st.write("")
        if st.button("↩ Restart", use_container_width=True):
            reset_to_setup()
            st.rerun()

    st.divider()

    # Score + recommendation
    score = data.get("overallScore")
    rec = data.get("recommendation", "")
    rec_color = {
        "Strong Yes": "green", "Yes": "green",
        "Maybe": "orange", "No": "red",
    }.get(rec, "gray")

    score_col, rec_col = st.columns(2)
    with score_col:
        st.metric("Overall Score", f"{score}/10" if score else "—")
    with rec_col:
        if rec:
            st.markdown(f"**Hiring Signal**")
            st.markdown(f":{rec_color}[**{rec}**]")

    st.divider()

    # Strengths
    strengths = data.get("strengths", [])
    if strengths:
        st.subheader("Strengths")
        for s in strengths:
            st.success(f"✓ {s}")

    # Improvements
    improvements = data.get("improvements", [])
    if improvements:
        st.subheader("Areas to Improve")
        for imp in improvements:
            st.warning(f"△ {imp}")

    # Overall assessment
    summary_text = data.get("summary", "")
    if summary_text:
        st.subheader("Overall Assessment")
        st.markdown(summary_text)

    st.divider()

    # Copy to clipboard button
    if st.button("📋 Copy Summary to Clipboard", use_container_width=True):
        lines = [
            f"Interview Feedback — {st.session_state.job_title} ({st.session_state.interview_type})",
            f"Score: {score}/10  |  Recommendation: {rec}",
            "",
            "STRENGTHS",
            *[f"• {s}" for s in strengths],
            "",
            "AREAS TO IMPROVE",
            *[f"• {i}" for i in improvements],
            "",
            "ASSESSMENT",
            summary_text,
        ]
        st.code("\n".join(lines), language=None)
        st.caption("Select all and copy the text above.")
