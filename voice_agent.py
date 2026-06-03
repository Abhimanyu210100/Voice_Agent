import asyncio
import base64
import json
import os
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

load_dotenv()

_URL = "wss://agents.assemblyai.com/v1/ws"
_SAMPLE_RATE = 24_000
_CHUNK_MS = 50
_CHUNK_FRAMES = int(_SAMPLE_RATE * _CHUNK_MS / 1000)  # 1200 frames per chunk


class VoiceInterviewSession:
    """
    Manages one Voice Agent API session in a background thread.

    Events queue entries:
        {"type": "user",  "text": str, "final": bool}   — candidate speech
        {"type": "agent", "text": str, "final": True}   — interviewer transcript
        {"type": "error", "text": str, "final": True}
        {"type": "done",  "text": "",  "final": True}   — session ended cleanly
    """

    def __init__(self, system_prompt: str, greeting: str):
        self.system_prompt = system_prompt
        self.greeting = greeting
        self.events: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=8)

    # ── internals ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            asyncio.run(self._session())
        except Exception as exc:
            self.events.put({"type": "error", "text": str(exc), "final": True})

    async def _session(self) -> None:
        headers = {"Authorization": f"Bearer {os.environ['ASSEMBLYAI_API_KEY']}"}
        mic_q: asyncio.Queue = asyncio.Queue()
        ready = asyncio.Event()
        loop = asyncio.get_running_loop()

        def mic_callback(indata, _frames, _time, _status) -> None:
            if ready.is_set() and not self._stop.is_set():
                loop.call_soon_threadsafe(mic_q.put_nowait, bytes(indata))

        async def pump_mic(ws) -> None:
            while not self._stop.is_set():
                try:
                    chunk = await asyncio.wait_for(mic_q.get(), timeout=0.3)
                    await ws.send(json.dumps({
                        "type": "input.audio",
                        "audio": base64.b64encode(chunk).decode(),
                    }))
                except asyncio.TimeoutError:
                    continue

        async with websockets.connect(_URL, additional_headers=headers) as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "system_prompt": self.system_prompt,
                    "greeting": self.greeting,
                    "input": {
                        "format": {"encoding": "audio/pcm"},
                        "turn_detection": {
                            "vad_threshold": 0.5,
                            "min_silence": 300,
                            "max_silence": 1200,
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "voice": "james",
                        "format": {"encoding": "audio/pcm"},
                    },
                },
            }))

            mic_stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK_FRAMES,
                callback=mic_callback,
            )
            speaker = sd.OutputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            with mic_stream, speaker:
                pump_task = asyncio.create_task(pump_mic(ws))

                async for raw in ws:
                    if self._stop.is_set():
                        break

                    ev = json.loads(raw)
                    t = ev.get("type", "")

                    if t == "session.ready":
                        ready.set()

                    elif t == "transcript.user.delta":
                        self.events.put({
                            "type": "user",
                            "text": ev.get("transcript", ""),
                            "final": False,
                        })

                    elif t == "transcript.user":
                        self.events.put({
                            "type": "user",
                            "text": ev.get("transcript", ""),
                            "final": True,
                        })

                    elif t == "reply.started":
                        # Agent has started speaking — show indicator immediately
                        self.events.put({
                            "type": "agent_speaking",
                            "text": "",
                            "final": False,
                        })

                    elif t == "transcript.agent":
                        # Full agent transcript arrives while/after audio plays
                        self.events.put({
                            "type": "agent",
                            "text": ev.get("transcript", ""),
                            "final": True,
                        })

                    elif t == "reply.audio":
                        arr = np.frombuffer(
                            base64.b64decode(ev["data"]), dtype=np.int16
                        )
                        speaker.write(arr)

                    elif t == "reply.done":
                        if ev.get("status") == "interrupted":
                            speaker.abort()
                            speaker.start()

                pump_task.cancel()
                try:
                    await pump_task
                except asyncio.CancelledError:
                    pass

        self.events.put({"type": "done", "text": "", "final": True})
