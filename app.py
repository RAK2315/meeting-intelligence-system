from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import json, asyncio, os, shutil, sqlite3
from datetime import datetime
from agents import (
    PIPELINE_GRAPH, resume_after_approval,
    get_google_creds, audit_log, init_db,
    transcribe_audio, extract_text_from_pdf,
    get_all_tasks, update_task_status, DB_PATH,
    detect_stalled_tasks, mark_tasks_stall_flagged,
    post_stall_alert_to_slack,
    node_extractor, node_action_writer, node_task_tracker,
    node_calendar, node_notion
)
from googleapiclient.discovery import build

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
init_db()

pipeline_states = {}

class TranscriptRequest(BaseModel):
    transcript: str
    meeting_id: str = None

class ApprovalRequest(BaseModel):
    meeting_id: str
    approved_emails: list
    clarifications: dict = {}

class TaskStatusUpdate(BaseModel):
    task_id: int
    status: str

class ClarificationUpdate(BaseModel):
    meeting_id: str
    task_key: str
    owner: str
    deadline: str

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/run")
async def run_pipeline_stream(req: TranscriptRequest):
    async def generate():
        audit_log.clear()
        meeting_id = req.meeting_id or f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        creds = get_google_creds()
        gmail = build('gmail', 'v1', credentials=creds)
        cal = build('calendar', 'v3', credentials=creds)

        def send(event, data):
            return f"data: {json.dumps({'event': event, 'data': data})}\n\n"

        initial_state = {
            "transcript": req.transcript,
            "meeting_id": meeting_id,
            "extracted": None,
            "emails": None,
            "approved_emails": None,
            "escalations": None,
            "calendar_events": None,
            "notion_tasks": None,
            "report": None,
            "error_count": 0,
            "awaiting_approval": False,
            "needs_clarification": [],
            "gmail_service": gmail,
            "calendar_service": cal
        }

        yield send("status", "🔍 Agent 1 [Extractor]: Parsing transcript (llama-70b)...")
        await asyncio.sleep(0.2)

        def run_to_approval(state):
            state = node_extractor(state)
            if not state['extracted'].get('tasks'):
                state['emails'] = []
                state['awaiting_approval'] = False
                return state
            state = node_action_writer(state)
            state = node_task_tracker(state)
            state = node_calendar(state)
            state = node_notion(state)
            return state

        final_state = await asyncio.to_thread(run_to_approval, initial_state)

        if final_state is None:
            yield send("error", "Pipeline failed")
            return

        pipeline_states[meeting_id] = final_state

        yield send("extracted", final_state['extracted'])
        yield send("escalations", final_state.get('escalations', []))
        yield send("calendar_events", final_state.get('calendar_events', []))
        yield send("notion_tasks", final_state.get('notion_tasks', []))
        yield send("needs_clarification", final_state.get('needs_clarification', []))
        yield send("emails_pending", {
            "meeting_id": meeting_id,
            "emails": final_state.get('emails', [])
        })

        # If there are tasks needing clarification, flag it
        if final_state.get('needs_clarification'):
            yield send("status", f"⚠️ {len(final_state['needs_clarification'])} task(s) need clarification")
            yield send("clarification_required", {
                "meeting_id": meeting_id,
                "tasks": final_state['needs_clarification']
            })

        yield send("status", "⏸ Awaiting human approval for emails...")
        yield send("awaiting_approval", {"meeting_id": meeting_id})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/approve")
async def approve_emails(req: ApprovalRequest):
    state = pipeline_states.get(req.meeting_id)
    if not state:
        return JSONResponse({"error": "Pipeline state not found"}, status_code=404)
    final_state = await asyncio.to_thread(
        resume_after_approval, state, req.approved_emails, req.clarifications
    )
    pipeline_states.pop(req.meeting_id, None)
    return JSONResponse({
        "report": final_state['report'],
        "escalations": final_state.get('escalations', []),
        "calendar_events": final_state.get('calendar_events', []),
        "notion_tasks": final_state.get('notion_tasks', []),
        "draft_ids": [e['to'] for e in req.approved_emails]
    })


@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        transcript = transcribe_audio(path)
        os.remove(path)
        return JSONResponse({"transcript": transcript, "words": len(transcript.split())})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        text = extract_text_from_pdf(path)
        os.remove(path)
        return JSONResponse({"transcript": text, "chars": len(text)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/tasks")
def get_tasks():
    return JSONResponse(get_all_tasks())


@app.post("/tasks/update")
def update_task(req: TaskStatusUpdate):
    update_task_status(req.task_id, req.status)
    return JSONResponse({"ok": True})


@app.post("/tasks/clear")
def clear_all_tasks():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM tasks")
    conn.execute("DELETE FROM meetings")
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True})


@app.get("/stalls")
async def check_stalls():
    """SLA breach prevention — detect tasks stuck 48h+ with no update."""
    stalled = detect_stalled_tasks(hours_threshold=48)
    if stalled:
        mark_tasks_stall_flagged([t['id'] for t in stalled])
        await asyncio.to_thread(post_stall_alert_to_slack, stalled)
    return JSONResponse({
        "stalled_count": len(stalled),
        "stalled_tasks": stalled
    })


@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "timestamp": datetime.now().isoformat()})