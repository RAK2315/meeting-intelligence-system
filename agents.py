import os, json, pickle, base64, time, sqlite3, urllib.request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from datetime import datetime
import fitz
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# --- CONFIG ---
from dotenv import load_dotenv
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY", "your_groq_key_here")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", "your_slack_webhook_here")
GROQ_MODEL = "llama-3.3-70b-versatile"
SCOPES = [
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/calendar'
]
DB_PATH = "tasks.db"
CHROMA_PATH = "./chroma_store"

groq_client = Groq(api_key=GROQ_KEY)

# --- AUDIT LOG ---
audit_log = []
def log(agent, action, detail, level="info"):
    entry = {"time": datetime.now().isoformat(), "agent": agent, "action": action, "detail": detail, "level": level}
    audit_log.append(entry)
    print(f"[{agent}] {'⚠' if level=='warning' else '✗' if level=='error' else '✓'} {action}: {detail}")

# --- LLM CALL ---
def call_llm(prompt, agent_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            log(agent_name, "error_detected", str(e)[:100], "error")
            time.sleep(2)
    log(agent_name, "fallback_activated", "All retries failed", "error")
    return None

# --- AUTH ---
def get_google_creds():
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as f:
            creds = pickle.load(f)
            if creds and creds.valid:
                return creds
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    with open('token.pickle', 'wb') as f:
        pickle.dump(creds, f)
    return creds

# --- INPUT PROCESSING ---
_whisper_model = None
def transcribe_audio(audio_path):
    global _whisper_model
    import whisper
    log("Whisper", "start", f"Transcribing {os.path.basename(audio_path)}")
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    result = _whisper_model.transcribe(audio_path)
    log("Whisper", "done", f"Transcribed {len(result['text'].split())} words")
    return result["text"]

def extract_text_from_pdf(file_path):
    log("PDFParser", "start", f"Extracting from {os.path.basename(file_path)}")
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text() for page in doc)
    log("PDFParser", "done", f"Extracted {len(text)} chars")
    return text

# --- SQLITE ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id TEXT, owner TEXT, task TEXT, deadline TEXT,
        status TEXT DEFAULT 'open', confidence REAL DEFAULT 0.85,
        created_at TEXT, updated_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS meetings (
        id TEXT PRIMARY KEY, summary TEXT, created_at TEXT, health TEXT)""")
    conn.commit()
    conn.close()

def clear_tasks_for_meeting(meeting_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM tasks WHERE meeting_id=?", (meeting_id,))
    conn.commit()
    conn.close()

def save_tasks_to_db(meeting_id, tasks):
    clear_tasks_for_meeting(meeting_id)
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    for t in tasks:
        conn.execute(
            "INSERT INTO tasks (meeting_id,owner,task,deadline,status,confidence,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (meeting_id, t['owner'], t['task'], t['deadline'], 'open',
             t.get('confidence', 0.85), now, now))
    conn.commit()
    conn.close()
    log("Database", "saved", f"Saved {len(tasks)} tasks for {meeting_id}")

def sync_overdue_status(meeting_id, escalations):
    """After saving tasks, mark overdue ones correctly in DB."""
    if not escalations:
        return
    conn = sqlite3.connect(DB_PATH)
    overdue_count = 0
    for task in escalations:
        if task.get('days_left', 0) < 0:
            conn.execute(
                "UPDATE tasks SET status='overdue', updated_at=? WHERE meeting_id=? AND owner=? AND task=?",
                (datetime.now().isoformat(), meeting_id, task['owner'], task['task']))
            overdue_count += 1
    conn.commit()
    conn.close()
    if overdue_count:
        log("Database", "overdue_synced", f"Marked {overdue_count} tasks as overdue in Task Board")

def get_all_tasks():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id,meeting_id,owner,task,deadline,status,confidence,created_at FROM tasks ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [{"id":r[0],"meeting_id":r[1],"owner":r[2],"task":r[3],
             "deadline":r[4],"status":r[5],"confidence":r[6],"created_at":r[7]} for r in rows]

def update_task_status(task_id, status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE tasks SET status=?,updated_at=? WHERE id=?",
                 (status, datetime.now().isoformat(), task_id))
    conn.commit()
    conn.close()

def save_meeting_to_db(meeting_id, summary, health):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO meetings (id,summary,created_at,health) VALUES (?,?,?,?)",
                 (meeting_id, summary, datetime.now().isoformat(), health))
    conn.commit()
    conn.close()

def get_recurring_issues(owners):
    conn = sqlite3.connect(DB_PATH)
    recurring = []
    for owner in owners:
        rows = conn.execute(
            "SELECT task,status FROM tasks WHERE owner=? AND status IN ('open','overdue') ORDER BY created_at DESC LIMIT 10",
            (owner,)).fetchall()
        if len(rows) >= 2:
            recurring.append({"owner": owner, "count": len(rows), "previous_tasks": [r[0] for r in rows]})
    conn.close()
    return recurring

# --- CHROMADB ---
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
    return chroma_client.get_or_create_collection("meetings", embedding_function=ef)

def store_meeting_in_memory(meeting_id, transcript, extracted_data):
    collection = get_chroma_collection()
    summary = (f"Meeting {meeting_id}. "
               f"Decisions: {', '.join(extracted_data['decisions'])}. "
               f"Tasks: {', '.join([t['task']+' ('+t['owner']+')' for t in extracted_data['tasks']])}.")
    collection.upsert(
        documents=[summary], ids=[meeting_id],
        metadatas=[{"meeting_id": meeting_id, "timestamp": datetime.now().isoformat()}])
    log("Memory", "stored", "Meeting stored in ChromaDB")

# --- SLACK ---
def post_to_slack(meeting_id, extracted_data, escalations):
    try:
        tasks_text = "\n".join([
            f"• {t['owner']}: {t['task']} (due {t['deadline']}) — {int(t.get('confidence',0.85)*100)}% confidence"
            for t in extracted_data['tasks'][:8]])
        decisions_text = "\n".join([f"✓ {d}" for d in extracted_data['decisions']]) or "None"
        esc_text = f"🚨 {len(escalations)} escalation(s) triggered" if escalations else "✅ All tasks on track"
        unresolved_text = "\n".join([f"? {u}" for u in extracted_data['unresolved']]) or "None"
        payload = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": f"🧠 MIS Report — {meeting_id}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Tasks ({len(extracted_data['tasks'])})*\n{tasks_text}"}},
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Decisions*\n{decisions_text}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Unresolved*\n{unresolved_text}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": esc_text}},
                {"type": "context", "elements": [{"type": "mrkdwn", "text": f"Posted by MIS · {datetime.now().strftime('%Y-%m-%d %H:%M')}"}]}
            ]
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(SLACK_WEBHOOK, data=data, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req)
        log("Slack", "posted", "Summary posted to #meeting-intelligence")
    except Exception as e:
        log("Slack", "post_failed", str(e)[:80], "warning")

# ─────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────

class PipelineState(TypedDict):
    transcript: str
    meeting_id: str
    extracted: Optional[dict]
    emails: Optional[List[dict]]
    approved_emails: Optional[List[dict]]
    escalations: Optional[List[dict]]
    calendar_events: Optional[List[dict]]
    report: Optional[dict]
    error_count: int
    awaiting_approval: bool
    gmail_service: object
    calendar_service: object

# ─────────────────────────────────────────────
# AGENT NODES
# ─────────────────────────────────────────────

def node_extractor(state: PipelineState) -> PipelineState:
    log("Extractor", "start", "Parsing transcript")
    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""You are a meeting analyst. Today's date is {today}. Extract structured data ONLY from the transcript below.

IMPORTANT RULES:
- Only extract names, tasks, decisions EXPLICITLY mentioned in this transcript
- Do NOT invent names not in the transcript. If unclear use "Unknown"
- Confidence reflects how clearly task/owner/deadline is stated (0.0 to 1.0)

DATE RULES — critical:
- Convert ALL relative dates to absolute YYYY-MM-DD using today = {today}
- "today" = {today}
- "tomorrow" = next day after {today}
- "end of day" = {today}
- "March 10th" or "March 10" = 2026-03-10
- "in two days" = 2 days after {today}
- If a task is described as overdue, extract the ORIGINAL missed deadline date
- Only use "not specified" if absolutely no date information exists anywhere for that task

Return ONLY valid JSON, no markdown, no explanation:
{{
  "decisions": ["decision1"],
  "tasks": [
    {{"task": "clear description", "owner": "name from transcript", "deadline": "YYYY-MM-DD or not specified", "confidence": 0.95}}
  ],
  "unresolved": ["question or issue with no resolution"]
}}

TRANSCRIPT:
{state['transcript']}"""

    raw = call_llm(prompt, "Extractor")

    if raw is None:
        log("Extractor", "fallback", "Using empty structure", "error")
        state['extracted'] = {"decisions":[],"tasks":[],"unresolved":["Extraction failed"],"recurring_issues":[]}
        state['error_count'] = state.get('error_count', 0) + 1
        return state

    try:
        cleaned = raw.strip().replace("```json","").replace("```","").strip()
        data = json.loads(cleaned)
        for t in data.get('tasks', []):
            if 'confidence' not in t:
                t['confidence'] = 0.85
        log("Extractor", "done", f"Found {len(data['tasks'])} tasks, {len(data['decisions'])} decisions")

        low_conf = [t for t in data['tasks'] if t.get('confidence', 1) < 0.7]
        if low_conf:
            log("Extractor", "low_confidence", f"{len(low_conf)} tasks below 70% confidence", "warning")

        owners = list(set(t['owner'] for t in data['tasks'] if t['owner'] != 'Unknown'))
        data['recurring_issues'] = get_recurring_issues(owners)
        for r in data['recurring_issues']:
            log("Extractor", "recurring_issue", f"{r['owner']} has {r['count']} open tasks from past meetings", "warning")

        state['extracted'] = data

    except json.JSONDecodeError:
        log("Extractor", "json_repair", "Malformed JSON, repairing", "warning")
        fixed = call_llm(f"Fix this JSON, return ONLY valid JSON:\n{raw}", "Extractor")
        try:
            data = json.loads(fixed.strip().replace("```json","").replace("```","").strip())
            for t in data.get('tasks', []):
                if 'confidence' not in t:
                    t['confidence'] = 0.7
            data['recurring_issues'] = []
            state['extracted'] = data
            log("Extractor", "json_repaired", "Repair successful")
        except:
            log("Extractor", "json_repair_failed", "Repair failed", "error")
            state['extracted'] = {"decisions":[],"tasks":[],"unresolved":["Extraction failed"],"recurring_issues":[]}
            state['error_count'] = state.get('error_count', 0) + 1

    return state


def node_action_writer(state: PipelineState) -> PipelineState:
    log("ActionWriter", "start", "Drafting emails (batch)")
    extracted = state['extracted']

    if not extracted.get("tasks"):
        log("ActionWriter", "skip", "No tasks found", "warning")
        state['emails'] = []
        state['awaiting_approval'] = False
        return state

    owners = {}
    for task in extracted["tasks"]:
        if task['owner'] != 'Unknown':
            owners.setdefault(task["owner"], []).append(task)

    if not owners:
        state['emails'] = []
        state['awaiting_approval'] = False
        return state

    recurring_notes = {
        r['owner']: f"(has {r['count']} previously unresolved tasks — be firm about urgency)"
        for r in extracted.get('recurring_issues', [])
    }

    all_tasks_str = "\n".join([
        f"{owner} {recurring_notes.get(owner,'')}: " +
        ", ".join([t['task'] + " by " + t['deadline'] for t in tasks])
        for owner, tasks in owners.items()
    ])

    prompt = f"""Write short professional follow-up emails for each person listed below.
Each email should reference their specific tasks and deadlines.
Tasks per person:
{all_tasks_str}

Key decisions from the meeting: {json.dumps(extracted['decisions'])}

Return a JSON array ONLY — no markdown, no explanation, just the array:
[{{"to":"name","subject":"Action Items from Today's Meeting","body":"email body text here"}}]"""

    raw = call_llm(prompt, "ActionWriter")

    if raw is None:
        emails = []
        for owner, tasks in owners.items():
            task_lines = "\n".join([f"- {t['task']} by {t['deadline']}" for t in tasks])
            emails.append({"to": owner, "subject": "Action Items from Today's Meeting",
                           "body": f"Hi {owner},\n\nYour action items from today's meeting:\n\n{task_lines}\n\nBest regards"})
        state['emails'] = emails
    else:
        try:
            emails = json.loads(raw.strip().replace("```json","").replace("```","").strip())
            for e in emails:
                log("ActionWriter", "drafted", f"Email for {e['to']}")
            state['emails'] = emails
        except:
            log("ActionWriter", "parse_error", "JSON parse failed, using fallback", "warning")
            emails = []
            for owner, tasks in owners.items():
                task_lines = "\n".join([f"- {t['task']} by {t['deadline']}" for t in tasks])
                emails.append({"to": owner, "subject": "Action Items from Today's Meeting",
                               "body": f"Hi {owner},\n\nYour action items:\n\n{task_lines}\n\nBest regards"})
            state['emails'] = emails

    state['awaiting_approval'] = True
    log("ActionWriter", "awaiting_approval", f"{len(state['emails'])} emails pending human review")
    return state


def node_task_tracker(state: PipelineState) -> PipelineState:
    log("TaskTracker", "start", "Checking deadlines")
    today = datetime.now().date()
    escalations = []
    gmail = state['gmail_service']

    for task in state['extracted'].get("tasks", []):
        dl = task.get("deadline", "not specified")
        if dl in ("not specified", "Not specified"):
            continue
        try:
            deadline = datetime.strptime(dl, "%Y-%m-%d").date()
            days_left = (deadline - today).days
            if days_left < 0:
                log("TaskTracker", "overdue_detected", f"{task['owner']}: {abs(days_left)}d overdue", "warning")
                escalations.append({**task, "status": "OVERDUE", "days_left": days_left})
            elif days_left <= 2:
                log("TaskTracker", "at_risk", f"{task['owner']}: {days_left}d remaining", "warning")
                escalations.append({**task, "status": "AT_RISK", "days_left": days_left})
            else:
                log("TaskTracker", "on_track", f"{task['owner']}: {days_left}d remaining")
        except ValueError:
            log("TaskTracker", "parse_error", f"Cannot parse '{dl}'", "warning")

    for task in escalations:
        is_overdue = task["days_left"] < 0
        body = (f"{'OVERDUE' if is_overdue else 'DUE SOON'}: {task['task']}\n"
                f"Owner: {task['owner']}\nDeadline: {task['deadline']}\n"
                f"{'Overdue by '+str(abs(task['days_left']))+' days' if is_overdue else 'Due in '+str(task['days_left'])+' days'}\n\n"
                f"Automated escalation from Meeting Intelligence System.")
        msg = MIMEText(body)
        msg['to'] = "rehtrooper@gmail.com"
        msg['subject'] = f"{'[OVERDUE]' if is_overdue else '[DUE SOON]'} {task['task'][:50]} — {task['owner']}"
        raw_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        try:
            if is_overdue:
                gmail.users().messages().send(userId='me', body={'raw': raw_msg}).execute()
                log("TaskTracker", "escalation_sent", f"Email SENT: {task['owner']}")
            else:
                gmail.users().drafts().create(userId='me', body={'message': {'raw': raw_msg}}).execute()
                log("TaskTracker", "escalation_drafted", f"Draft: {task['owner']}")
        except Exception as e:
            log("TaskTracker", "escalation_failed", str(e)[:80], "warning")

    if not escalations:
        log("TaskTracker", "all_on_track", "No escalations needed")

    state['escalations'] = escalations
    return state


def node_calendar(state: PipelineState) -> PipelineState:
    log("Calendar", "start", "Creating calendar events")
    events = []
    cal = state['calendar_service']
    meeting_id = state['meeting_id']

    for task in state['extracted'].get("tasks", []):
        dl = task.get("deadline", "not specified")
        if dl in ("not specified", "Not specified"):
            continue
        try:
            datetime.strptime(dl, "%Y-%m-%d")
            event = {
                'summary': f"[MIS] {task['task']} — {task['owner']}",
                'description': (f"Auto-created by MIS\nMeeting: {meeting_id}\n"
                                f"Owner: {task['owner']}\nConfidence: {int(task.get('confidence',0.85)*100)}%"),
                'start': {'date': dl}, 'end': {'date': dl},
                'reminders': {'useDefault': False, 'overrides': [
                    {'method': 'email', 'minutes': 24*60},
                    {'method': 'popup', 'minutes': 60}
                ]}
            }
            result = cal.events().insert(calendarId='primary', body=event).execute()
            events.append({"task": task['task'], "owner": task['owner'],
                          "event_id": result['id'], "deadline": dl})
            log("Calendar", "event_created", f"{task['owner']}: {task['task'][:40]}")
        except Exception as e:
            log("Calendar", "event_failed", str(e)[:80], "warning")

    state['calendar_events'] = events
    return state


def node_send_emails(state: PipelineState) -> PipelineState:
    gmail = state['gmail_service']
    approved = state.get('approved_emails', [])
    log("Gmail", "sending", f"Sending {len(approved)} approved emails")
    for email in approved:
        try:
            msg = MIMEText(email['body'])
            msg['to'] = "rehtrooper@gmail.com"
            msg['subject'] = f"[To: {email['to']}] {email['subject']}"
            raw_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            draft = gmail.users().drafts().create(
                userId='me', body={'message': {'raw': raw_msg}}).execute()
            log("Gmail", "draft_created", f"Draft for {email['to']}: {draft['id']}")
        except Exception as e:
            log("Gmail", "draft_failed", str(e)[:80], "warning")
    return state


def node_auditor(state: PipelineState) -> PipelineState:
    log("Auditor", "start", "Building audit report")
    real_errors = [e for e in audit_log if e["level"] == "error"]
    all_warnings = [e for e in audit_log if e["level"] in ["warning", "error"]]
    if all_warnings:
        log("Auditor", "anomalies_detected", f"{len(all_warnings)} warnings in pipeline", "warning")

    extracted = state['extracted']
    emails = state.get('approved_emails', state.get('emails', []))
    escalations = state.get('escalations', [])
    calendar_events = state.get('calendar_events', [])

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "decisions_count": len(extracted["decisions"]),
            "tasks_count": len(extracted["tasks"]),
            "emails_drafted": len(emails),
            "unresolved_count": len(extracted["unresolved"]),
            "calendar_events": len(calendar_events),
            "escalations": len(escalations),
            "recurring_issues": len(extracted.get("recurring_issues", [])),
            "warnings": len(all_warnings),
            "pipeline_health": "degraded" if real_errors else "healthy"
        },
        "decisions": extracted["decisions"],
        "tasks": extracted["tasks"],
        "unresolved": extracted["unresolved"],
        "recurring_issues": extracted.get("recurring_issues", []),
        "calendar_events": calendar_events,
        "audit_trail": audit_log
    }
    with open("audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log("Auditor", "done", f"Health: {report['summary']['pipeline_health']}")
    state['report'] = report
    return state


def node_slack(state: PipelineState) -> PipelineState:
    post_to_slack(state['meeting_id'], state['extracted'], state.get('escalations', []))
    return state


# ─────────────────────────────────────────────
# LANGGRAPH GRAPH
# ─────────────────────────────────────────────

def should_continue_after_extraction(state: PipelineState) -> str:
    if not state['extracted'].get('tasks'):
        return "auditor"
    return "action_writer"

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("extractor", node_extractor)
    g.add_node("action_writer", node_action_writer)
    g.add_node("task_tracker", node_task_tracker)
    g.add_node("calendar", node_calendar)
    g.add_node("send_emails", node_send_emails)
    g.add_node("auditor", node_auditor)
    g.add_node("slack", node_slack)
    g.set_entry_point("extractor")
    g.add_conditional_edges("extractor", should_continue_after_extraction, {
        "action_writer": "action_writer",
        "auditor": "auditor"
    })
    g.add_edge("action_writer", "task_tracker")
    g.add_edge("task_tracker", "calendar")
    g.add_edge("calendar", "auditor")
    g.add_edge("auditor", "slack")
    g.add_edge("slack", END)
    return g.compile()

PIPELINE_GRAPH = build_graph()

# ─────────────────────────────────────────────
# APPROVAL RESUME — saves tasks THEN syncs overdue
# ─────────────────────────────────────────────

def resume_after_approval(state: PipelineState, approved_emails: list) -> PipelineState:
    state['approved_emails'] = approved_emails
    state['awaiting_approval'] = False
    state = node_send_emails(state)
    state = node_auditor(state)
    state = node_slack(state)

    # Step 1: save all tasks as 'open'
    save_tasks_to_db(state['meeting_id'], state['extracted']['tasks'])

    # Step 2: immediately mark overdue ones correctly
    sync_overdue_status(state['meeting_id'], state.get('escalations', []))

    store_meeting_in_memory(state['meeting_id'], state['transcript'], state['extracted'])
    save_meeting_to_db(state['meeting_id'], json.dumps(state['extracted']['decisions']),
                       state['report']['summary']['pipeline_health'])
    return state

def create_draft(gmail_service, to_name, subject, body):
    msg = MIMEText(body)
    msg['to'] = "rehtrooper@gmail.com"
    msg['subject'] = f"[To: {to_name}] {subject}"
    raw_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    draft = gmail_service.users().drafts().create(
        userId='me', body={'message': {'raw': raw_msg}}).execute()
    log("Gmail", "draft_created", f"Draft for {to_name}: {draft['id']}")
    return draft['id']