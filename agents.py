import os, json, pickle, base64, time, sqlite3, urllib.request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import fitz
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
GROQ_KEY = os.getenv("GROQ_KEY", "")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", "")
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")

# Model routing: small model for simple tasks, large for extraction
GROQ_LARGE = "llama-3.3-70b-versatile"   # extraction, complex reasoning
GROQ_SMALL = "llama-3.1-8b-instant"       # email drafting, simple tasks

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

# --- LLM CALL WITH MODEL ROUTING ---
def call_llm(prompt, agent_name, max_retries=3, use_large=False):
    """Route to small model by default, large for complex tasks."""
    model = GROQ_LARGE if use_large else GROQ_SMALL
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "model" in err.lower() and attempt == 0:
                # Fallback to large if small fails
                model = GROQ_LARGE
                log(agent_name, "model_fallback", f"Small model failed, routing to large", "warning")
            else:
                log(agent_name, "error_detected", err[:100], "error")
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
    # Add stall_flagged column if not exists
    try:
        conn.execute("ALTER TABLE tasks ADD COLUMN stall_flagged INTEGER DEFAULT 0")
    except:
        pass
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
    if not escalations:
        return
    conn = sqlite3.connect(DB_PATH)
    count = 0
    for task in escalations:
        if task.get('days_left', 0) < 0:
            conn.execute(
                "UPDATE tasks SET status='overdue', updated_at=? WHERE meeting_id=? AND owner=? AND task=?",
                (datetime.now().isoformat(), meeting_id, task['owner'], task['task']))
            count += 1
    conn.commit()
    conn.close()
    if count:
        log("Database", "overdue_synced", f"Marked {count} tasks as overdue")

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

def detect_stalled_tasks(hours_threshold=48):
    """Find tasks with no update in X hours — SLA breach prevention."""
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(hours=hours_threshold)).isoformat()
    rows = conn.execute(
        "SELECT id,meeting_id,owner,task,deadline,status,updated_at FROM tasks WHERE status='open' AND updated_at < ? AND stall_flagged=0",
        (cutoff,)).fetchall()
    conn.close()
    stalled = [{"id":r[0],"meeting_id":r[1],"owner":r[2],"task":r[3],
                "deadline":r[4],"status":r[5],"updated_at":r[6]} for r in rows]
    return stalled

def mark_tasks_stall_flagged(task_ids):
    conn = sqlite3.connect(DB_PATH)
    for tid in task_ids:
        conn.execute("UPDATE tasks SET stall_flagged=1, updated_at=? WHERE id=?",
                     (datetime.now().isoformat(), tid))
    conn.commit()
    conn.close()

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

# --- NOTION INTEGRATION ---
def create_notion_task(task, owner, deadline, meeting_id, confidence):
    """Create a task in Notion database."""
    if not NOTION_TOKEN or not NOTION_DATABASE_ID:
        log("Notion", "skipped", "No Notion credentials configured", "warning")
        return None
    try:
        props = {
            "Name": {"title": [{"text": {"content": task[:100]}}]},
            "Owner": {"rich_text": [{"text": {"content": owner}}]},
            "Meeting": {"rich_text": [{"text": {"content": meeting_id}}]},
            "Status": {"select": {"name": "Not started"}},
            "Confidence": {"number": round(confidence * 100)}
        }
        # Only add Deadline if it's a valid date string
        if deadline and deadline not in ("not specified", "Not specified", "TBD", ""):
            try:
                from datetime import datetime
                datetime.strptime(deadline, "%Y-%m-%d")
                props["Deadline"] = {"date": {"start": deadline}}
            except ValueError:
                pass  # Skip invalid dates
        payload = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": props
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            "https://api.notion.com/v1/pages",
            data=data,
            headers={
                'Authorization': f'Bearer {NOTION_TOKEN}',
                'Content-Type': 'application/json',
                'Notion-Version': '2022-06-28'
            }
        )
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        log("Notion", "task_created", f"{owner}: {task[:40]}")
        return result.get('id')
    except Exception as e:
        log("Notion", "task_failed", str(e)[:80], "warning")
        return None

# --- SLACK ---
def post_to_slack(meeting_id, extracted_data, escalations):
    if not SLACK_WEBHOOK:
        log("Slack", "skipped", "No webhook configured", "warning")
        return
    try:
        tasks_text = "\n".join([
            f"• {t['owner']}: {t['task']} (due {t['deadline']}) — {int(t.get('confidence',0.85)*100)}% conf"
            for t in extracted_data['tasks'][:8]])
        decisions_text = "\n".join([f"✓ {d}" for d in extracted_data['decisions']]) or "None"
        esc_text = f"🚨 {len(escalations)} escalation(s)" if escalations else "✅ All on track"
        unresolved_text = "\n".join([f"? {u}" for u in extracted_data['unresolved']]) or "None"

        # Flag ambiguous tasks
        ambiguous = [t for t in extracted_data['tasks'] if t.get('confidence', 1) < 0.6 or t.get('owner') == 'Unknown']
        ambiguous_text = "\n".join([f"⚠ {t['task']} — owner unclear" for t in ambiguous]) if ambiguous else "None"

        payload = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": f"🧠 MIS Report — {meeting_id}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Tasks ({len(extracted_data['tasks'])})*\n{tasks_text}"}},
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Decisions*\n{decisions_text}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Unresolved*\n{unresolved_text}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Needs Clarification*\n{ambiguous_text}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": esc_text}},
                {"type": "context", "elements": [{"type": "mrkdwn", "text": f"MIS · {datetime.now().strftime('%Y-%m-%d %H:%M')}"}]}
            ]
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(SLACK_WEBHOOK, data=data, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req)
        log("Slack", "posted", "Summary posted to #meeting-intelligence")
    except Exception as e:
        log("Slack", "post_failed", str(e)[:80], "warning")

def post_stall_alert_to_slack(stalled_tasks):
    if not SLACK_WEBHOOK or not stalled_tasks:
        return
    try:
        tasks_text = "\n".join([f"• {t['owner']}: {t['task']} (last updated: {t['updated_at'][:10]})" for t in stalled_tasks[:5]])
        payload = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "⚠️ MIS Stall Alert — Tasks Stuck 48h+"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*{len(stalled_tasks)} task(s) have had no update in 48+ hours:*\n{tasks_text}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "These tasks may be stalled. Consider following up or reassigning."}},
                {"type": "context", "elements": [{"type": "mrkdwn", "text": f"Auto-detected by MIS TaskTracker · {datetime.now().strftime('%Y-%m-%d %H:%M')}"}]}
            ]
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(SLACK_WEBHOOK, data=data, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req)
        log("Stall", "slack_alert_sent", f"{len(stalled_tasks)} stalled tasks reported")
    except Exception as e:
        log("Stall", "slack_failed", str(e)[:80], "warning")

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
    notion_tasks: Optional[List[dict]]
    report: Optional[dict]
    error_count: int
    awaiting_approval: bool
    needs_clarification: Optional[List[dict]]
    gmail_service: object
    calendar_service: object

# ─────────────────────────────────────────────
# AGENT NODES
# ─────────────────────────────────────────────

def node_extractor(state: PipelineState) -> PipelineState:
    log("Extractor", "start", "Parsing transcript")
    today = datetime.now().strftime("%Y-%m-%d")

    # Use LARGE model for extraction — most critical step
    prompt = f"""You are a meeting analyst. Today's date is {today}. Extract structured data ONLY from the transcript below.

IMPORTANT RULES:
- Only extract names, tasks, decisions EXPLICITLY mentioned in this transcript
- Do NOT invent names. If owner is genuinely unclear, use "Unknown"
- Confidence reflects how clearly task/owner/deadline is stated (0.0 to 1.0)
- If confidence < 0.6, set needs_clarification to true for that task

DATE RULES:
- Convert ALL relative dates to absolute YYYY-MM-DD using today = {today}
- "today" = {today}, "tomorrow" = next day, "end of week" = nearest Friday
- "March 10th" = 2026-03-10. If month passed use 2026.
- If task is described as overdue, extract the ORIGINAL missed deadline
- Use "not specified" only if truly no date information exists

Return ONLY valid JSON, no markdown:
{{
  "decisions": ["decision1"],
  "tasks": [
    {{
      "task": "clear description",
      "owner": "name or Unknown",
      "deadline": "YYYY-MM-DD or not specified",
      "confidence": 0.95,
      "needs_clarification": false,
      "clarification_reason": ""
    }}
  ],
  "unresolved": ["question with no resolution"]
}}

TRANSCRIPT:
{state['transcript']}"""

    raw = call_llm(prompt, "Extractor", use_large=True)

    if raw is None:
        log("Extractor", "fallback", "Using empty structure", "error")
        state['extracted'] = {"decisions":[],"tasks":[],"unresolved":["Extraction failed"],"recurring_issues":[]}
        state['needs_clarification'] = []
        state['error_count'] = state.get('error_count', 0) + 1
        return state

    try:
        cleaned = raw.strip().replace("```json","").replace("```","").strip()
        data = json.loads(cleaned)

        for t in data.get('tasks', []):
            if 'confidence' not in t:
                t['confidence'] = 0.85
            if 'needs_clarification' not in t:
                t['needs_clarification'] = t.get('confidence', 1) < 0.6 or t.get('owner') == 'Unknown'
            if 'clarification_reason' not in t:
                t['clarification_reason'] = ''

        log("Extractor", "done", f"Found {len(data['tasks'])} tasks, {len(data['decisions'])} decisions")

        # Identify tasks needing clarification
        clarification_needed = [t for t in data['tasks'] if t.get('needs_clarification')]
        if clarification_needed:
            log("Extractor", "clarification_needed",
                f"{len(clarification_needed)} tasks need human clarification", "warning")

        low_conf = [t for t in data['tasks'] if t.get('confidence', 1) < 0.7]
        if low_conf:
            log("Extractor", "low_confidence", f"{len(low_conf)} tasks below 70% confidence", "warning")

        owners = list(set(t['owner'] for t in data['tasks'] if t['owner'] != 'Unknown'))
        data['recurring_issues'] = get_recurring_issues(owners)
        for r in data['recurring_issues']:
            log("Extractor", "recurring_issue", f"{r['owner']} has {r['count']} open tasks from past meetings", "warning")

        state['extracted'] = data
        state['needs_clarification'] = clarification_needed

    except json.JSONDecodeError:
        log("Extractor", "json_repair", "Malformed JSON, repairing", "warning")
        # Use small model for repair — simple task
        fixed = call_llm(f"Fix this JSON, return ONLY valid JSON:\n{raw}", "Extractor", use_large=False)
        try:
            data = json.loads(fixed.strip().replace("```json","").replace("```","").strip())
            for t in data.get('tasks', []):
                if 'confidence' not in t:
                    t['confidence'] = 0.7
                t.setdefault('needs_clarification', False)
                t.setdefault('clarification_reason', '')
            data['recurring_issues'] = []
            state['extracted'] = data
            state['needs_clarification'] = []
            log("Extractor", "json_repaired", "Repair successful")
        except:
            log("Extractor", "json_repair_failed", "Repair failed", "error")
            state['extracted'] = {"decisions":[],"tasks":[],"unresolved":["Extraction failed"],"recurring_issues":[]}
            state['needs_clarification'] = []
            state['error_count'] = state.get('error_count', 0) + 1

    return state


def node_action_writer(state: PipelineState) -> PipelineState:
    log("ActionWriter", "start", "Drafting emails (batch) — using small model")
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
        r['owner']: f"(has {r['count']} previously unresolved tasks — be firm)"
        for r in extracted.get('recurring_issues', [])
    }

    all_tasks_str = "\n".join([
        f"{owner} {recurring_notes.get(owner,'')}: " +
        ", ".join([t['task'] + " by " + t['deadline'] for t in tasks])
        for owner, tasks in owners.items()
    ])

    # Use SMALL model for email drafting — cost efficiency
    prompt = f"""Write short professional follow-up emails for each person below.
Tasks per person:
{all_tasks_str}

Decisions: {json.dumps(extracted['decisions'])}

Return JSON array ONLY, no markdown:
[{{"to":"name","subject":"Action Items from Today's Meeting","body":"email body"}}]"""

    raw = call_llm(prompt, "ActionWriter", use_large=False)

    if raw is None:
        emails = []
        for owner, tasks in owners.items():
            task_lines = "\n".join([f"- {t['task']} by {t['deadline']}" for t in tasks])
            emails.append({"to": owner, "subject": "Action Items from Today's Meeting",
                           "body": f"Hi {owner},\n\nYour action items:\n\n{task_lines}\n\nBest regards"})
        state['emails'] = emails
    else:
        try:
            emails = json.loads(raw.strip().replace("```json","").replace("```","").strip())
            for e in emails:
                log("ActionWriter", "drafted", f"Email for {e['to']}")
            state['emails'] = emails
        except:
            log("ActionWriter", "parse_error", "Fallback to templates", "warning")
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


def node_notion(state: PipelineState) -> PipelineState:
    """Create tasks in Notion database."""
    if not NOTION_TOKEN or not NOTION_DATABASE_ID:
        log("Notion", "skipped", "Notion not configured — set NOTION_TOKEN and NOTION_DATABASE_ID in .env")
        state['notion_tasks'] = []
        return state

    log("Notion", "start", "Creating tasks in Notion")
    notion_tasks = []
    meeting_id = state['meeting_id']

    for task in state['extracted'].get("tasks", []):
        page_id = create_notion_task(
            task['task'], task['owner'], task['deadline'],
            meeting_id, task.get('confidence', 0.85)
        )
        if page_id:
            notion_tasks.append({"task": task['task'], "owner": task['owner'], "notion_id": page_id})

    log("Notion", "done", f"Created {len(notion_tasks)} Notion tasks")
    state['notion_tasks'] = notion_tasks
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
    notion_tasks = state.get('notion_tasks', [])
    needs_clarification = state.get('needs_clarification', [])

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_routing": {
            "extraction": GROQ_LARGE,
            "email_drafting": GROQ_SMALL,
            "note": "Smart routing: large model for reasoning, small for generation"
        },
        "summary": {
            "decisions_count": len(extracted["decisions"]),
            "tasks_count": len(extracted["tasks"]),
            "emails_drafted": len(emails),
            "unresolved_count": len(extracted["unresolved"]),
            "calendar_events": len(calendar_events),
            "escalations": len(escalations),
            "recurring_issues": len(extracted.get("recurring_issues", [])),
            "needs_clarification": len(needs_clarification),
            "notion_tasks_created": len(notion_tasks),
            "warnings": len(all_warnings),
            "pipeline_health": "degraded" if real_errors else "healthy"
        },
        "decisions": extracted["decisions"],
        "tasks": extracted["tasks"],
        "unresolved": extracted["unresolved"],
        "needs_clarification": needs_clarification,
        "recurring_issues": extracted.get("recurring_issues", []),
        "calendar_events": calendar_events,
        "notion_tasks": notion_tasks,
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
    g.add_node("notion", node_notion)
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
    g.add_edge("calendar", "notion")
    g.add_edge("notion", "auditor")
    g.add_edge("auditor", "slack")
    g.add_edge("slack", END)
    return g.compile()

PIPELINE_GRAPH = build_graph()

# ─────────────────────────────────────────────
# APPROVAL RESUME
# ─────────────────────────────────────────────

def resume_after_approval(state: PipelineState, approved_emails: list,
                          clarifications: dict = None) -> PipelineState:
    # Apply any clarifications from human
    if clarifications:
        for task in state['extracted']['tasks']:
            task_key = f"{task['owner']}:{task['task']}"
            if task_key in clarifications:
                task['owner'] = clarifications[task_key].get('owner', task['owner'])
                task['deadline'] = clarifications[task_key].get('deadline', task['deadline'])
                task['needs_clarification'] = False
                log("Clarification", "applied", f"Updated: {task['task'][:40]}")

    state['approved_emails'] = approved_emails
    state['awaiting_approval'] = False
    state = node_send_emails(state)
    state = node_auditor(state)
    state = node_slack(state)
    save_tasks_to_db(state['meeting_id'], state['extracted']['tasks'])
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