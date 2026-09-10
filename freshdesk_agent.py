from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
import os
import json
import secrets
from pathlib import Path
from contextlib import asynccontextmanager

import requests
import ngrok
import ollama
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv

load_dotenv() 

FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
BASE_URL = (f"https://{FRESHDESK_DOMAIN}.freshdesk.com/api/v2")
FRESHDESK_PASSWORD = os.getenv("FRESHDESK_PASSWORD", "X")
SUPPORT_AGENT_ID = int(os.getenv("SUPPORT_AGENT_ID"))
SUPPORT_EMPLOYEE_ID = int(os.getenv("SUPPORT_EMPLOYEE_ID"))

AUTH = (FRESHDESK_API_KEY, FRESHDESK_PASSWORD)

WEBHOOK_USERNAME = os.getenv("WEBHOOK_USERNAME")
WEBHOOK_PASSWORD = os.getenv("WEBHOOK_PASSWORD")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
ollama_client = ollama.Client(host=OLLAMA_HOST)

PROMPT_PATH = Path(__file__).parent / "AI-classification-setup" / "classification_system_prompt.txt"
CLASSIFICATION_SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

security = HTTPBasic()

def verify_webhook_auth(credentials: HTTPBasicCredentials = Depends(security)) -> bool:
    """Checks the Basic Auth header Freshdesk sends against our own secret.
    This is a credential you invent yourself and enter in the Freshdesk
    Automation's 'Authentication' field - separate from the Freshdesk API key."""
    if not WEBHOOK_USERNAME or not WEBHOOK_PASSWORD:
        raise RuntimeError("WEBHOOK_USERNAME and WEBHOOK_PASSWORD must be configured")
 
    valid_user = secrets.compare_digest(credentials.username, WEBHOOK_USERNAME)
    valid_pass = secrets.compare_digest(credentials.password, WEBHOOK_PASSWORD)
    if not (valid_user and valid_pass):
        raise HTTPException(status_code=401, detail="Invalid webhook credentials")
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))
    forward_kwargs = {"addr": "127.0.0.1:8085"}
    if NGROK_DOMAIN:
        forward_kwargs["domain"] = NGROK_DOMAIN
    forwarder = await ngrok.forward(**forward_kwargs)
    print(f"Public webhook URL: {forwarder.url()}/freshdesk-webhook")
    yield
    ngrok.disconnect()


app = FastAPI(lifespan=lifespan)


def classify_and_draft_reply(description: str) -> dict:
    """Ask the local model to classify the ticket and draft a reply."""
    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        format="json",
        think=True,
    )

    thinking = getattr(response.message, "thinking", None)
    if thinking:
        print(f"Model reasoning (not sent to tenant): {thinking}")
 
    content = response.message.content
    print(f"Model raw output: {content!r}")
 
    result = _parse_model_json(content)
    if result is None or result.get("category") not in ("wifi_or_internet", "other") or "reply" not in result:
        print(f"Model returned unexpected output, falling back to human handoff. Parsed as: {result}")
        return {
            "category": "other",
            "reply": "Hi, thanks for reaching out. Our team will contact you soon.",
        }
 
    return result


def _parse_model_json(content: str) -> dict | None:
    """Parse the model's JSON output, tolerating extra text around it.
 
    Known Ollama issue: format="json" isn't always reliably enforced for
    Qwen3.5 when think=False, so the model can occasionally wrap the JSON
    in stray text. This tries a plain parse first, then falls back to
    extracting the first {...} block before giving up.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
 
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass
 
    return None


def reply_to_ticket(ticket_id: int, message_html: str, assign: int) -> dict:
    """POST a public reply to a Freshdesk ticket - this is what emails the requester."""
    if not BASE_URL or not FRESHDESK_API_KEY:
        raise RuntimeError("FRESHDESK_DOMAIN and FRESHDESK_API_KEY must be configured")
 
    response = requests.post(
        f"{BASE_URL}/tickets/{ticket_id}/reply",
        auth=AUTH,
        json={"body": message_html},
        timeout=15,
    )
    assign_agent = requests.put(
        f"{BASE_URL}/tickets/{ticket_id}",
        auth=AUTH,
        json={"responder_id": SUPPORT_AGENT_ID},
        timeout=15,
    )
    assign_employee = requests.put(
        f"{BASE_URL}/tickets/{ticket_id}",
        auth=AUTH,
        json={"responder_id": SUPPORT_EMPLOYEE_ID},
        timeout=15,
    )

    response.raise_for_status()
    assign_agent.raise_for_status()
    assign_employee.raise_for_status()
    assigning = assign_agent if assign == 1 else assign_employee
    return {"reply": response.json(), "assign": assigning.json()}


def process_ticket(ticket_id: int, requester_email: str, description: str) -> None:
    """Runs the slow AI classification + Freshdesk reply after the webhook
    has already been acknowledged, so Freshdesk/the sender never times out
    waiting on the model."""
    decision = classify_and_draft_reply(description)
    assign = 1 if decision["category"] == "wifi_or_internet" else 2
    reply_message = decision["reply"]
 
    try:
        reply_to_ticket(ticket_id, reply_message, assign)
        target = "AI agent" if assign == 1 else "human employee"
        print(f"Replied to ticket {ticket_id} (requester: {requester_email}) and assigned it to the {target}.")
    except requests.exceptions.HTTPError as e:
        print(f"Failed to reply to ticket {ticket_id}: {e.response.text}")


@app.post("/freshdesk-webhook", status_code=202)
async def receive_ticket(request: Request, background_tasks: BackgroundTasks, authorized: bool = Depends(verify_webhook_auth)):
    payload = await request.json()
    print(f"Received webhook payload: {payload}")
 
    ticket_id = payload.get("ticket_id")
    requester_email = payload.get("requester_email")
    description = payload.get("description_text")
 
    background_tasks.add_task(process_ticket, int(ticket_id), requester_email, description)
 
    return {"status": "accepted", "ticket_id": int(ticket_id)}