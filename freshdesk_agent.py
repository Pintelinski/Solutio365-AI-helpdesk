from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
import os
import json
import secrets
import re
from pathlib import Path
from contextlib import asynccontextmanager

import requests
import ngrok
import ollama
import chromadb
from pypdf import PdfReader
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

EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = Path(__file__).parent / "chroma_db"
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
known_issues_collection = chroma_client.get_collection(name="known_issues")

ATTACHMENTS_DIR = Path(__file__).parent / "attachments"

# --- TESTING OVERRIDE: remove this line before going live ---
TEST_EMAIL_OVERRIDE = os.getenv("TEST_EMAIL_OVERRIDE")  # forces all outgoing mail to this address for testing
# ---------------------------------------------------------------
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")

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

def download_and_extract_pdfs(ticket_id: int, attachments: list[dict]) -> str:
    """Download PDF attachments and extract their text content."""
    ticket_dir = ATTACHMENTS_DIR / str(ticket_id)
    extracted_texts = []

    for attachment in attachments:
        if attachment.get("content_type") != "application/pdf":
            continue
        url = attachment.get("attachment_url")
        name = attachment.get("name", "attachment.pdf")
        if not url:
            continue
        try:
            pdf_response = requests.get(url, timeout=15)
            pdf_response.raise_for_status()
            ticket_dir.mkdir(parents=True, exist_ok=True)
            file_path = ticket_dir / name
            file_path.write_bytes(pdf_response.content)

            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            extracted_texts.append(f"[PDF attachment '{name}']\n{text}")
            print(f"Extracted text from PDF: {name} ({len(text)} chars)")
        except Exception as e:
            print(f"Failed to process PDF {name}: {e}")

    return "\n\n".join(extracted_texts)

def retrieve_relevant_issues(description: str, n_results: int = 2) -> str:
    """Embed the ticket text and find the most similar known issues in Chroma.
    Returns a formatted text block to inject into the prompt, or an empty
    string if nothing relevant enough is found."""
    if not description or not description.strip():
        return ""
    query_embedding = ollama_client.embeddings(model=EMBED_MODEL, prompt=description)["embedding"]
    results = known_issues_collection.query(query_embeddings=[query_embedding], n_results=n_results)
 
    if not results["documents"] or not results["documents"][0]:
        return ""
 
    blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        blocks.append(f"- Known pattern: {doc}\n  Suggested category: {meta['category']}\n  Guidance: {meta['guidance']}")
 
    return "Relevant known issues (for reference, use your judgment):\n" + "\n".join(blocks)

def classify_and_draft_reply(description: str, image_paths: list[Path], pdf_text: str = "") -> dict:
    """Ask the local model to classify the ticket and draft a reply.
    If image_paths is given, the images are attached to the user message so
    the model can look at them directly (e.g. a photo of a router)."""
    retrieval_query = f"{description}\n{pdf_text}".strip()
    relevant_issues = retrieve_relevant_issues(retrieval_query)

    if image_paths:
        message_text = f"{description}\n\n[{len(image_paths)} image attachment(s) are included with this message.]"
    else:
        message_text = f"{description}\n\n[No image attachments were included with this message. Do not claim to have seen a photo or screenshot.]"

    if relevant_issues:
        message_text += f"\n\n{relevant_issues}"

    if pdf_text:
        message_text += f"\n\npdf text: {pdf_text}"
        print(pdf_text)
    else:
        message_text += "\n\n[No PDF attachment was included with this message. Do not reference form fields like 'omschrijving' or 'Toestemming om woning te betreden' unless a PDF was actually provided.]"

    user_message = {"role": "user", "content": message_text}
    if image_paths:
        user_message["images"] = [path.read_bytes() for path in image_paths]

    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            user_message,
        ],
        format="json",
        think=False,
        options={"num_ctx": 16384, "temperature": 0.4, "num_thread": 6},
    )

    thinking = getattr(response.message, "thinking", None)
    if thinking:
        print(f"Model reasoning (not sent to tenant): {thinking}")

    content = response.message.content

    result = _parse_model_json(content)
    if result is None or result.get("category") not in ("wifi_info_needed", "wifi_resolved", "wifi_escalate", "other") or "reply" not in result:
        print(f"Model returned unexpected output, falling back to human handoff. Parsed as: {result}")
        return {
            "category": "other",
            "reply": "Hi, thanks for reaching out. Our team will contact you soon.",
            "target_email": None,
        }

    result["target_email"] = validate_target_email(result.get("target_email"), description, pdf_text)
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


def validate_target_email(target_email: str | None, description: str, pdf_text: str) -> str | None:
    """Only trust target_email if it's a well-formed address that actually
    appears somewhere in the source text - guards against the model
    inventing/guessing an address rather than reading one."""
    if not target_email:
        return None
    if not EMAIL_PATTERN.fullmatch(target_email.strip()):
        print(f"Rejected target_email (not a valid email format): {target_email!r}")
        return None
    combined_text = f"{description}\n{pdf_text}"
    if target_email.strip().lower() not in combined_text.lower():
        print(f"Rejected target_email (not found in source text): {target_email!r}")
        return None
    return target_email.strip()


def get_ticket_attachments(ticket_id: int) -> tuple[list[dict], list[str]]:
    """Fetch the full ticket from Freshdesk and return both:
    - traditional file attachments (from the 'attachments' field)
    - inline image URLs pasted directly into the email body (Freshdesk embeds
      these as <img> tags in the HTML 'description' field, not in
      'attachments' - this is the common case for copy-pasted screenshots).
    """
    response = requests.get(f"{BASE_URL}/tickets/{ticket_id}", auth=AUTH, timeout=15)
    response.raise_for_status()
    ticket = response.json()

    attachments = ticket.get("attachments", [])

    description_html = ticket.get("description", "") or ""
    inline_image_urls = re.findall(r'<img[^>]+src="([^"]+)"', description_html)

    print(f"Ticket {ticket_id}: {len(attachments)} file attachment(s), {len(inline_image_urls)} inline image(s) in body")
    return attachments, inline_image_urls


def download_image_attachments(ticket_id: int, attachments: list[dict], inline_image_urls: list[str]) -> list[Path]:
    """Download both traditional file attachments and inline pasted images
    to disk, grouped in a per-ticket folder (attachments/<ticket_id>/) so
    they can be found and deleted later once the ticket is resolved."""
    ticket_dir = ATTACHMENTS_DIR / str(ticket_id)
    saved_paths = []

    for attachment in attachments:
        content_type = attachment.get("content_type", "")
        if not content_type.startswith("image/"):
            continue
        url = attachment.get("attachment_url")
        name = attachment.get("name", "attachment")
        if not url:
            continue
        try:
            img_response = requests.get(url, timeout=15)
            img_response.raise_for_status()
            ticket_dir.mkdir(parents=True, exist_ok=True)
            file_path = ticket_dir / name
            file_path.write_bytes(img_response.content)
            saved_paths.append(file_path)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download attachment {name}: {e}")

    for i, url in enumerate(inline_image_urls):
        try:
            img_response = requests.get(url, auth=AUTH, timeout=15)
            img_response.raise_for_status()
            ticket_dir.mkdir(parents=True, exist_ok=True)
            file_path = ticket_dir / f"inline_{i}.png"
            file_path.write_bytes(img_response.content)
            saved_paths.append(file_path)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download inline image from {url}: {e}")

    return saved_paths


def reply_to_ticket(ticket_id: int, message_html: str, assign: int, target_email: str | None = None) -> dict:
    """POST a public reply to a Freshdesk ticket - this is what emails the requester."""
    if not BASE_URL or not FRESHDESK_API_KEY:
        raise RuntimeError("FRESHDESK_DOMAIN and FRESHDESK_API_KEY must be configured")

    message_text = message_html.replace("\n", "<br>")

    # TESTING OVERRIDE - forces all outgoing mail to your own address regardless
    # of what target_email logic below would otherwise pick. Remove this line,
    # keep the real logic beneath it, once you're done testing.
    send_to = TEST_EMAIL_OVERRIDE if target_email else None
    # send_to = target_email  # <- real logic, re-enable this once override is removed

    if send_to:
        response = requests.post(
            f"{BASE_URL}/tickets/{ticket_id}/reply_to_forward",
            auth=AUTH,
            json={"body": message_text, "to_emails": [send_to]},
            timeout=15,
        )
    else:
        response = requests.post(
            f"{BASE_URL}/tickets/{ticket_id}/reply",
            auth=AUTH,
            json={"body": message_text},
            timeout=15,
        )
    response.raise_for_status()

    responder_id = SUPPORT_AGENT_ID if assign == 1 else SUPPORT_EMPLOYEE_ID
    assign_response = requests.put(
        f"{BASE_URL}/tickets/{ticket_id}",
        auth=AUTH,
        json={"responder_id": responder_id},
        timeout=15,
    )
    assign_response.raise_for_status()

    return {"reply": response.json(), "assign": assign_response.json()}


def process_ticket(ticket_id: int, requester_email: str, description: str) -> None:
    """Runs the slow AI classification + Freshdesk reply after the webhook
    has already been acknowledged, so Freshdesk/the sender never times out
    waiting on the model."""
    attachments, inline_image_urls = get_ticket_attachments(ticket_id)
    image_paths = download_image_attachments(ticket_id, attachments, inline_image_urls)
    print(f"Ticket {ticket_id}: saved {len(image_paths)} image(s) total to {ATTACHMENTS_DIR / str(ticket_id)}")

    pdf_text = download_and_extract_pdfs(ticket_id, attachments)
    decision = classify_and_draft_reply(description, image_paths, pdf_text)
    assign = 1 if decision["category"] in ("wifi_info_needed", "wifi_resolved") else 2
    reply_message = decision["reply"]

    try:
        reply_to_ticket(ticket_id, reply_message, assign, decision.get("target_email"))
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