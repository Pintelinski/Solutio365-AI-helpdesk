from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
import os
import json
import secrets
import re
import hashlib
import shutil
import ipaddress
import socket
from urllib.parse import urlparse, urljoin
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from contextlib import asynccontextmanager
from html import unescape

from langdetect import detect, DetectorFactory
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
FRESHDESK_PASSWORD = os.getenv("FRESHDESK_PASSWORD")
SUPPORT_AGENT_ID = int(os.getenv("SUPPORT_AGENT_ID"))
SUPPORT_EMPLOYEE_ID = int(os.getenv("SUPPORT_EMPLOYEE_ID"))
SUPPORT_AGENT_INTERCOM_ID = int(os.getenv("SUPPORT_AGENT_INTERCOM_ID"))
ALLOWED_INLINE_IMAGE_HOSTS = {
    f"{FRESHDESK_DOMAIN}.freshdesk.com".lower(),
    "eucattachment.freshdesk.com",
    "solutio365.nl",
    "www.solutio365.nl",
    "codomo.eu",
    "magisrealestate.com",
    "holland2stay.com",
 }

AUTH = (FRESHDESK_API_KEY, FRESHDESK_PASSWORD)

WEBHOOK_USERNAME = os.getenv("WEBHOOK_USERNAME")
WEBHOOK_PASSWORD = os.getenv("WEBHOOK_PASSWORD")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
ollama_client = ollama.Client(host=OLLAMA_HOST)

PROMPT_PATH = Path(__file__).parent / "AI-classification-setup" / "classification_system_prompt.txt"
CLASSIFICATION_SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

PDF_EXTRACTION_PROMPT_PATH = Path(__file__).parent / "AI-classification-setup" / "pdf_extraction_prompt.txt"
PDF_EXTRACTION_PROMPT = PDF_EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8")

IMAGE_ANALYSIS_PROMPT_PATH = Path(__file__).parent / "AI-classification-setup" / "image_analysis_prompt.txt"
IMAGE_ANALYSIS_PROMPT = IMAGE_ANALYSIS_PROMPT_PATH.read_text(encoding="utf-8")

DetectorFactory.seed = 0
LANGUAGE_MAP = {"nl": "Dutch", "en": "English"}

EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = Path(__file__).parent / "chroma_db"
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
known_issues_collection = chroma_client.get_collection(name="known_issues")

ATTACHMENTS_DIR = Path(__file__).parent / "attachments"

VALID_CATEGORIES = ("wifi", "intercom", "tv", "other", "ignore")

MIN_PDF_IMAGE_DIMENSION = 250   # px - embedded PDF images smaller than this on either side are skipped
MAX_PDF_IMAGES = 6              # per PDF
MAX_IMAGES_TO_ANALYZE = 4       # per ticket
MAX_IMAGE_SIDE = 2000           # px - larger extracted PDF images are scaled down
VALID_IMAGE_TYPES = ("router_photo", "speedtest", "other")
MAC_PATTERN = re.compile(r"(?:[0-9A-Fa-f]{2}[:\-.]?){5}[0-9A-Fa-f]{2}")

URL_PATTERN = re.compile(r'https?://[^\s<>"\']+')
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b")
NAME_BEFORE_EMAIL_PATTERN_TEMPLATE = r"([A-Z][\w.\-]+(?:\s+[A-Z][\w.\-]+){0,3})\s*<\s*__EMAIL__\s*>"
FORBIDDEN_EMAIL_DOMAINS = {
    domain.strip().lower().lstrip("@")
    for domain in os.getenv("FORBIDDEN_EMAIL_DOMAINS", "").split(",")
    if domain.strip()
}
FORBIDDEN_EMAIL_ADDRESSES = {
    address.strip().lower()
    for address in os.getenv("FORBIDDEN_EMAIL_ADDRESSES", "").split(",")
    if address.strip()
}
ALLOWED_EMAIL_TLDS = {
    f".{tld.strip().lower().lstrip('.')}"
    for tld in os.getenv("ALLOWED_EMAIL_TLDS", "com,net,org,nl").split(",")
    if tld.strip()
}

NON_CONFIGURABLE_ADDRESSES_PATH = Path(__file__).parent / "AI-classification-setup" / "non_configurable_intercom_addresses.json"
ROUTER_TIPS_PATH = Path(__file__).parent / "AI-classification-setup" / "router_troubleshooting_tips.json"

TV_ADDRESSES_PATH = Path(__file__).parent / "AI-classification-setup" / "tv_addresses.json"
TV_TEMPLATES_PATH = Path(__file__).parent / "AI-classification-setup" / "tv_reply_templates.json"
TV_MANUALS_DIR = Path(__file__).parent / "AI-classification-setup" / "tv_manuals"

OFFICE_TIMEZONE = ZoneInfo("Europe/Amsterdam")
OUT_OF_OFFICE_TEXT = {
    "English": "Thank you for your ticket. Our support team is currently out of office. We are available Monday to Friday, 08:00-17:00. We will look into your ticket as soon as we are back.",
    "Dutch": "Bedankt voor uw ticket. Ons supportteam is momenteel niet aanwezig. Wij zijn bereikbaar van maandag tot en met vrijdag van 08:00 tot 17:00 uur. We bekijken uw ticket zodra we weer aanwezig zijn.",
}

STREET_ADDRESS_PATTERN = re.compile(
    r"\b((?:(?!(?i:de|het|een|op|aan|in|naar|bij|uit|door|en|is|mijn|ons|onze|zijn|voor|met|die|dat|deze|ik|woon|wonen|wij)\s)"
    r"[A-Za-zÀ-ÿ'\-]+\s+)?"
    r"(?:(?i:van|der|den|ter|ten)\s+)*"
    r"[A-Za-zÀ-ÿ'\-]*(?i:straat|laan|weg|plein|gracht|singel|dijk|kade|pad|hof|dreef|steeg|boulevard|park|plantsoen|kanaal|baan|erf))"
    r"\s+(\d{1,4}[A-Za-z]?(?:[-/]\d{1,3})?)\b"
)

# --- TESTING OVERRIDE: remove this line before going live ---
TEST_EMAIL_OVERRIDE = os.getenv("TEST_EMAIL_OVERRIDE")  # forces all outgoing mail to this address for testing
# ---------------------------------------------------------------

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


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def is_out_of_office(now: datetime | None = None) -> bool:
    """Mon-Fri 08:00-17:00 Europe/Amsterdam = in office. Everything else
    (evenings, nights, weekends) = out of office. Pure clock check."""
    now = now or datetime.now(OFFICE_TIMEZONE)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return True
    return not (8 <= now.hour < 17)

def extract_pdf_images(reader: PdfReader, out_dir: Path, prefix: str) -> list[Path]:
    """Save the meaningful images embedded in a PDF (e.g. a speedtest or router
    photo in a 'bijlage') so they can be analyzed like any other image.
    Skips small decorative images (logos, icons) and exact duplicates, and
    re-encodes everything as PNG so the vision model always gets a readable file."""
    saved = []
    seen_hashes = set()
    safe_prefix = re.sub(r"[^\w\-]", "_", prefix)

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_images = list(page.images)
        except Exception as e:
            print(f"Could not read images on page {page_number} of {prefix}: {e}")
            continue

        for index, image in enumerate(page_images):
            if len(saved) >= MAX_PDF_IMAGES:
                return saved
            try:
                digest = hashlib.md5(image.data).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)

                pil_image = image.image
                width, height = pil_image.size
                if width < MIN_PDF_IMAGE_DIMENSION or height < MIN_PDF_IMAGE_DIMENSION:
                    continue

                if pil_image.mode not in ("RGB", "L"):
                    pil_image = pil_image.convert("RGB")
                pil_image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

                out_dir.mkdir(parents=True, exist_ok=True)
                file_path = out_dir / f"{safe_prefix}_p{page_number}_{index}.png"
                pil_image.save(file_path, format="PNG")
                saved.append(file_path)
                print(f"Saved image from PDF: {file_path} ({width}x{height})")
            except Exception as e:
                print(f"Skipping unreadable image {index} on page {page_number} of {prefix}: {e}")

    return saved


def download_and_extract_pdfs(ticket_id: int, attachments: list[dict]) -> tuple[str, list[Path]]:
    """Download PDF attachments. Returns (raw text, images embedded in the PDFs)."""
    ticket_dir = ATTACHMENTS_DIR / str(ticket_id)
    extracted_texts = []
    pdf_images = []

    for attachment in attachments:
        if attachment.get("content_type") != "application/pdf":
            continue
        url = attachment.get("attachment_url")
        name = re.sub(r"[^\w.\-]", "_", Path(attachment.get("name") or "attachment.pdf").name).lstrip(".") or "attachment.pdf"
        if not url:
            continue
        try:
            request_headers = {
                "Accept": "application/pdf,*/*",
                "User-Agent": "Solutio365-AI-Helpdesk/1.0",
            }
            pdf_response = requests.get(url, auth=AUTH, headers=request_headers, timeout=15)
            if pdf_response.status_code in (400, 401, 403):
                # Some attachment URLs are signed/download URLs that reject Basic Auth.
                pdf_response = requests.get(url, headers=request_headers, timeout=15)
            pdf_response.raise_for_status()
            if len(pdf_response.content) > MAX_DOWNLOAD_BYTES:
                print(f"Skipping attachment {name}: too large")
                continue
            if not pdf_response.content.startswith(b"%PDF-"):
                preview = pdf_response.content[:200].decode("utf-8", errors="replace").replace("\n", " ")
                print(f"Skipping attachment {name}: downloaded content is not a PDF ({pdf_response.status_code}; {preview})")
                continue
            ticket_dir.mkdir(parents=True, exist_ok=True)
            file_path = ticket_dir / name
            file_path.write_bytes(pdf_response.content)

            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            extracted_texts.append(text)
            print(f"Extracted raw text from PDF {name} ({len(text)} chars)")

            pdf_images.extend(extract_pdf_images(reader, ticket_dir, file_path.stem))
        except Exception as e:
            print(f"Failed to process PDF {name}: {e}")

    return "\n\n".join(extracted_texts), pdf_images


def find_download_links(text: str) -> list[str]:
    """Find http(s) URLs in ticket text - some companies send tickets as a
    download link to an external document instead of a real attachment."""
    if not text:
        return []
    links = []
    for raw_url in URL_PATTERN.findall(unescape(text)):
        url = raw_url.rstrip(".,;:!?)]}")
        if url not in links:
            links.append(url)
    return links


def fetch_public_url(url: str, max_redirects: int = 5) -> requests.Response | None:
    """GET a URL taken from untrusted ticket text (SSRF guard): https only, hosts must
    resolve exclusively to public IPs, redirects followed manually so each hop is re-checked."""
    for _ in range(max_redirects + 1):
        try:
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("not https")
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
            if not infos or not all(ipaddress.ip_address(i[4][0]).is_global for i in infos):
                raise ValueError("non-public address")
        except (socket.gaierror, ValueError, UnicodeError) as e:
            print(f"Blocked URL {url}: {e}")
            return None
        response = requests.get(url, timeout=15, stream=True, allow_redirects=False)
        if not response.is_redirect:
            return response
        url = urljoin(url, response.headers["Location"])
        response.close()
    print("Too many redirects")
    return None


def download_linked_document(url: str, ticket_dir: Path) -> tuple[str, list[Path]]:
    """Try to download a URL found in ticket text and treat it as a PDF or
    image based on the file's actual content (generic download endpoints often
    send a useless Content-Type). Returns (pdf_text, image_paths). Applies
    basic safety limits since this URL comes from untrusted ticket content:
    http(s) only, size-capped, short timeout."""
    try:
        response = fetch_public_url(url)
        if response is None:
            return "", []
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            print(f"Skipping linked document (too large): {url}")
            return "", []

        content = b""
        for chunk in response.iter_content(chunk_size=65536):
            content += chunk
            if len(content) > MAX_DOWNLOAD_BYTES:
                print(f"Aborting download (exceeded size limit): {url}")
                return "", []

        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()

        is_pdf = content.startswith(b"%PDF-") or content_type == "application/pdf" or url.lower().endswith(".pdf")
        is_image = content_type.startswith("image/") or content[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1")

        if is_pdf:
            ticket_dir.mkdir(parents=True, exist_ok=True)
            file_path = ticket_dir / "linked_document.pdf"
            file_path.write_bytes(content)
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            print(f"Downloaded and extracted linked PDF from {url} ({len(text)} chars)")
            return text, extract_pdf_images(reader, ticket_dir, "linked_document")

        if is_image:
            ticket_dir.mkdir(parents=True, exist_ok=True)
            ext = content_type.split("/")[-1] if content_type.startswith("image/") else "jpg"
            file_path = ticket_dir / f"linked_image.{ext}"
            file_path.write_bytes(content)
            print(f"Downloaded linked image from {url}")
            return "", [file_path]

        print(f"Skipping linked document (unsupported content type '{content_type}'): {url}")
        return "", []

    except requests.exceptions.RequestException as e:
        print(f"Failed to download linked document from {url}: {e}")
        return "", []


def extract_ticket_context(text: str) -> dict:
    """Use the model to pull clean fields out of noisy source text - a PDF
    work order OR a forwarded email chain (mixed languages, signatures,
    wrapper text, disclaimers). Same extraction logic works for both."""
    if not text.strip():
        return {}

    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": PDF_EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ],
        format="json",
        think=False,
        options={"num_ctx": 16384, "temperature": 0.2},
    )
    result = _parse_model_json(response.message.content) or {}
    return result


def normalize_mac(value) -> str | None:
    """Sanity-check a MAC address read from a photo. Vision models misread
    characters, so anything that isn't 12 hex digits is dropped, not trusted."""
    if not isinstance(value, str):
        return None
    match = MAC_PATTERN.search(value)
    if not match:
        return None
    hex_only = re.sub(r"[^0-9A-Fa-f]", "", match.group(0)).upper()
    if len(hex_only) != 12:
        return None
    return ":".join(hex_only[i:i + 2] for i in range(0, 12, 2))


def clean_image_analysis(analysis: dict) -> dict:
    """Normalize the vision model's JSON so the rest of the code can rely on it."""
    def clean_text(value):
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value or value.lower() in ("null", "none", "unknown", "n/a"):
            return None
        return value

    image_type = analysis.get("image_type")
    if image_type not in VALID_IMAGE_TYPES:
        image_type = "other"

    cleaned = {
        "file": analysis.get("file"),
        "image_type": image_type,
        "router_brand": clean_text(analysis.get("router_brand")),
        "router_model": clean_text(analysis.get("router_model")),
        "serial_number": clean_text(analysis.get("serial_number")),
        "mac_address": normalize_mac(analysis.get("mac_address")),
    }

    if image_type != "router_photo":
        for key in ("router_brand", "router_model", "serial_number", "mac_address"):
            cleaned[key] = None

    return cleaned


def analyze_images(image_paths: list[Path], ticket_id: int | None = None) -> list[dict]:
    """Classify each image, using cached results for a ticket where available."""
    state = load_ticket_state(ticket_id) if ticket_id is not None else {"image_analyses": {}}
    cached = state.setdefault("image_analyses", {})
    analyses = []
    for path in image_paths[:MAX_IMAGES_TO_ANALYZE]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in cached:
            analysis = cached[digest]
            analysis["file"] = path.name
            print(f"Using cached image analysis for {path.name}: {analysis}")
            analyses.append(analysis)
            continue
        try:
            response = ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": IMAGE_ANALYSIS_PROMPT},
                    {"role": "user", "content": "Analyze this image.", "images": [path.read_bytes()]},
                ],
                format="json",
                think=False,
                options={"num_ctx": 8192, "temperature": 0.1, "num_thread": 6},
            )
            raw = _parse_model_json(response.message.content) or {}
        except Exception as e:
            print(f"Image analysis failed for {path.name}: {e}")
            continue

        raw["file"] = path.name
        analysis = clean_image_analysis(raw)
        print(f"Image analysis for {path.name}: {analysis}")
        if ticket_id is not None:
            cached[digest] = analysis
        analyses.append(analysis)

    if ticket_id is not None:
        save_ticket_state(ticket_id, state)

    return analyses


def format_image_notes(analyses: list[dict], total_images: int) -> str:
    """Turn the analysis results into short notes for the classification model.
    Deliberately contains no MAC addresses or serial numbers - the reply only
    needs to know whether the label was readable, and the real values go to
    the private note for the helpdesk instead."""
    if total_images == 0:
        return "[No image attachments were included with this message. Do not claim to have seen a photo or screenshot.]"

    lines = []
    for number, analysis in enumerate(analyses, start=1):
        if analysis["image_type"] == "router_photo":
            has_identifiers = any(analysis[key] for key in ("router_brand", "router_model", "serial_number", "mac_address"))
            if has_identifiers:
                lines.append(f"[Image {number}: router photo received, identifying information on the label is readable]")
            else:
                lines.append(f"[Image {number}: router photo received, but no identifying information on the label is readable]")
        elif analysis["image_type"] == "speedtest":
            lines.append(f"[Image {number}: speedtest screenshot received]")
        else:
            lines.append(f"[Image {number}: other image, not a router photo or speedtest]")

    skipped = total_images - len(analyses)
    if skipped > 0:
        lines.append(f"[{skipped} more image(s) were attached but could not be analyzed.]")

    return "\n".join(lines)


def build_internal_note(analyses: list[dict]) -> str:
    """Private note for the helpdesk with the router details read from photos.
    Returns an empty string if there is nothing worth noting."""
    lines = []
    for number, analysis in enumerate(analyses, start=1):
        if analysis["image_type"] != "router_photo":
            continue
        details = [
            f"{label}: {analysis[key]}"
            for label, key in (("Brand", "router_brand"), ("Model", "router_model"), ("MAC", "mac_address"), ("Serial", "serial_number"))
            if analysis.get(key)
        ]
        detail_text = ", ".join(details) if details else "no identifying information readable"
        lines.append(f"Image {number} ({analysis['file']}): {detail_text}")

    if not lines:
        return ""
    return "Router details read from the attached image(s) by the AI - please verify against the photo:<br>" + "<br>".join(lines)


def detect_reply_language(pdf_context: dict, description: str) -> str:
    extracted = pdf_context.get("problem_description") or ""
    text = f"{extracted}\n{description or ''}".strip()

    if not text:
        return "English"

    lower_text = text.casefold()

    dutch_markers = (
        "ik", "mijn", "het", "een", "niet", "geen", "werkt",
        "probleem", "wifi", "internet", "woning", "adres",
        "bedankt", "hallo", "slecht", "traag", "verbinding",
    )
    english_markers = (
        "i", "my", "the", "a", "not", "doesn't", "problem",
        "works", "internet", "home", "address", "thank",
        "hello", "slow", "connection",
    )

    dutch_score = sum(
        1 for marker in dutch_markers
        if re.search(rf"\b{re.escape(marker)}\b", lower_text)
    )
    english_score = sum(
        1 for marker in english_markers
        if re.search(rf"\b{re.escape(marker)}\b", lower_text)
    )

    if dutch_score >= english_score + 2:
        return "Dutch"
    if english_score >= dutch_score + 2:
        return "English"

    try:
        detected = detect(text)
        return LANGUAGE_MAP.get(detected, "English")
    except Exception:
        return "English"


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
        blocks.append(f"- Known pattern: {doc}\n  Category: {meta['category']}\n  Guidance: {meta['guidance']}")

    return "Relevant known issues (for reference, use your judgment):\n" + "\n".join(blocks)


def classify_and_draft_reply(description: str, image_paths: list[Path], pdf_text: str = "", requester_name: str | None = None, ticket_id: int | None = None) -> dict:
    """Classify the ticket and draft a reply. Images are analyzed in their own
    focused pass first; the final call only sees short text notes about them,
    so it stays text-only (faster, and it can't hallucinate about images)."""
    retrieval_query = f"{description}\n{pdf_text}".strip()
    relevant_issues = retrieve_relevant_issues(retrieval_query)

    # Extract from the PDF if there is one, otherwise from the raw ticket
    # text itself - handles forwarded email chains the same way as PDFs.
    source_text = pdf_text if pdf_text.strip() else description
    context = extract_ticket_context(source_text)
    reply_language = detect_reply_language(context, description)
    location = context.get("location") or find_address_in_text(f"{description}\n{pdf_text}")

    image_analyses = analyze_images(image_paths, ticket_id)

    combined_text = f"{description}\n{pdf_text}"
    target_email, CC_emails = select_target_email(description, pdf_text)
    tenant_name = find_name_near_email(combined_text, target_email) or context.get("tenant_name") or requester_name

    full_name = context.get("tenant_full_name")
    if full_name and len(full_name.strip().split()) < 2:
        print(f"Rejecting tenant_full_name (not actually a full name): {full_name!r}")
        full_name = None

    missing_info = []
    message_text = f"{description}\n\n{format_image_notes(image_analyses, len(image_paths))}"
    if context.get("problem_description"):
        message_text += f"\n\n[Extracted problem description (use this, not the raw text above, as the tenant's actual issue): {context['problem_description']}]"
    if location:
        message_text += f"\n[Extracted address/location: {location}]"
    if context.get("permission_to_enter") and pdf_text.strip():
        message_text += f"\n[Extracted permission to enter home: {context['permission_to_enter']}]"
    if tenant_name:
        message_text += f"\n\n[Tenant name: {tenant_name}]"
    if full_name:
        message_text += f"\n[Extracted tenant full name (for intercom tickets): {full_name}]"
    if context.get("phone_number"):
        message_text += f"\n[Extracted phone number: {context['phone_number']}]"

    message_text += f"\n\n[REQUIRED REPLY LANGUAGE: {reply_language}. This has already been determined for you - write your entire reply in {reply_language}, regardless of any other language appearing elsewhere in this message.]"

    if relevant_issues:
        message_text += f"\n\n{relevant_issues}"

    unable_items = load_ticket_state(ticket_id).get("unable_to_provide", []) if ticket_id is not None else []
    state = load_ticket_state(ticket_id) if ticket_id is not None else {"unable_to_provide": []}
    unable_items = set(state.setdefault("unable_to_provide", []))
    unable_items.update(detect_unable_items(description))
    state["unable_to_provide"] = sorted(unable_items)
    if ticket_id is not None:
        save_ticket_state(ticket_id, state)

    if unable_items:
        message_text += f"\n\n[Customer was previously unable to provide these items; do not ask for them again: {', '.join(unable_items)}]"

    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": message_text},
        ],
        format="json",
        think=False,
        options={"num_ctx": 16384, "temperature": 0.1, "num_thread": 6},
    )

    thinking = getattr(response.message, "thinking", None)
    if thinking:
        print(f"Model reasoning (not sent to tenant): {thinking}")

    content = response.message.content

    result = _parse_model_json(content)
    if result is None or result.get("category") not in VALID_CATEGORIES or "reply" not in result:
        print(f"Model returned unexpected output, falling back to human handoff. Parsed as: {result}")
        return {"category": "other", "missing_info": [], "reply": "Hi, thanks for reaching out. Our team will contact you soon."}

    if not isinstance(result.get("missing_info"), list):
        result["missing_info"] = []

    state = load_ticket_state(ticket_id) if ticket_id is not None else {"unable_to_provide": []}
    unable_items = set(state.setdefault("unable_to_provide", []))
    unable_items.update(detect_unable_items(description))
    state["unable_to_provide"] = sorted(unable_items)
    if ticket_id is not None:
        save_ticket_state(ticket_id, state)

    category = result.get("category")
    provided_items = set()
    if location:
        provided_items.add("address/location")
    if any(a["image_type"] == "router_photo" for a in image_analyses):
        provided_items.add("router photo")
    if any(a["image_type"] == "speedtest" for a in image_analyses):
        provided_items.add("speedtest screenshot")
    if full_name:
        provided_items.add("tenant full name")
    if context.get("phone_number"):
        provided_items.add("phone number")

    provided_items.update(unable_items)

    if category == "wifi":
        required_items = {"address/location", "router photo", "speedtest screenshot"}
    elif category == "intercom":
        required_items = {"address/location", "tenant full name", "phone number"}
    elif category == "tv":
        required_items = {"address/location"}
    else:
        required_items = set()

    deterministic_missing = sorted(required_items - provided_items)
    result["missing_info"] = deterministic_missing
    if ticket_id is not None:
        save_ticket_state(ticket_id, state)

    message_text_for_reply = (
        "Only ask for these still-missing items: " + ", ".join(deterministic_missing)
        if deterministic_missing
        else "No required information remains outstanding. Do not ask any questions. Confirm that all required information has been received, including anything the customer said they cannot provide. Then, if the ticket contains reliable router identification or other relevant evidence, offer one concise and safe troubleshooting step grounded in that evidence. Do not invent model-specific instructions or repeat a generic issue acknowledgement. Otherwise say the helpdesk will follow up."
    )
    router_details = [
        f"{analysis.get('router_brand') or ''} {analysis.get('router_model') or ''}".strip()
        for analysis in image_analyses
        if analysis.get("image_type") == "router_photo"
        and (analysis.get("router_brand") or analysis.get("router_model"))
    ]
    if router_details:
        message_text_for_reply += " Identified router information: " + ", ".join(router_details) + "."

    try:
        followup = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"{message_text}\n\n[FINAL REPLY REQUIREMENTS: {message_text_for_reply}]"},
            ],
            format="json",
            think=False,
            options={"num_ctx": 16384, "temperature": 0.1, "num_thread": 6},
        )
        followup_result = _parse_model_json(followup.message.content)
        if followup_result and isinstance(followup_result.get("reply"), str):
            result["reply"] = followup_result["reply"]
    except Exception as error:
        print(f"Failed to format final information reply: {error}")

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



def select_target_email(description: str, pdf_text: str) -> tuple[str | None, list[str]]:
    """Choose the first source email that is not on the forbidden lists,
    and return the remaining eligible ones as CC candidates.

    The source text is authoritative; the language model is not asked to
    guess which address should receive the reply.
    """
    combined_text = f"{description}\n{pdf_text}"
    candidates = []
    seen = set()

    for match in EMAIL_PATTERN.findall(combined_text):
        email = match.strip(".,;:()[]<>").lower()
        if email in seen:
            continue
        seen.add(email)
        candidates.append(email)

    allowed = []
    for email in candidates:
        domain = email.rsplit("@", 1)[1]
        if not any(domain.endswith(tld) for tld in ALLOWED_EMAIL_TLDS):
            continue
        if email in FORBIDDEN_EMAIL_ADDRESSES or domain in FORBIDDEN_EMAIL_DOMAINS:
            continue
        allowed.append(email)

    target_email = allowed[0] if allowed else None
    print(f"Eligible target emails: {allowed}; selected target email: {target_email!r}")
    CC_emails = allowed[1:] if len(allowed) > 1 else []
    return target_email, CC_emails


def normalize_address(address: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace - so formatting
    differences between how a tenant writes their address and how it's
    stored in the reference list don't cause false negatives."""
    normalized = address.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def find_address_in_text(text: str) -> str | None:
    """Deterministic fallback for when the extraction model misses the address.
    Returns the street and house number as written, or None."""
    match = STREET_ADDRESS_PATTERN.search(text or "")
    return f"{match.group(1)} {match.group(2)}" if match else None


def is_non_configurable_intercom(address: str | None) -> bool:
    """Check the tenant's address against the known list of addresses whose
    intercom cannot be configured remotely. Exact-list lookup, not semantic
    matching - this needs to be reliable, not "close enough"."""
    if not address:
        return False

    try:
        known_addresses = json.loads(NON_CONFIGURABLE_ADDRESSES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False

    ticket_key = find_address_in_text(address)
    for known in known_addresses:
        known_key = find_address_in_text(known)
        if ticket_key and known_key:
            # Compare street plus house number, so "763" never matches "76" or "7630".
            # Any city name after the number is ignored.
            if normalize_address(ticket_key) == normalize_address(known_key):
                return True
        elif normalize_address(known) in normalize_address(address):
            # Entries without a recognised street suffix fall back to the old check.
            return True
    return False


def lookup_tv_manual(address: str | None) -> tuple[str | None, Path | None]:
    """Match an address against the TV system lists, same pattern as
    is_non_configurable_intercom(). Returns (system, manual_path), or
    (None, None) if no match."""
    if not address:
        return None, None
    try:
        tv_data = json.loads(TV_ADDRESSES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None

    normalized_address = normalize_address(address)
    for known in tv_data.get("smart_iptv", []):
        if normalize_address(known) in normalized_address or normalized_address in normalize_address(known):
            return "smart_iptv", TV_MANUALS_DIR / "Instructie SMART IPTV(2025).pdf"
    for known, filename in tv_data.get("ssiptv", {}).items():
        if normalize_address(known) in normalized_address or normalized_address in normalize_address(known):
            return "ssiptv", TV_MANUALS_DIR / filename
    return None, None


def get_router_tip(image_analyses: list[dict], reply_language: str) -> str | None:
    """Look up a fixed, pre-translated troubleshooting tip if a router photo's
    brand was identified. Fixed wording, not model-generated - this is exactly
    the kind of "must be worded precisely" content (e.g. Mikrotik's
    reset-button warning) that shouldn't depend on a model rephrasing it
    correctly under temperature/translation each time."""
    try:
        tips = json.loads(ROUTER_TIPS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None

    for analysis in image_analyses:
        if analysis.get("image_type") != "router_photo":
            continue
        brand = (analysis.get("router_brand") or "").strip().lower()
        if not brand:
            continue
        for known_brand, translations in tips.items():
            if known_brand in brand or brand in known_brand:
                return translations.get(reply_language) or translations.get("English")
    return None


def insert_tip_before_closing(reply_text: str, tip: str) -> str:
    """Splice the tip in right before the closing line, rather than asking
    the model to place it - the closing lines are fixed, known strings
    (enforced by the prompt), so we can find them reliably."""
    for closing in ("Met vriendelijke groet,\nSupport team Solutio365", "Greetings,\nSupport team Solutio365"):
        if closing in reply_text:
            return reply_text.replace(closing, f"{tip}\n\n{closing}")
    return f"{reply_text}\n\n{tip}"


def find_name_near_email(text: str, email: str | None) -> str | None:
    """Forwarded emails almost always include a 'Name <email>' header line
    (e.g. 'Van: Yordan Rusev <rusev3005@gmail.com>'). This is a highly
    reliable, deterministic signal when present - check it before falling
    back to the model's own extraction."""
    if not email:
        return None
    pattern = re.compile(NAME_BEFORE_EMAIL_PATTERN_TEMPLATE.replace("__EMAIL__", re.escape(email)), re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else None



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

     attachments = list(ticket.get("attachments", []) or [])

     html_bodies = [ticket.get("description", "") or ""]

     conversations_response = requests.get(
         f"{BASE_URL}/tickets/{ticket_id}/conversations",
         auth=AUTH,
         timeout=15,
     )
     conversations_response.raise_for_status()
     for conversation in conversations_response.json():
         if conversation.get("private"):
             continue
         html_bodies.append(conversation.get("body") or "")
         attachments.extend(conversation.get("attachments", []) or [])

     inline_image_urls = []
     seen_urls = set()
     for html_body in html_bodies:
         for url in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', unescape(html_body), flags=re.I):
             url = unescape(url).strip()
             if url and url not in seen_urls:
                 seen_urls.add(url)
                 inline_image_urls.append(url)

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
        name = re.sub(r"[^\w.\-]", "_", Path(attachment.get("name") or "attachment").name).lstrip(".") or "attachment"
        if not url:
            continue
        try:
            img_response = requests.get(url, auth=AUTH, timeout=15)
            img_response.raise_for_status()
            ticket_dir.mkdir(parents=True, exist_ok=True)
            file_path = ticket_dir / name
            file_path.write_bytes(img_response.content)
            saved_paths.append(file_path)
            print(f"Saved image attachment: {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download attachment {name}: {e}")

    for i, url in enumerate(inline_image_urls):
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
        except ValueError:
            continue
        if parsed.scheme != "https" or host not in ALLOWED_INLINE_IMAGE_HOSTS:
            print(f"Skipping inline image from non-Freshdesk host: {parsed.hostname}")
            continue
        try:
            img_response = requests.get(url, timeout=15)
            img_response.raise_for_status()
            ticket_dir.mkdir(parents=True, exist_ok=True)
            file_path = ticket_dir / f"inline_{i}.png"
            file_path.write_bytes(img_response.content)
            saved_paths.append(file_path)
            print(f"Saved inline image: {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download inline image from {url}: {e}")

    return saved_paths


def cleanup_ticket_files(ticket_id: int):
    """Delete the attachment folder for a ticket after processing."""
    ticket_dir = ATTACHMENTS_DIR / str(ticket_id)

    if ticket_dir.exists():
        shutil.rmtree(ticket_dir, ignore_errors=True)
        print(f"Deleted attachment folder: {ticket_dir}")

def add_private_note(ticket_id: int, note_html: str) -> None:
    """Add an internal note only agents can see (the tenant never receives it)."""
    response = requests.post(
        f"{BASE_URL}/tickets/{ticket_id}/notes",
        auth=AUTH,
        json={"body": note_html, "private": True},
        timeout=15,
    )
    response.raise_for_status()


def reply_to_ticket(ticket_id: int, message_html: str, assign: int, target_email: str | None, special_agent: int | None = None, CC_emails: list[str] | None = None, attachment_path: Path | None = None) -> dict:
    """POST a public reply and assign by the caller's explicit routing value."""
    message_text = message_html.replace("\n", "<br>")

    if attachment_path and attachment_path.exists():
        with open(attachment_path, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/tickets/{ticket_id}/reply",
                auth=AUTH,
                data={"body": message_text},
                files=[("attachments[]", (attachment_path.name, f))],
                timeout=30,
            )
        response.raise_for_status()
        assign_response = requests.put(f"{BASE_URL}/tickets/{ticket_id}", auth=AUTH, json={"responder_id": assign}, timeout=15)
        assign_response.raise_for_status()
        return {"reply": response.json(), "assign": assign_response.json()}

    # TESTING OVERRIDE - forces all outgoing mail to your own address regardless
    # of what target_email logic below would otherwise pick. Remove these lines,
    # keep the real logic beneath them, once you're done testing.
    send_to = TEST_EMAIL_OVERRIDE if target_email else None
    cc_list = [TEST_EMAIL_OVERRIDE] if (CC_emails and TEST_EMAIL_OVERRIDE) else []
    # send_to = target_email  # <- real logic, re-enable this once override is removed
    # cc_list = CC_emails or []  # <- real logic, re-enable this once override is removed

    if send_to:
        response = requests.post(
            f"{BASE_URL}/tickets/{ticket_id}/reply_to_forward",
            auth=AUTH,
            json={"body": message_text, "to_emails": [send_to], "cc_emails": cc_list},
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

    assign_response = requests.put(
        f"{BASE_URL}/tickets/{ticket_id}",
        auth=AUTH,
        json={"responder_id": assign},
        timeout=15,
    )
    assign_response.raise_for_status()

    return {"reply": response.json(), "assign": assign_response.json()}


def process_ticket(ticket_id: int, requester_email: str, requester_name: str, description: str) -> None:
    """Runs the slow AI classification + Freshdesk reply after the webhook
    has already been acknowledged, so Freshdesk/the sender never times out
    waiting on the model."""
    full_description = get_full_ticket_context(ticket_id, description)
    if full_description:
        description = full_description

    attachments, inline_image_urls = get_ticket_attachments(ticket_id)
    image_paths = download_image_attachments(ticket_id, attachments, inline_image_urls)

    pdf_text, pdf_images = download_and_extract_pdfs(ticket_id, attachments)
    image_paths.extend(pdf_images)

    if not pdf_text.strip():
        ticket_dir = ATTACHMENTS_DIR / str(ticket_id)
        for url in find_download_links(description)[:3]:
            linked_text, linked_images = download_linked_document(url, ticket_dir)
            if linked_text:
                pdf_text += f"\n\n{linked_text}"
            image_paths.extend(linked_images)

    print(f"Ticket {ticket_id}: {len(image_paths)} image(s) total in {ATTACHMENTS_DIR / str(ticket_id)}")

    decision = classify_and_draft_reply(description, image_paths, pdf_text, requester_name, ticket_id)

    image_analyses = decision.get("_image_analyses", [])
    state = load_ticket_state(ticket_id)
    analysis_cache = state.get("image_analyses", {})
    noted_hashes = set(state.get("noted_router_image_hashes", []))
    new_router_analyses = [
        {**analysis, "file": analysis.get("file") or digest[:12]}
        for digest, analysis in analysis_cache.items()
        if digest not in noted_hashes and analysis.get("image_type") == "router_photo"
    ]
    router_note = build_internal_note(new_router_analyses)
    if router_note:
        try:
            add_private_note(ticket_id, router_note)
            noted_hashes.update(
                digest
                for digest, analysis in analysis_cache.items()
                if analysis.get("image_type") == "router_photo"
            )
            state["noted_router_image_hashes"] = sorted(noted_hashes)
            save_ticket_state(ticket_id, state)
            print(f"Added private router note to ticket {ticket_id} for {len(new_router_analyses)} new router photo(s)")
        except requests.exceptions.RequestException as error:
            print(f"Failed to add private router note to ticket {ticket_id}: {error}")
    elif not image_analyses:
        print(f"Ticket {ticket_id}: no image analyses available for router note")
    else:
        print(f"Ticket {ticket_id}: no new router photos identified for private note")

    if decision["category"] == "ignore":
        print(f"Ticket {ticket_id}: classified as ignore, no reply or assignment.")
        cleanup_ticket_files(ticket_id)
        return

    reply_language = decision.get("_reply_language", "English")
    ooo_prefix = is_out_of_office()

    missing_info = decision.get("missing_info", [])
    if not isinstance(missing_info, list):
        missing_info = []
    assign = SUPPORT_AGENT_INTERCOM_ID if missing_info else SUPPORT_EMPLOYEE_ID
    routing_reason = "missing information" if missing_info else "all required information provided"
    print(f"Ticket {ticket_id}: routing to responder {assign} ({routing_reason})")

    target_email = decision.get("_target_email")
    CC_emails = decision.get("_CC_emails", [])

    if decision["category"] == "tv":
        address = decision.get("_extracted_address")
        name = decision.get("_tenant_name")
        greeting = (f"Beste {name}," if reply_language == "Dutch" else f"Hi {name},") if name else ("Beste," if reply_language == "Dutch" else "Hi,")
        closing = "Met vriendelijke groet,\nSupport team Solutio365" if reply_language == "Dutch" else "Greetings,\nSupport team Solutio365"
        templates = json.loads(TV_TEMPLATES_PATH.read_text(encoding="utf-8"))

        system, manual_path = (None, None)
        if not address:
            body = templates["need_address"][reply_language]
        else:
            system, manual_path = lookup_tv_manual(address)
            if system is None:
                body = templates["address_not_recognized"][reply_language]
            elif "galvanistraat" in address.lower() or "lee towers" in address.lower():
                number_match = re.search(r"\d+", address)

                if number_match:
                    address_number = int(number_match.group())
                    tower = 2 if address_number > 1090 else 1
                else:
                    tower = None

                tower_note = (
                    f"\n\nNote: Your address says you live in tower {tower}, "
                    "so please pay attention to that when following the instructions."
                    if tower
                    else ""
                )

                body = templates[system][reply_language] + tower_note
            else:
                body = templates[system][reply_language]

        reply_message = f"{greeting}\n\n{body}\n\n{closing}"
        if ooo_prefix:
            reply_message = insert_tip_before_closing(reply_message, OUT_OF_OFFICE_TEXT[reply_language])
        attach = manual_path if address and system else None
        reply_to_ticket(ticket_id, reply_message, assign=assign, target_email=target_email, attachment_path=attach)
        print(f"Ticket {ticket_id}: TV ticket handled ({'manual sent: ' + system if address and system else 'asked for/re-asked address'}).")
        if not missing_info:
            cleanup_ticket_files(ticket_id)
        else:
            print(f"Ticket {ticket_id}: keeping attachment folder until missing information is provided.")
        return

    reply_message = decision["reply"]
    if ooo_prefix:
        reply_message = insert_tip_before_closing(reply_message, OUT_OF_OFFICE_TEXT[reply_language])

    if decision["category"] == "wifi":
        tip = get_router_tip(decision.get("_image_analyses", []), decision.get("_reply_language", "English"))
        if tip:
            reply_message = insert_tip_before_closing(reply_message, tip)
            print(f"Added router troubleshooting tip to ticket {ticket_id}")

    try:
        reply_to_ticket(ticket_id, reply_message, assign, target_email, None, CC_emails)
        target = "intercom collection agent" if missing_info else "support employee"
        print(f"Replied to ticket {ticket_id} (requester: {requester_email}) and assigned it to the {target}.")
    except requests.exceptions.HTTPError as e:
        print(f"Failed to reply to ticket {ticket_id}: {e.response.text}")

    if missing_info:
        print(f"Ticket {ticket_id}: keeping attachment folder until missing information is provided.")
    else:
        cleanup_ticket_files(ticket_id)


@app.post("/freshdesk-webhook", status_code=202)
async def receive_ticket(request: Request, background_tasks: BackgroundTasks, authorized: bool = Depends(verify_webhook_auth)):
    payload = await request.json()
    print(f"Received webhook payload: {payload}")

    ticket_id = payload.get("ticket_id")
    requester_email = payload.get("requester_email") or ""
    requester_name = payload.get("requester_name") or ""
    description = payload.get("description_text") or ""

    background_tasks.add_task(process_ticket, int(ticket_id), requester_email, requester_name, description)

    return {"status": "accepted", "ticket_id": int(ticket_id)}

def get_full_ticket_context(ticket_id: int, webhook_description: str) -> str:
    """Combine the original ticket and public replies for reprocessing after a customer reply."""
    context_parts = []

    if webhook_description:
        context_parts.append(webhook_description.strip())

    try:
        ticket_response = requests.get(f"{BASE_URL}/tickets/{ticket_id}", auth=AUTH, timeout=15)
        ticket_response.raise_for_status()
        ticket = ticket_response.json()

        description_html = ticket.get("description") or ""
        if description_html:
            description_text = re.sub(r"<br\s*/?>", "\n", description_html, flags=re.I)
            description_text = re.sub(r"</p>", "\n", description_text, flags=re.I)
            description_text = unescape(re.sub(r"<[^>]+>", "", description_text)).strip()
            if description_text:
                context_parts.insert(0, description_text)

        conversations_response = requests.get(
            f"{BASE_URL}/tickets/{ticket_id}/conversations",
            auth=AUTH,
            timeout=15,
        )
        conversations_response.raise_for_status()
        for conversation in conversations_response.json():
            if conversation.get("private"):
                continue
            body = conversation.get("body_text") or conversation.get("body") or ""
            if "<" in body:
                body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
                body = re.sub(r"</p>", "\n", body, flags=re.I)
                body = unescape(re.sub(r"<[^>]+>", "", body)).strip()
            else:
                body = unescape(body).strip()
            if body:
                context_parts.append(body)
    except requests.exceptions.RequestException as error:
        print(f"Failed to fetch full ticket context for {ticket_id}: {error}")

    unique_parts = []
    seen = set()
    for part in context_parts:
        normalized = re.sub(r"\s+", " ", part).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_parts.append(part)
    return "\n\n".join(unique_parts)

def ticket_state_path(ticket_id: int) -> Path:
    return ATTACHMENTS_DIR / str(ticket_id) / "ticket_state.json"


def load_ticket_state(ticket_id: int) -> dict:
    try:
        path = ticket_state_path(ticket_id)
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(state, dict):
                return state
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        print(f"Could not load ticket state for {ticket_id}: {error}")
    return {"unable_to_provide": [], "image_analyses": {}}


def save_ticket_state(ticket_id: int, state: dict) -> None:
    path = ticket_state_path(ticket_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_path, path)


def detect_unable_items(text: str) -> set[str]:
    text = (text or "").lower()
    unable = set()

    patterns = {
        "speedtest screenshot": (
            r"(?:cannot|can't|cant|unable to|no way to|can't give|cant give|cannot give|can't provide|cannot provide).*(?:speedtest)",
            r"(?:no|without|don't have|do not have|don't posses|doesn't have|cannot have|can't have).*(?:speedtest)",
            r"(?:speedtest|speed test).{0,100}(?:not possible|impossible|not available|cannot be done|can't be done|not feasible)",
            r"(?:not possible|impossible|not available|cannot be done|can't be done|not feasible).{0,100}(?:speedtest|speed test)",
            r"(?:there is|it's|it is) no way to.{0,100}(?:speedtest|speed test)",
            r"(?:geen|kan geen|niet in staat).*(?:speedtest)",
        ),
        "router photo": (
            r"(?:cannot|can't|cant|unable to|no way to|can't give|cant give|cannot give|can't provide|cannot provide).*(?:router photo|photo of the router|picture of the router)",
            r"(?:geen|kan geen|niet in staat).*(?:routerfoto|foto van de router)",
        ),
        "address/location": (
            r"(?:cannot|can't|cant|unable to|no way to).*(?:address|location)",
            r"(?:geen|kan geen|niet in staat).*(?:adres)",
        ),
        "tenant full name": (
            r"(?:cannot|can't|cant|unable to|no way to).*(?:full name|name and surname|first and last name)",
            r"(?:geen|kan geen|niet in staat).*(?:volledige naam)",
        ),
        "phone number": (
            r"(?:cannot|can't|cant|unable to|no way to).*(?:phone|telephone|mobile)",
            r"(?:geen|kan geen|niet in staat).*(?:telefoonnummer|telefoon)",
        ),
    }

    for item, expressions in patterns.items():
        if any(re.search(pattern, text, re.I) for pattern in expressions):
            unable.add(item)
    return unable