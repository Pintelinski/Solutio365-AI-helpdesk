from fastapi import FastAPI, HTTPException, Request, Depends
import os
import secrets
from contextlib import asynccontextmanager

import requests
import ngrok
from fastapi.security import HTTPBasic, HTTPBasicCredentials


FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
BASE_URL = (f"https://{FRESHDESK_DOMAIN}.freshdesk.com/api/v2")
FRESHDESK_PASSWORD = os.getenv("FRESHDESK_PASSWORD", "X")

AUTH = (FRESHDESK_API_KEY, FRESHDESK_PASSWORD)

WEBHOOK_USERNAME = os.getenv("WEBHOOK_USERNAME")
WEBHOOK_PASSWORD = os.getenv("WEBHOOK_PASSWORD")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN")

def verify_webhook_auth(credentials: HTTPBasicCredentials = Depends(HTTPBasic)) -> bool:
    """Checks the Basic Auth header Freshdesk sends against our own secret.
    This has nothing to do with the Freshdesk API key - it's a credential
    you invent yourself and enter in the Freshdesk Automation's
    'Authentication' field."""
    if not WEBHOOK_USERNAME or not WEBHOOK_PASSWORD:
        raise RuntimeError("WEBHOOK_USERNAME and WEBHOOK_PASSWORD must be configured")
 
    valid_user = secrets.compare_digest(credentials.username, WEBHOOK_USERNAME)
    valid_pass = secrets.compare_digest(credentials.password, WEBHOOK_PASSWORD)
    if not (valid_user and valid_pass):
        raise HTTPException(status_code=401, detail="Invalid webhook credentials")
    return True


def get_ticket_details(ticket_id: int) -> dict:
    """Fetch full ticket information from Freshdesk."""
    if not BASE_URL or not FRESHDESK_API_KEY:
        raise RuntimeError("FRESHDESK_DOMAIN and FRESHDESK_API_KEY must be configured")

    response = requests.get(f"{BASE_URL}/tickets/{ticket_id}", auth=AUTH, timeout=15)
    response.raise_for_status()
    return response.json()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN", ""))
    forward_kwargs = {"addr": "127.0.0.1:8085", "authtoken_from_env": True}
    if NGROK_DOMAIN:
        forward_kwargs["domain"] = NGROK_DOMAIN
    forwarder = await ngrok.forward(**forward_kwargs)
    print(f"Public webhook URL: {forwarder.url()}/freshdesk-webhook")
    yield
    ngrok.disconnect()

app = FastAPI(lifespan=lifespan)

@app.post("/freshdesk-webhook", status_code=202)
async def receive_ticket(request: Request):
    payload = await request.json()
    print(f"Received webhook payload: {payload}")
    ticket = payload.get("ticket", payload) if isinstance(payload, dict) else {}

    if not isinstance(ticket, dict):
        raise HTTPException(status_code=400, detail="Webhook payload must contain a ticket object")

    ticket_id = ticket.get("ticket_id") or ticket.get("id")
    if ticket_id is None:
        raise HTTPException(status_code=400, detail="Webhook payload is missing ticket_id")

    requester = ticket.get("requester") or {}
    ticket_details = get_ticket_details(int(ticket_id))

    return {
        "status": "received",
        "ticket_id": int(ticket_id),
        "description": ticket.get("description_text") or ticket.get("description"),
        "email": ticket.get("email") or requester.get("email"),
        "name": ticket.get("name") or requester.get("name"),
        "ticket": ticket_details,
    }