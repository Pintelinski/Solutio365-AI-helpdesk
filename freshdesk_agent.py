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
SUPPORT_AGENT_ID = int(os.getenv("SUPPORT_AGENT_ID"))

AUTH = (FRESHDESK_API_KEY, FRESHDESK_PASSWORD)

WEBHOOK_USERNAME = os.getenv("WEBHOOK_USERNAME")
WEBHOOK_PASSWORD = os.getenv("WEBHOOK_PASSWORD")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN")


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

def reply_to_ticket(ticket_id: int, message_html: str) -> dict:
    """POST a public reply to a Freshdesk ticket - this is what emails the requester."""
    if not BASE_URL or not FRESHDESK_API_KEY:
        raise RuntimeError("FRESHDESK_DOMAIN and FRESHDESK_API_KEY must be configured")
 
    response = requests.post(
        f"{BASE_URL}/tickets/{ticket_id}/reply",
        auth=AUTH,
        json={"body": message_html},
        timeout=15,
    )
    assign = requests.put(
        f"{BASE_URL}/tickets/{ticket_id}",
        auth=AUTH,
        json={"responder_id": SUPPORT_AGENT_ID},
        timeout=15,
    )
    response.raise_for_status()
    assign.raise_for_status()
    return {"reply": response.json(), "assign": assign.json()}


@app.post("/freshdesk-webhook", status_code=202)
async def receive_ticket(request: Request):
    payload = await request.json()
    print(f"Received webhook payload: {payload}")

    ticket_id = payload.get("ticket_id")
    requester_email = payload.get("requester_email")
    requester_name = payload.get("requester_name")

    reply_message = (
        f"Hi {requester_name}, thanks for reaching out. Could you send us a screenshot of a "
        "speedtest, your address, and your IP address so we can look into this?"
    )
 
    try:
        reply_to_ticket(int(ticket_id), reply_message)
        print(f"Replied to ticket {ticket_id} (requester: {requester_email}) and assigned it to the AI agent.")
    except requests.exceptions.HTTPError as e:
        print(f"Failed to reply to ticket {ticket_id}: {e.response.text}")
        raise HTTPException(status_code=502, detail="Failed to send reply via Freshdesk")
 
    return {"status": "replied", "ticket_id": int(ticket_id), "email": requester_email}