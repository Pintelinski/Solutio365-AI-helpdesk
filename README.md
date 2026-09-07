# Solutio365-AI-helpdesk
Internship project where I need to make an open-source AI agent that will handle low complexity support tickets with automated replies handling the initial communication exchange to reduce response time and reduce unnecessary mailing back and forth


## Setup

1. Install the dependencies:

	```powershell
	pip install -r requirements.txt
	```

2. Set the Freshdesk environment variables in the same terminal:

	```powershell
	$env:FRESHDESK_DOMAIN = "yourcompany"
	$env:FRESHDESK_API_KEY = "your-api-key"
	```

	`FRESHDESK_PASSWORD` is optional; Freshdesk accepts any password value when authenticating with an API key.

3. Start the webhook server:

	```powershell
	uvicorn freshdesk_agent:app --host 0.0.0.0 --port 8085
	```

Freshdesk should send its automation webhook to:

```text
http://<your-public-host>:8000/freshdesk-webhook
```

The endpoint accepts a top-level `ticket_id` or a nested `ticket.id`, retrieves the full ticket through the Freshdesk API, and returns the received ticket information. The server must be publicly reachable; for local testing, use a tunnel such as ngrok or Cloudflare Tunnel.