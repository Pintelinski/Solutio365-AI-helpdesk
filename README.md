# Solutio365-AI-helpdesk
Internship project where I need to make an open-source AI agent that will handle low complexity support tickets with automated replies handling the initial communication exchange to reduce response time and reduce unnecessary mailing back and forth


## Code Setup

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
	uvicorn freshdesk_agent:app --host 0.0.0.0 --port 8085 --env-file .env
	```

Freshdesk should send its automation webhook to:

```text
http://<your-public-host>:8000/freshdesk-webhook
```

The endpoint accepts a top-level `ticket_id` or a nested `ticket.id`, retrieves the full ticket through the Freshdesk API, and returns the received ticket information. The server must be publicly reachable; for local testing, use a tunnel such as ngrok or Cloudflare Tunnel.

## Server Setup

1. Install Linux Ubuntu 24.04 LTS server (Tip: setup OpenSSH for easy command copy and paste)

2. Make your Ubuntu ready to be configured:

    ```cli
    sudo apt update && sudo apt upgrade -y (This could take a while)
    ```

3. Install Ollama (AI agent model runner):

    ```cli
    curl -fsSL https://ollama.com/install.sh | sh
    ```

4. Check if Ollama installed correctly:

    ```cli
    systemctl status ollama
    ```

5. Check the https://ollama.com/library for your desired AI model and note down its tag.

6. Pull your desired model and then let it run:

    ```
    ollama pull <your-desired-model-tag>
    ollama run <your-desired-model-tag>
    ```

7. Run a test prompt to see performance and response quality

You now installed an AI on your own server. You will be able to interact with this in the same way you would with other LLM models.

We wil now go over to installing all python dependencies to eventually pull and run the support AI agent code where we would also be able to receive and send the API requests from and to Freshdesk

1. Install python essentials on the server:

    ```cli
    sudo apt install python3 python3-pip python3-venv -y
    ```