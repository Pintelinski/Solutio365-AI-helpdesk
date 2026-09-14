# Solutio365-AI-helpdesk
Internship project where I need to make an open-source AI agent that will handle low complexity support tickets with automated replies handling the initial communication exchange to reduce response time and reduce unnecessary mailing back and forth


## Code Setup

### Install Dependencies

Create and activate a virtual environment, then install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The application uses FastAPI/Uvicorn for the webhook, `requests` for Freshdesk API calls, the ngrok Python SDK for the tunnel, `python-dotenv` for local configuration, and the Ollama Python client for local model inference.

### Configure Environment

Copy the example configuration and fill in the real values:

```powershell
Copy-Item .env.example .env
```

Required Freshdesk and webhook settings:

```dotenv
FRESHDESK_DOMAIN=yourcompany
FRESHDESK_API_KEY=your-api-key
FRESHDESK_PASSWORD=X
SUPPORT_AGENT_ID=1234567890
SUPPORT_EMPLOYEE_ID=1234567891
WEBHOOK_USERNAME=webhook-username
WEBHOOK_PASSWORD=webhook-password
```

The webhook username and password are credentials you create for Freshdesk's webhook authentication. They are separate from the Freshdesk API key. The two support IDs are Freshdesk responder IDs: one for the AI agent and one for the human employee.

For the embedded ngrok tunnel, also configure:

```dotenv
NGROK_DOMAIN=your-free-ngrok-domain.ngrok-free.dev
NGROK_AUTHTOKEN=your-ngrok-authtoken
```

For Ollama, configure the host and model that are available to the machine running this application:

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b
```

The file `AI-classification-setup/classification_system_prompt.txt` must exist. It defines the JSON categories and reply rules used by the model. Keep `.env` out of version control and never commit API keys, passwords, or tokens.

### Freshdesk Webhook

Configure a Freshdesk automation that triggers when a ticket is created and sends a `POST` request to:

```text
https://your-ngrok-domain.ngrok-free.dev/freshdesk-webhook
```

Use the webhook username and password from `.env` for Basic Authentication, set the content type to `application/json`, and use this body:

```json
{
    "ticket_id": "{{ticket.id}}",
    "subject": "{{ticket.subject}}",
    "description_text": "{{ticket.description_text}}",
    "requester_email": "{{ticket.requester.email}}",
    "requester_name": "{{ticket.requester.name}}",
    "priority": "{{ticket.priority}}"
}
```

The endpoint validates the JSON object and ticket ID, prints the received payload, and returns `202 Accepted`. It then processes the ticket in a background task so Freshdesk does not wait for Ollama or the Freshdesk API operations.

### Processing Flow

For each accepted webhook, the application:

1. Fetches the complete Freshdesk ticket.
2. Finds traditional image attachments and inline images in the HTML description.
3. Downloads those images into `attachments/<ticket-id>/`.
4. Sends the ticket description and available images to Ollama.
5. Expects one of four categories: `wifi_info_needed`, `wifi_resolved`, `wifi_escalate`, or `other`.
6. Posts the model's public reply to Freshdesk.
7. Assigns the ticket to `SUPPORT_AGENT_ID` for `wifi_info_needed` and `wifi_resolved`; all other categories go to `SUPPORT_EMPLOYEE_ID`.

The reply and assignment are outbound Freshdesk API requests, so Uvicorn logs the incoming webhook as `POST /freshdesk-webhook`; it does not log the outbound Freshdesk `POST` and `PUT` as server requests.

### Local Checks

Check that the app imports and compiles:

```powershell
python -m py_compile freshdesk_agent.py
```

After the server is running, FastAPI documentation is available at:

```text
http://127.0.0.1:8085/docs
```

The public URL printed during startup must be reachable before configuring it in Freshdesk. If a webhook body is invalid JSON, the application logs its raw bytes and returns `400`. If Freshdesk API or attachment processing fails after acknowledgement, the details are printed in the server terminal.

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

2. Make your directory where you would want the AI agent to be:

    ```cli
    mkdir ~/<your-directory-name>
    ```

3. Go into your directory and clone the Git repository into your directory and setup the environmental variables:

    ```cli
    cd ~/<your-directory-name>
    git clone https://github.com/Pintelinski/Solutio365-AI-helpdesk 
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env
    nano .env #fill in your actual environmental variables in the places of the placeholders
    ```

4. Your app should be ready to run, so we can run it with the following command:

    ```cli
    uvicorn freshdesk_agent:app --host 0.0.0.0 --port 8085 --env-file .env
    ```