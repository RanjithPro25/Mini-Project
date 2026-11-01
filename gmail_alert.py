import os
import base64
import mimetypes
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

BASE_DIR = Path(__file__).resolve().parent
AUTH_DIR = BASE_DIR / "gmail_notification"
CREDENTIALS_FILE = next((p for p in (AUTH_DIR / "credentials.json", BASE_DIR / "credentials.json") if p.exists()), AUTH_DIR / "credentials.json")
TOKEN_FILE = next((p for p in (AUTH_DIR / "token.json", BASE_DIR / "token.json") if p.exists()), AUTH_DIR / "token.json")
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
# Used only by the standalone manual test; app alerts always pass saved contact emails.
CONTACT_EMAIL = "venomguard26@gmail.com"


def get_gmail_service():
    credentials = None
    if TOKEN_FILE.exists():
        credentials = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    elif not credentials or not credentials.valid:
        if not CREDENTIALS_FILE.exists():
            raise FileNotFoundError(f"Gmail OAuth credentials file not found: {CREDENTIALS_FILE}")
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
        credentials = flow.run_local_server(port=0)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=credentials)


def send_emergency_email(species, common_name, venom_status, location,
                         medical_information, recipients, user_profile):
    """Send the requested concise emergency email separately to each contact."""
    recipient_list = list(dict.fromkeys(
        str(address).strip() for address in (recipients or []) if str(address).strip()
    ))
    if not recipient_list:
        raise ValueError("No emergency contact email addresses were supplied.")

    profile = user_profile or {}
    full_name = str(profile.get("full_name") or "Not provided").strip()
    phone = str(profile.get("phone") or "Not provided").strip()
    user_email = str(profile.get("email") or "Not provided").strip()
    location = str(location or "Not provided").strip()
    medical_information = str(medical_information or "Not provided").strip()

    body = (
        f"Name: {full_name}\n"
        f"Phone number: {phone}\n"
        f"User email: {user_email}\n"
        f"Species name: {species or 'Not provided'}\n"
        f"Common name: {common_name or 'Not provided'}\n"
        f"Venomous status: {venom_status or 'Not provided'}\n"
        f"Location coordinates and landmark: {location}\n"
        f"Medical information: {medical_information}\n"
        f"Medical attention: Seek emergency medical care immediately. Do not wait for email replies or AI identification.\n"
        f"First aid: Keep the person calm and still. Immobilize the bitten limb and remove rings, watches, or tight clothing. Do not cut or suck the wound. Do not apply a tourniquet, ice, or herbal remedies. Do not attempt to catch or handle the snake.\n\n"
        f"{full_name} is in an emergency situation. Kindly help them."
    )

    service = get_gmail_service()
    sent = []
    failed = []
    for recipient in recipient_list:
        try:
            message = EmailMessage()
            message["To"] = recipient
            message["Subject"] = "VENOMGUARD - EMERGENCY ALERT"
            message.set_content(body)
            encoded = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
            service.users().messages().send(userId="me", body={"raw": encoded}).execute()
            sent.append(recipient)
            print(f"[GMAIL] Accepted emergency email for {recipient}")
        except Exception as error:
            failed.append(recipient)
            print(f"[GMAIL ERROR] {recipient}: {error}")
    return {"sent": sent, "failed": failed}


if __name__ == "__main__":
    print("VENOMGUARD GMAIL API TEST")
    recipient = input("Recipient email address: ").strip()
    choice = input("Type SEND to send a test email: ").strip()
    if choice == "SEND" and recipient:
        result = send_emergency_email(
            species="Naja naja",
            common_name="Spectacled Cobra",
            venom_status="Venomous",
            location="Test location",
            medical_information="Test only",
            recipients=[recipient],
            user_profile={
                "full_name": "VenomGuard Test User",
                "email": "test@example.com",
                "phone": "+919876543210",
            },
        )
        print("SUCCESS: GMAIL ALERT SENT" if result["sent"] and not result["failed"]
              else "FAILED: GMAIL ALERT NOT SENT")
    else:
        print("Test cancelled.")
