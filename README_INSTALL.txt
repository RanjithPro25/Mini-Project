VENOMGUARD AUTOMATIC EMERGENCY ALERT ADD-ON

UI behavior
- The UI asks only for contact name and phone number (with country code).
- Contacts are stored in this browser's localStorage.
- On an explicit displayed "Venomous" result with displayed confidence >= 70%, the add-on attempts to send both SMS and WhatsApp messages to all saved contacts.
- It does not send for "Non-venomous", unknown, missing confidence, or confidence below 70%.
- The message calls the result a preliminary AI prediction and explicitly says it does not mean a bite occurred.
- Only add contacts who have agreed to receive these alerts. Automated messaging may incur charges.

IMPORTANT: Automatic SMS/WhatsApp delivery requires a configured provider account. This package uses Twilio. WhatsApp sender access and approved templates may be required by your WhatsApp Business provider, and provider acceptance does not guarantee delivery or that a person reads the message.

FILES
- templates/index.html: reference copy with add-on script/style links. If your current HTML is newer, preserve it and add only these references:
  <link rel="stylesheet" href="/static/emergency_contacts.css">
  <script src="/static/emergency_contacts.js" defer></script> immediately before </body>.
- static/emergency_contacts.js
- static/emergency_contacts.css
- emergency_alerts.py

INSTALL WITHOUT MODIFYING MODEL CODE
1. Back up the project.
2. Copy static/emergency_contacts.js and static/emergency_contacts.css into your existing static/ folder.
3. Preserve your current templates/index.html and add the CSS link in <head> and script tag before </body> if they are not already present.
4. Copy emergency_alerts.py into the same project directory as app_local.py.
5. Add these two lines to app_local.py after `app = Flask(__name__)` (this registers a separate route; do not change model/inference code):

   from emergency_alerts import emergency_alerts_bp
   app.register_blueprint(emergency_alerts_bp)

6. Install the SDK in the same Python environment: `pip install twilio`
7. Set these environment variables before starting Flask (do not put credentials in JavaScript or commit them to Git):
   TWILIO_ACCOUNT_SID
   TWILIO_AUTH_TOKEN
   TWILIO_SMS_FROM          (your Twilio SMS-capable sender, usually +countrycode...)
   TWILIO_WHATSAPP_FROM     (your approved Twilio WhatsApp sender, usually whatsapp:+countrycode...)
8. Restart Flask and refresh with Ctrl+F5.

Windows PowerShell example (use your actual provider values; do not share them):
$env:TWILIO_ACCOUNT_SID="your_account_sid"
$env:TWILIO_AUTH_TOKEN="your_auth_token"
$env:TWILIO_SMS_FROM="+1xxxxxxxxxx"
$env:TWILIO_WHATSAPP_FROM="whatsapp:+14155238886"
python app_local.py

TESTING / LIMITATIONS
- First test with consenting contacts and provider test numbers.
- If the UI's displayed result markup differs from `.species-output-head`, `.fact`, or `.evidence-box`, the automatic trigger will not fire; inspect the browser console and adapt the result reader to your actual output HTML.
- Current contacts are browser-local and sent to the Flask endpoint only when a qualifying result appears. This endpoint is intended for a local trusted app, not public internet exposure. Do not expose it publicly without authentication, rate limiting, server-side user/contact storage, CSRF/origin protections, and abuse controls.
- Alert sends are deduplicated per browser by result and contact list. Clearing browser local storage resets this dedupe record.
- AI confidence is not a medical probability. Never use this as the sole basis for medical decisions. A suspected snakebite needs urgent medical care.
- SMS/WhatsApp provider API acceptance is not proof of delivery/read. WhatsApp may require an approved template outside its service window.
