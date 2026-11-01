VENOMGUARD GPS + EMERGENCY EMAIL UPDATE

Files included:
- app_local.py
- emergency_alerts.py
- gmail_alert.py
- templates/index.html
- static/app.js
- static/emergency_contacts.js
- static/emergency_contacts.css
- static/style.css
- static/assets (UI assets)

Installation:
1. Back up your current project.
2. Replace the matching files in your VenomGuard project with these files.
3. Keep your trained models and datasets in their existing locations.
4. Keep Gmail OAuth credentials/token in your local gmail_notification folder.
   They are intentionally NOT included in this ZIP.
5. Run: python app_local.py
6. Open the local URL shown in the terminal.
7. Allow browser location permission or enter a location manually.
8. Test Gmail delivery with your own contact address first.

Location:
- Uses browser GPS permission for latitude/longitude and reported accuracy.
- Attempts reverse geocoding through OpenStreetMap Nominatim to obtain a nearby
  address/landmark.
- If GPS or landmark lookup is unavailable, coordinates or landmark may be
  missing; enter the location manually.
- Reverse-geocoded landmark is an approximate map result, not a guarantee of
  the user's exact entrance or physical position.

Emergency email:
- Includes user profile, snake scientific/common names, venomous status,
  location coordinates/landmark, medical information, medical attention
  guidance, and basic snakebite first aid.
- First aid: keep the person calm and still; immobilize the bitten limb;
  remove rings/watches/tight clothing; do not cut or suck the wound; do not
  apply a tourniquet, ice, or herbal remedies; do not handle the snake.
- Seek emergency medical care immediately. Do not wait for email replies or
  AI identification.
- A venomous snake detection does not prove that a person was bitten.

Security:
- This ZIP excludes .env and token.json files. Do not share OAuth tokens,
  client secrets, .env files, or credentials.json.
- The uploaded archive contained sensitive local authentication files. Revoke
  the exposed Google OAuth token/access in your Google Account and re-authorize
  locally if that token was active.
