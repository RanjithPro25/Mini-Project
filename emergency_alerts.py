from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify, request

emergency_alerts_bp = Blueprint("emergency_alerts", __name__)
DB_PATH = Path(__file__).resolve().parent / "venomguard_emergency.db"


def _connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _email(value: str) -> str:
    address = str(value or "").strip().lower()
    pattern = r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+"
    if len(address) > 254 or not re.fullmatch(pattern, address, re.I):
        raise ValueError("Enter a valid email address.")
    return address


def _phone(value: str) -> str:
    phone = re.sub(r"[\s().-]", "", str(value or "").strip())
    if not re.fullmatch(r"\+[1-9]\d{7,14}", phone):
        raise ValueError("Use international phone format, for example +919876543210.")
    return phone


def _init_db():
    with _connect() as con:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='emergency_contacts'"
        ).fetchone()
        if exists:
            columns = {row[1] for row in con.execute("PRAGMA table_info(emergency_contacts)")}
            if "email" not in columns:
                con.execute("ALTER TABLE emergency_contacts RENAME TO emergency_contacts_old")
                con.execute("""CREATE TABLE emergency_contacts (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL)""")
                con.execute("DROP TABLE emergency_contacts_old")
        else:
            con.execute("""CREATE TABLE emergency_contacts (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1), full_name TEXT NOT NULL,
            email TEXT NOT NULL, phone TEXT NOT NULL, updated_at TEXT NOT NULL)""")


def get_user_profile():
    with _connect() as con:
        row = con.execute(
            "SELECT full_name, email, phone FROM user_profile WHERE id=1"
        ).fetchone()
    return dict(row) if row else None


def send_venomous_detection_alert(species: str, common_name: str, venom_status: str,
                                  confidence: str = "", location: str = "",
                                  medical_information: str = ""):
    """Send only for a species that the server has verified as explicitly venomous."""
    accepted = {"venomous", "yes", "true", "highly venomous", "medically significant"}
    if str(venom_status or "").strip().lower() not in accepted:
        return {"ok": True, "sent": False, "reason": "Species is not explicitly marked venomous."}

    profile = get_user_profile()
    if not profile:
        return {"ok": False, "sent": False,
                "error": "Complete your User Profile before automatic alerts can be sent."}

    with _connect() as con:
        contacts = [dict(row) for row in con.execute(
            "SELECT name, email FROM emergency_contacts ORDER BY created_at"
        ).fetchall()]
    if not contacts:
        return {"ok": False, "sent": False,
                "error": "Add at least one emergency contact first."}

    from gmail_alert import send_emergency_email
    try:
        result = send_emergency_email(
            species=species,
            common_name=common_name,
            venom_status=venom_status,
            location=location,
            medical_information=medical_information,
            recipients=[contact["email"] for contact in contacts],
            user_profile=profile,
        )
    except Exception as error:
        print("[GMAIL ERROR]", error)
        return {"ok": False, "sent": False, "error": "Gmail could not send the emergency email. Check Gmail authentication."}

    sent = result["sent"]
    failed = result["failed"]
    if sent and not failed:
        return {"ok": True, "sent": True, "sent_count": len(sent),
                "total_count": len(contacts), "message": "Emergency message was sent."}
    return {"ok": False, "sent": bool(sent), "sent_count": len(sent),
            "failed_count": len(failed), "total_count": len(contacts),
            "error": f"Email sent to {len(sent)} of {len(contacts)} contacts. Check Gmail authentication and failed addresses."}


_init_db()


@emergency_alerts_bp.get("/api/emergency/profile")
def read_profile():
    return jsonify({"ok": True, "profile": get_user_profile()})


@emergency_alerts_bp.post("/api/emergency/profile")
def save_profile():
    data = request.get_json(silent=True) or {}
    full_name = str(data.get("full_name") or "").strip()[:100]
    if len(full_name) < 2:
        return jsonify({"ok": False, "error": "Enter your full name."}), 400
    try:
        email = _email(data.get("email"))
        phone = _phone(data.get("phone"))
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    with _connect() as con:
        con.execute("""INSERT INTO user_profile (id, full_name, email, phone, updated_at)
            VALUES (1, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET
            full_name=excluded.full_name, email=excluded.email,
            phone=excluded.phone, updated_at=excluded.updated_at""",
            (full_name, email, phone, _now()))
    return jsonify({"ok": True, "profile": {
        "full_name": full_name, "email": email, "phone": phone
    }})


@emergency_alerts_bp.get("/api/emergency/contacts")
def get_contacts():
    with _connect() as con:
        rows = con.execute(
            "SELECT id, name, email, created_at FROM emergency_contacts ORDER BY created_at"
        ).fetchall()
    return jsonify({"ok": True, "contacts": [dict(row) for row in rows]})


@emergency_alerts_bp.post("/api/emergency/contacts")
def add_contact():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name") or "").strip()[:80]
    if len(name) < 2:
        return jsonify({"ok": False, "error": "Enter the contact's name (at least 2 characters)."}), 400
    try:
        email = _email(data.get("email"))
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    row = {"id": str(uuid.uuid4()), "name": name, "email": email, "created_at": _now()}
    try:
        with _connect() as con:
            con.execute(
                "INSERT INTO emergency_contacts (id, name, email, created_at) VALUES (?, ?, ?, ?)",
                (row["id"], row["name"], row["email"], row["created_at"])
            )
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "That email address is already saved."}), 409
    return jsonify({"ok": True, "contact": row})


@emergency_alerts_bp.delete("/api/emergency/contacts/<contact_id>")
def delete_contact(contact_id):
    with _connect() as con:
        cur = con.execute("DELETE FROM emergency_contacts WHERE id = ?", (contact_id,))
    if cur.rowcount == 0:
        return jsonify({"ok": False, "error": "Contact not found."}), 404
    return jsonify({"ok": True})
