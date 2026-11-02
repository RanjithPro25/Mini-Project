(() => {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[char]);

  let contactsCache = [];
  let alertInProgress = false;
  let lastAlertSignature = '';

  function status(message, kind = '') {
    const element = $('#vgEmergencyStatus');
    if (!element) return;
    element.className = `vg-status ${kind}`;
    element.textContent = message;
  }
  function profileStatus(message, kind = '') {
    const element = $('#vgProfileStatus');
    if (!element) return;
    element.className = `vg-status ${kind}`;
    element.textContent = message;
  }
  async function loadProfile() {
    const response = await fetch('/api/emergency/profile');
    const data = await response.json();
    if (!response.ok || data.ok === false) throw new Error(data.error || 'Could not load profile.');
    if (data.profile) {
      $('#vgProfileName').value = data.profile.full_name || '';
      $('#vgProfileEmail').value = data.profile.email || '';
      $('#vgProfilePhone').value = data.profile.phone || '';
    }
  }
  async function saveProfile(event) {
    event.preventDefault();
    const full_name = $('#vgProfileName').value.trim();
    const email = $('#vgProfileEmail').value.trim();
    const phone = $('#vgProfilePhone').value.trim();
    if (full_name.length < 2) return profileStatus('Enter your full name.', 'vg-error');
    if (!$('#vgProfileEmail').checkValidity()) return profileStatus('Enter a valid email address.', 'vg-error');
    try {
      const response = await fetch('/api/emergency/profile', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ full_name, email, phone })
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || 'Profile could not be saved.');
      profileStatus('User profile saved.', 'vg-good');
    } catch (error) { profileStatus(error.message, 'vg-error'); }
  }
  async function loadContacts() {
    const response = await fetch('/api/emergency/contacts');
    const data = await response.json();
    if (!response.ok || data.ok === false) throw new Error(data.error || 'Could not load contacts.');
    contactsCache = Array.isArray(data.contacts) ? data.contacts : [];
    renderContacts();
  }
  function renderContacts() {
    const list = $('#vgContactsList');
    if (!list) return;
    if (!contactsCache.length) {
      list.innerHTML = '<p class="vg-note">No emergency contacts saved yet.</p>';
      return;
    }
    list.innerHTML = contactsCache.map(contact => `
      <div class="vg-contact"><div><b>${esc(contact.name)}</b><small>${esc(contact.email)}</small></div>
      <div class="vg-actions"><button type="button" class="vg-btn vg-muted" data-remove="${esc(contact.id)}">Remove</button></div></div>`).join('');
  }
  async function addContact(event) {
    event.preventDefault();
    const name = $('#vgContactName').value.trim();
    const email = $('#vgContactEmail').value.trim();
    if (name.length < 2) return status("Enter the contact's name.", 'vg-error');
    if (!$('#vgContactEmail').checkValidity()) return status('Enter a valid email address.', 'vg-error');
    try {
      const response = await fetch('/api/emergency/contacts', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email })
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || 'Contact could not be saved.');
      $('#vgContactName').value = '';
      $('#vgContactEmail').value = '';
      await loadContacts();
      status('Emergency contact saved.', 'vg-good');
    } catch (error) { status(error.message, 'vg-error'); }
  }
  function readVisibleSpecies() {
    const output = $('#outputContent');
    const head = output && $('.species-output-head', output);
    if (!head) return null;
    const scientificName = $('h2', head)?.textContent?.trim() || '';
    if (!scientificName || /waiting for|no result|unknown|no snake detected|species uncertain/i.test(scientificName)) return null;
    const facts = {};
    output.querySelectorAll('.fact').forEach(item => {
      const key = $('b', item)?.textContent?.trim();
      const value = $('span', item)?.textContent?.trim();
      if (key && value) facts[key.toLowerCase()] = value;
    });
    const confidenceBox = [...output.querySelectorAll('.evidence-box')]
      .find(item => $('b', item)?.textContent?.trim() === 'Confidence');
    const confidence = confidenceBox ? ($('span', confidenceBox)?.textContent?.trim() || '') : '';
    return { scientificName, venom: String(facts['venom status'] || '').trim().toLowerCase(), confidence };
  }

  async function maybeSendAutomaticAlert() {
    if (alertInProgress) return;
    const detected = readVisibleSpecies();
    if (!detected) {
      lastAlertSignature = '';
      return;
    }
    const isVenomous = ['venomous', 'yes', 'true', 'highly venomous', 'medically significant'].includes(detected.venom);
    if (!isVenomous) return;
    const signature = `${detected.scientificName.toLowerCase()}|${detected.confidence}`;
    if (signature === lastAlertSignature) return;
    lastAlertSignature = signature;
    alertInProgress = true;
    status('Venomous snake detected. Sending emergency email to your saved contacts...', '');
    try {
      const response = await fetch('/api/emergency/auto-send', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scientific_name: detected.scientificName,
          confidence: detected.confidence
        })
      });
      const data = await response.json();
      if (!response.ok || data.ok !== true) throw new Error(data.error || 'Emergency message could not be sent.');
      if (data.sent) status('Emergency message was sent.', 'vg-good');
      else status(data.reason || 'No emergency email was sent.', 'vg-error');
    } catch (error) {
      status(error.message || 'Emergency message could not be sent.', 'vg-error');
    } finally { alertInProgress = false; }
  }

  function mount() {
    if ($('#vgEmergencyPanel')) return;
    const outputSection = $('#outputSection');
    const anchor = document.querySelector('main.container') || document.querySelector('main') || document.body;
    const panel = document.createElement('section');
    panel.id = 'vgEmergencyPanel';
    panel.className = 'panel';
    panel.innerHTML = `
      <div class="section-title"><span class="icon">👤</span><div>
        <h2>User Profile</h2><p>Your profile details will be included in automatic emergency emails.</p>
      </div></div>
      <form id="vgProfileForm">
        <div class="vg-grid">
          <label>Full name<input id="vgProfileName" maxlength="100" autocomplete="name" required placeholder="Your full name"></label>
          <label>Email address<input id="vgProfileEmail" type="email" maxlength="254" autocomplete="email" required placeholder="you@example.com"></label>
          <label>Phone number<input id="vgProfilePhone" type="tel" maxlength="25" autocomplete="tel" required placeholder="+91 9876543210"></label>
        </div>
        <div class="vg-actions"><button type="submit">Save User Profile</button></div>
        <div id="vgProfileStatus" class="vg-status" role="status" aria-live="polite">Complete your profile to enable emergency alerts.</div>
      </form>
      <hr>
      <div class="section-title"><span class="icon">🆘</span><div>
        <h2>Emergency Contacts</h2><p>Save the name and email address of each trusted emergency contact.</p>
      </div></div>
      <form id="vgAddContactForm">
        <div class="vg-grid">
          <label>Contact name<input id="vgContactName" maxlength="80" autocomplete="name" required placeholder="Family member or friend"></label>
          <label>Email address<input id="vgContactEmail" type="email" maxlength="254" autocomplete="email" required placeholder="name@example.com"></label>
        </div>
        <div class="vg-actions"><button type="submit">Add Contact</button></div>
      </form>
      <div id="vgContactsList" aria-live="polite"></div>
      <hr>
      <div class="section-title"><span class="icon">📍</span><div>
        <h2>Emergency Alerts</h2><p>Location is resolved by the server. Species-specific medical guidance is read from the local knowledge dataset.</p>
      </div></div>
      <div id="vgEmergencyStatus" class="vg-status" role="status" aria-live="polite">Automatic alerts are sent when a verified venomous species is detected.</div>`;
    if (outputSection) outputSection.insertAdjacentElement('afterend', panel);
    else anchor.appendChild(panel);

    $('#vgProfileForm').addEventListener('submit', saveProfile);
    $('#vgAddContactForm').addEventListener('submit', addContact);
    $('#vgContactsList').addEventListener('click', async event => {
      const button = event.target.closest('[data-remove]');
      if (!button) return;
      try {
        const response = await fetch(`/api/emergency/contacts/${encodeURIComponent(button.dataset.remove)}`, { method: 'DELETE' });
        const data = await response.json();
        if (!response.ok || data.ok === false) throw new Error(data.error || 'Could not remove contact.');
        await loadContacts();
        status('Emergency contact removed.', 'vg-good');
      } catch (error) { status(error.message, 'vg-error'); }
    });
    loadProfile().catch(error => profileStatus(`Could not load profile: ${error.message}`, 'vg-error'));
    loadContacts().catch(error => status(`Could not load contacts: ${error.message}`, 'vg-error'));

    const output = $('#outputContent');
    if (output && window.MutationObserver) {
      const observer = new MutationObserver(() => { maybeSendAutomaticAlert(); });
      observer.observe(output, { childList: true, subtree: true, characterData: true });
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
