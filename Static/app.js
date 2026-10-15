const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

async function apiJson(url, options={}) {
  const r = await fetch(url, options);
  const text = await r.text();
  const type = r.headers.get("content-type") || "";
  if (!type.includes("application/json")) {
    throw new Error(`VenomGuard server returned an unexpected response (${r.status}). Restart app_local.py.`);
  }
  const data = JSON.parse(text);
  if (!r.ok || data.ok === false) {
    throw new Error(data.error || "VenomGuard request failed.");
  }
  return data;
}

let selectedFile = null;
let activeCandidates = [];
let candidateSource = "";
let lastResult = null;
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let voices = [];
let onlineMode = localStorage.getItem("venomguard-online") === "true";
let detectionSession = "imageMode";

function esc(x) {
  return String(x ?? "").replace(/[&<>'"]/g, m => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[m]));
}

function setStatus(text, type="waiting") {
  const el = $("#statusPill");
  el.textContent = text;
  el.className = `status ${type}`;
}

// Front-end analysis progress: smoothly shows 0% -> 100% while the AI request is running.
let analysisTimer = null;
let analysisValue = 0;
function startAnalysisProgress(label="Analyzing...") {
  clearInterval(analysisTimer);
  analysisValue = 0;
  $("#analysisProgress").hidden = false;
  $("#analysisProgressLabel").textContent = `${label} 0%`;
  $("#analysisProgressPercent").textContent = "0%";
  $("#analysisProgressBar").style.width = "0%";
  $("#analysisProgressDetail").textContent = "Preparing AI analysis...";
  analysisTimer = setInterval(() => {
    if (analysisValue < 92) analysisValue += analysisValue < 55 ? 2 : 1;
    $("#analysisProgressLabel").textContent = `${label} ${analysisValue}%`;
    $("#analysisProgressPercent").textContent = `${analysisValue}%`;
    $("#analysisProgressBar").style.width = `${analysisValue}%`;
    setStatus(`${label} ${analysisValue}%`, "busy");
    $("#analysisProgressDetail").textContent = analysisValue < 30 ? "Preparing input and AI models..." : analysisValue < 65 ? "Detecting and analyzing snake features..." : "Ranking species and preparing the result...";
  }, 180);
}
function finishAnalysisProgress() {
  clearInterval(analysisTimer);
  analysisTimer = null;
  analysisValue = 100;
  $("#analysisProgressPercent").textContent = "100%";
  $("#analysisProgressBar").style.width = "100%";
  $("#analysisProgressLabel").textContent = "Analyzed";
  $("#analysisProgressDetail").textContent = "AI analysis completed. Showing the result below.";
}
function failAnalysisProgress(message) {
  clearInterval(analysisTimer);
  analysisTimer = null;
  $("#analysisProgressLabel").textContent = "Analysis failed";
  $("#analysisProgressDetail").textContent = message || "The analysis could not be completed.";
}

function clearDetectionOutput() {
  clearInterval(analysisTimer);
  analysisTimer = null;
  analysisValue = 0;
  activeCandidates = [];
  candidateSource = "";
  lastResult = null;
  $("#outputContent").className = "output-content empty-output";
  $("#outputContent").innerHTML = '<div class="empty-icon">🐍</div><h3>No result yet</h3><p>Run this detection session to see its result.</p>';
  $("#candidateGrid").innerHTML = "";
  $("#candidateStatus").textContent = "No candidates yet.";
  $("#onlineSection").hidden = true;
  $("#onlineSources").innerHTML = "";
  $("#onlineStatus").textContent = "Online search has not been requested.";
  $("#outputEyebrow").textContent = "AI OUTPUT";
  $("#outputTitle").textContent = "Waiting for an analysis";
  $("#outputSubtitle").textContent = "Run the current detection session to see its result.";
  $("#statusPill").textContent = "Waiting";
  $("#statusPill").className = "status waiting";
  $("#analysisProgress").hidden = true;
}

function switchMode(id) {
  detectionSession = id;
  clearDetectionOutput();
  $$(".mode-tab").forEach(x => x.classList.toggle("active", x.dataset.mode === id));
  $$(".mode-section").forEach(x => x.classList.toggle("active", x.id === id));
}

$$(".mode-tab").forEach(b => b.onclick = () => switchMode(b.dataset.mode));

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
const savedTheme = localStorage.getItem("venomguard-theme") || "light";
if (savedTheme === "dark") {
  document.body.classList.add("dark");
  $("#themeToggle").checked = true;
}
$("#themeToggle").onchange = e => {
  document.body.classList.toggle("dark", e.target.checked);
  localStorage.setItem("venomguard-theme", e.target.checked ? "dark" : "light");
};

function setExecutionMode(isOnline) {
  onlineMode = !!isOnline;
  $("#onlineModeToggle").checked = onlineMode;
  $("#offlineModeToggle").checked = !onlineMode;
  localStorage.setItem("venomguard-online", onlineMode ? "true" : "false");
  document.body.classList.toggle("online-active", onlineMode);
  loadVoices();
  const badge = document.querySelector(".mode-switch-label");
  if (badge) badge.textContent = onlineMode ? "Online mode" : "Offline mode";
}
$("#offlineModeToggle").onchange = () => setExecutionMode(false);
$("#onlineModeToggle").onchange = () => setExecutionMode(true);
setExecutionMode(onlineMode);

async function runOnlineFallback(query, reason="Local dataset did not provide a confident match.") {
  if (!onlineMode) return null;
  $("#onlineSection").hidden = false;
  $("#onlineStatus").textContent = "Searching Wikipedia, iNaturalist, GBIF and selected snake-information websites…";
  try {
    const j = await apiJson("/api/online-search", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({query})
    });
    const p = j.profile || {};
    const results = j.results || [];
    $("#onlineStatus").innerHTML = `${esc(reason)} <b>${j.found ? "Online information found." : "No reliable online result was found."}</b>`;
    $("#onlineSources").innerHTML = results.map(x => `
      <article class="online-source-card">
        ${x.image ? `<img src="${esc(x.image)}" alt="${esc(x.title || "Snake")}" onerror="this.style.display='none'">` : ""}
        <div><b>${esc(x.title || x.scientific_name || "Online result")}</b><span>${esc(x.source || "Web source")}</span><p>${esc(x.description || "No summary was supplied by this source.")}</p><a href="${esc(x.url || "#")}" target="_blank" rel="noopener noreferrer">Open source ↗</a></div>
      </article>`).join("");
    if (j.found) {
      renderFullOutput(p, {
        mode:"ONLINE WEB KNOWLEDGE",
        pill:"🌐 Online information",
        source:"External snake-information sources",
        mainImage:p.sample_image_url || "/static/assets/sample_snake.svg",
        confidenceLevel:"Web research — not local model confidence",
        classifier:p.scientific_name,
        datasetText:"Retrieved at request time from external snake-information sources. Facts are shown only when supplied by the retrieved sources."
      });
      populateTalk(p, buildSpokenSummary(p));
    }
    return j;
  } catch(e) {
    $("#onlineStatus").textContent = e.message;
    return null;
  }
}

// ---------------------------------------------------------------------------
// Image selection and camera
// ---------------------------------------------------------------------------
const fileInput = $("#fileInput");
const dropzone = $("#dropzone");

$("#chooseBtn").onclick = e => { e.preventDefault(); fileInput.click(); };
fileInput.onchange = () => { if (fileInput.files[0]) selectFile(fileInput.files[0]); };
dropzone.ondragover = e => { e.preventDefault(); dropzone.classList.add("drag"); };
dropzone.ondragleave = () => dropzone.classList.remove("drag");
dropzone.ondrop = e => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
};

function selectFile(file) {
  if (!file.type.startsWith("image/")) return alert("Please select an image file.");
  if (file.size > 10 * 1024 * 1024) return alert("Maximum image size is 10 MB.");
  selectedFile = file;
  $("#fileName").textContent = file.name;
  $("#analyzeBtn").disabled = false;
  const url = URL.createObjectURL(file);
  $("#preview").src = url;
  $("#imageProgress").textContent = "Image ready for local analysis";
}

$("#startCamera").onclick = async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:"environment"},width:{ideal:1280},height:{ideal:720}},audio:false});
    $("#cameraVideo").srcObject = mediaStream;
    $("#startCamera").disabled = true;
    $("#capturePhoto").disabled = false;
    $("#stopCamera").disabled = false;
    $("#cameraStatus").textContent = "Camera active. The captured photo is processed locally.";
  } catch (e) {
    $("#cameraStatus").textContent = "Camera permission was not available. You can still upload an image.";
  }
};

$("#capturePhoto").onclick = () => {
  const video = $("#cameraVideo");
  const canvas = $("#cameraCanvas");
  if (!video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => {
    const file = new File([blob], `camera_${Date.now()}.jpg`, {type:"image/jpeg"});
    selectFile(file);
    $("#cameraStatus").textContent = "Captured image is ready. Click Analyze Snake.";
  }, "image/jpeg", 0.94);
};

$("#stopCamera").onclick = () => stopCamera();
function stopCamera() {
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  mediaStream = null;
  $("#cameraVideo").srcObject = null;
  $("#startCamera").disabled = false;
  $("#capturePhoto").disabled = true;
  $("#stopCamera").disabled = true;
}

// ---------------------------------------------------------------------------
// Image analysis
// ---------------------------------------------------------------------------
$("#analyzeBtn").onclick = async () => {
  if (!selectedFile) return;
  const fd = new FormData();
  fd.append("image", selectedFile);
  $("#analyzeBtn").disabled = true;
  $("#imageProgress").textContent = "Analyzing...";
  setStatus("Analyzing 0%", "busy");
  startAnalysisProgress("Analyzing snake image...");
  try {
    const j = await apiJson("/api/analyze", {method:"POST", body:fd});
    if (detectionSession !== "imageMode") return;
    lastResult = j.result;
    await renderImageResult(j.result);
    finishAnalysisProgress();
    setStatus("✓ Analyzed", "ok");
    $("#imageProgress").textContent = "Analyzed";
  } catch (e) {
    failAnalysisProgress(e.message);
    setStatus("Analysis Failed", "error");
    $("#imageProgress").textContent = e.message;
    alert(e.message);
  } finally {
    $("#analyzeBtn").disabled = false;
  }
};

async function renderImageResult(r) {
  const details = r.species_details || {};
  renderFullOutput(details, {
    mode:"AI DETECTED SPECIES",
    pill:"✓ Detected by AI",
    source:"Image detection • YOLOv8 + EfficientNet-B3 + Phase 14",
    mainImage:r.preview_data,
    yolo:r.yolo_status,
    yoloConfidence:r.yolo_confidence,
    classifier:r.scientific_name,
    classifierConfidence:r.confidence_percent,
    confidenceLevel:r.confidence_category,
    datasetText:"Matched against the trained 313-class EfficientNet model and local Phase 14 knowledge base."
  });
  if (r.candidate_cards?.length) renderCandidateCards(r.candidate_cards, "Image model Top-5 ranking");
  if (onlineMode && Number(r.confidence_percent || 0) < 50) {
    const q = `unknown snake species; local candidates: ${(r.candidate_cards || []).slice(0,5).map(x => x.scientific_name).join(", ")}`;
    await runOnlineFallback(q, "The local image classifier is below 50% confidence, so Online Mode is checking external snake-information sources.");
  }
  populateTalk(details, `The detected snake is ${details.common_name || details.scientific_name || r.scientific_name}. Confidence ${Number(r.confidence_percent || 0).toFixed(1)} percent. Venom status: ${details.venomous || r.venom_status || "unresolved"}.`);
}

// ---------------------------------------------------------------------------
// Natural-language text ranking
// ---------------------------------------------------------------------------
$("#describeBtn").onclick = async () => {
  const text = $("#descriptionInput").value.trim();
  if (!text) return alert("Describe the snake first.");
  await rankText(text, "Text Detection");
};

async function rankText(text, source) {
  const session = source === "Voice Detection" ? "voiceMode" : "textMode";
  detectionSession = session;
  $("#textStatus").textContent = "Analyzing...";
  setStatus("Analyzing 0%", "busy");
  startAnalysisProgress(`${source} analyzing...`);
  try {
    const j = await apiJson("/api/candidates", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text})
    });
    if (detectionSession !== session) return;
    activeCandidates = j.candidates || [];
    renderCandidateCards(activeCandidates, source);
    $("#candidateStatus").innerHTML = `${j.confident ? "<b>Strong candidate separation.</b>" : "<b>Uncertain ranking.</b>"} The five candidates are ranked from natural-language evidence. Compare the real sample images and select the snake you actually saw.`;
    finishAnalysisProgress();
    $("#textStatus").textContent = "Analyzed — Top 5 ready";
    setStatus("✓ Analyzed", "ok");
    $("#outputEyebrow").textContent = "TEXT MATCHING";
    $("#outputTitle").textContent = "Select the snake you actually saw";
    $("#outputSubtitle").textContent = "The selected candidate will open the complete species record below.";
    if (onlineMode && (!j.confident || !activeCandidates.length || Number(activeCandidates[0]?.ranking_confidence || 0) < 0.50)) {
      await runOnlineFallback(text, "The local ranking is below the high-confidence range, so Online Mode is checking external sources.");
    }
  } catch (e) {
    failAnalysisProgress(e.message);
    $("#textStatus").textContent = e.message;
    setStatus("Matching Failed", "error");
  }
}

// ---------------------------------------------------------------------------
// Candidate cards
// ---------------------------------------------------------------------------
function renderCandidateCards(candidates, source) {
  activeCandidates = candidates || [];
  candidateSource = source || "AI ranking";
  const grid = $("#candidateGrid");
  if (!activeCandidates.length) {
    grid.innerHTML = "";
    $("#candidateStatus").textContent = "No candidates available.";
    return;
  }
  grid.innerHTML = activeCandidates.map((c,i) => {
    const pct = Number(c.ranking_confidence ?? c.confidence ?? 0) * 100;
    const img = c.sample_image_url || "/static/assets/sample_snake.svg";
    const attrs = (c.matched_attributes || []).slice(0,4).map(x => `<span>${esc(x)}</span>`).join("");
    return `<article class="candidate-card" data-index="${i}">
      <div class="rank">#${i+1}</div>
      <img src="${esc(img)}" alt="${esc(c.scientific_name || "Snake")}" onerror="this.src='/static/assets/sample_snake.svg'">
      <div class="candidate-info">
        <h3>${esc(c.scientific_name || "Unknown")}</h3>
        <p>${esc(c.common_name || "Common name not recorded")}</p>
        <div class="candidate-score"><i style="width:${Math.min(100,pct)}%"></i></div>
        <b>${pct.toFixed(1)}% ranking</b>
        <div class="match-tags">${attrs || "<span>Candidate match</span>"}</div>
        <button class="primary select-candidate" data-index="${i}">Select this species</button>
      </div>
    </article>`;
  }).join("");
  $$(".select-candidate").forEach(b => b.onclick = () => selectCandidate(Number(b.dataset.index)));
}

async function selectCandidate(index) {
  const c = activeCandidates[index];
  if (!c) return;
  $$(".candidate-card").forEach(x => x.classList.remove("selected"));
  document.querySelector(`.candidate-card[data-index="${index}"]`)?.classList.add("selected");
  try {
    const j = await apiJson(`/api/species/${encodeURIComponent(c.scientific_name)}`);
    renderFullOutput(j.record, {
      mode:"USER SELECTED SPECIES",
      pill:"✓ Selected by user",
      source:candidateSource,
      mainImage:j.record.sample_image_url || "/static/assets/sample_snake.svg",
      confidenceLevel:"Ranking only",
      classifier:c.scientific_name,
      classifierConfidence:(Number(c.ranking_confidence || 0) * 100).toFixed(1),
      datasetText:"Selected from the ranked candidates. The Phase 14 record supplies the factual species information."
    });
    populateTalk(j.record, buildSpokenSummary(j.record));
    $("#outputSection").scrollIntoView({behavior:"smooth", block:"start"});
  } catch (e) {
    alert(e.message);
  }
}

// ---------------------------------------------------------------------------
// Full-width output renderer — shared by image, text, voice and database.
// ---------------------------------------------------------------------------
function renderFullOutput(r, meta={}) {
  const fields = [
    ["Accepted name", r.accepted_name || r.scientific_name],
    ["Family", r.family], ["Genus", r.genus], ["Distribution", r.distribution],
    ["Venom status", r.venomous], ["Venom category", r.venom_category],
    ["Venom scale", r.venomous_scale], ["WHO medical importance", r.who_medically_important],
    ["WHO risk category", r.who_risk_category], ["Emergency level", r.emergency_level],
    ["Medical significance", r.medical_significance], ["Identification features", r.identification_features],
    ["Habitat", r.habitat], ["Behavior", r.behavior], ["Diet", r.diet],
    ["Precautions", r.precautions], ["What not to do", r.what_not_to_do],
    ["Snakebite first aid", r.snakebite_first_aid], ["Medical attention", r.medical_attention],
    ["Verification status", r.verification_status], ["Conflict status", r.conflict_status],
    ["Conflicting sources", r.conflicting_sources], ["Evidence", r.evidence],
    ["Source", r.source], ["Last checked", r.last_checked]
  ].filter(([,v]) => v !== null && v !== undefined && String(v).trim() !== "");
  const wide = new Set(["Medical significance","Identification features","Habitat","Behavior","Diet","Precautions","What not to do","Snakebite first aid","Medical attention","Conflicting sources","Evidence"]);
  const image = meta.mainImage || r.sample_image_url || "/static/assets/sample_snake.svg";
  const evidence = `
    <div class="evidence-box"><b>YOLOv8</b><span>${esc(meta.yolo || "Not applicable")}${meta.yoloConfidence !== undefined && meta.yoloConfidence !== "" ? ` • ${Number(meta.yoloConfidence).toFixed(3)} confidence` : ""}</span></div>
    <div class="evidence-box"><b>EfficientNet-B3</b><span>${esc(meta.classifier || r.scientific_name || "Not available")}${meta.classifierConfidence !== undefined && meta.classifierConfidence !== "" ? ` • ${esc(meta.classifierConfidence)}%` : ""}</span></div>
    <div class="evidence-box"><b>Confidence</b><span>${esc(meta.confidenceLevel || "Not available")}</span></div>
    <div class="evidence-box"><b>Source</b><span>${esc(meta.datasetText || "Phase 14 validated knowledge record")}</span></div>`;
  const source = r.source_url || r.source;
  $("#outputContent").className = "output-content";
  $("#outputContent").innerHTML = `
    <div class="species-output">
      <div class="species-output-head">
        <div><span class="eyebrow">${esc(meta.mode || "SPECIES RESULT")}</span><h2>${esc(r.scientific_name || "Unknown")}</h2><p>${esc(r.common_name || "Common name not recorded")}</p></div>
        <span class="selected-badge">${esc(meta.pill || "✓ Result")}</span>
      </div>
      <div class="species-output-main">
        <div class="species-visual"><img src="${esc(image)}" alt="${esc(r.scientific_name || "Snake")}" onerror="this.src='/static/assets/sample_snake.svg'"></div>
        <div class="facts-grid">${fields.map(([k,v]) => `<div class="fact ${wide.has(k) ? "wide" : ""}"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join("")}</div>
      </div>
      <div class="evidence-grid">${evidence}</div>
      <div class="source-line">${source ? `Recorded source: ${String(source).startsWith("http") ? `<a href="${esc(source)}" target="_blank" rel="noopener">${esc(source)}</a>` : esc(source)}` : "Recorded source is not available."}</div>
    </div>`;
  $("#outputEyebrow").textContent = meta.mode || "SPECIES RESULT";
  $("#outputTitle").textContent = r.scientific_name || "Species Result";
  $("#outputSubtitle").textContent = r.common_name || "Complete Phase 14 species information";
}

function buildSpokenSummary(r) {
  return `Species: ${r.common_name || r.scientific_name || "unknown"}. Scientific name: ${r.scientific_name || "not available"}. Family: ${r.family || "not recorded"}. Venom status: ${r.venomous || "unresolved"}. Venom category: ${r.venom_category || "not recorded"}. Emergency level: ${r.emergency_level || "not recorded"}. Precautions: ${r.precautions || "keep a safe distance and do not handle the snake."}`;
}

function populateTalk(r, summary) {
  $("#talkText").value = summary || buildSpokenSummary(r);
}

// ---------------------------------------------------------------------------
// Voice-to-text
// ---------------------------------------------------------------------------
let speechRecognition = null;
let voiceBusy = false;
let voiceVoskStatus = {en:false, ta:false};

function browserVoiceAvailable() {
  return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}

async function loadVoiceStatus() {
  try {
    const j = await apiJson("/api/voice-status");
    voiceVoskStatus = j.vosk || {en:false, ta:false};
  } catch (_) {}
  updateVoiceStatusText();
}

function updateVoiceStatusText() {
  const lang = $("#voiceLanguage")?.value || "en-IN";
  const key = lang.toLowerCase().startsWith("ta") ? "ta" : "en";
  const local = !!voiceVoskStatus[key];
  const browser = browserVoiceAvailable();
  if (local) {
    $("#voiceOfflineStatus").textContent = "Local Vosk voice is ready for this language.";
  } else if (browser) {
    $("#voiceOfflineStatus").textContent = onlineMode
      ? "Browser voice recognition is ready."
      : "Browser voice recognition is available; local Vosk model is not installed for this language.";
  } else {
    $("#voiceOfflineStatus").textContent = "Voice recognition is unavailable. Install a local Vosk model or use a browser with SpeechRecognition support.";
  }
}

function stopBrowserRecognition() {
  try { speechRecognition?.stop(); } catch (_) {}
  speechRecognition = null;
}

function startBrowserRecognition(lang) {
  const R = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!R) throw new Error("Browser speech recognition is not available.");

  stopBrowserRecognition();
  speechRecognition = new R();
  speechRecognition.lang = lang;
  speechRecognition.interimResults = true;
  speechRecognition.continuous = false;
  speechRecognition.maxAlternatives = 1;
  voiceBusy = true;

  $("#recordIndicator").textContent = "Listening…";
  $("#recordVoice").textContent = "■ Stop Recording";
  $("#voiceWave").classList.add("recording");
  $("#voiceStatus").textContent = onlineMode
    ? "Listening for your snake description…"
    : "Listening… browser speech recognition may use the browser's speech service.";

  speechRecognition.onresult = e => {
    let finalText = "";
    let interimText = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0]?.transcript || "";
      if (e.results[i].isFinal) finalText += t + " ";
      else interimText += t;
    }
    const text = (finalText || interimText).trim();
    if (text) $("#voiceTranscript").value = text;
    $("#voiceStatus").textContent = finalText.trim() ? "Voice converted to text" : "Listening…";
  };

  speechRecognition.onerror = e => {
    voiceBusy = false;
    $("#voiceStatus").textContent = `Voice error: ${e.error || "recognition failed"}`;
    $("#recordIndicator").textContent = "Ready";
    $("#recordVoice").textContent = "● Start Recording";
    $("#voiceWave").classList.remove("recording");
  };

  speechRecognition.onend = () => {
    voiceBusy = false;
    $("#recordIndicator").textContent = "Ready";
    $("#recordVoice").textContent = "● Start Recording";
    $("#voiceWave").classList.remove("recording");
    if ($("#voiceTranscript").value.trim()) {
      $("#voiceStatus").textContent = "Voice converted to text — review or match the description.";
    }
    speechRecognition = null;
  };

  speechRecognition.start();
}

// Local Vosk recording is kept as a fallback for browsers without
// SpeechRecognition. The previous implementation recorded first and then
// started browser recognition, forcing users to speak twice. That behavior is
// intentionally removed.
async function recordAudioForVosk() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone access is unavailable.");
  const stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, echoCancellation:true, noiseSuppression:true, autoGainControl:true}});
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = e => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  source.connect(processor);
  processor.connect(ctx.destination);
  $("#recordIndicator").textContent = "Recording…";
  $("#recordVoice").textContent = "■ Stop Recording";
  $("#voiceWave").classList.add("recording");

  return new Promise(resolve => {
    window.__stopVoskRecord = () => {
      processor.disconnect(); source.disconnect(); stream.getTracks().forEach(t=>t.stop());
      const sampleRate = ctx.sampleRate;
      const length = chunks.reduce((n,a)=>n+a.length,0);
      const pcm = new Float32Array(length);
      let offset=0;
      for (const c of chunks) { pcm.set(c, offset); offset += c.length; }
      const wav = encodeWav(pcm, sampleRate);
      ctx.close();
      $("#recordIndicator").textContent = "Processing voice…";
      $("#recordVoice").textContent = "● Start Recording";
      $("#voiceWave").classList.remove("recording");
      window.__stopVoskRecord = null;
      resolve(new Blob([wav], {type:"audio/wav"}));
    };
  });
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (o,s) => [...s].forEach((c,i)=>view.setUint8(o+i,c.charCodeAt(0)));
  write(0,"RIFF"); view.setUint32(4,36+samples.length*2,true); write(8,"WAVE"); write(12,"fmt ");
  view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true); view.setUint32(24,sampleRate,true);
  view.setUint32(28,sampleRate*2,true); view.setUint16(32,2,true); view.setUint16(34,16,true); write(36,"data"); view.setUint32(40,samples.length*2,true);
  let p=44;
  for(const x of samples){const v=Math.max(-1,Math.min(1,x));view.setInt16(p,v<0?v*0x8000:v*0x7fff,true);p+=2;}
  return buffer;
}

$("#voiceLanguage").onchange = () => updateVoiceStatusText();

$("#recordVoice").onclick = async () => {
  const lang = $("#voiceLanguage").value;

  // If the browser supports SpeechRecognition, use it immediately. This is
  // the reliable path for Chrome/Edge and prevents the old speak-twice bug.
  if (browserVoiceAvailable()) {
    if (speechRecognition || voiceBusy) {
      stopBrowserRecognition();
      voiceBusy = false;
      $("#recordIndicator").textContent = "Ready";
      $("#recordVoice").textContent = "● Start Recording";
      $("#voiceWave").classList.remove("recording");
      $("#voiceStatus").textContent = "Recording stopped.";
      return;
    }
    try {
      startBrowserRecognition(lang);
      return;
    } catch (e) {
      $("#voiceStatus").textContent = e.message;
    }
  }

  // Fallback: local Vosk, only when browser recognition is unavailable.
  if (window.__stopVoskRecord) {
    window.__stopVoskRecord();
    return;
  }
  try {
    const promise = recordAudioForVosk();
    const blob = await promise;
    const fd = new FormData();
    fd.append("audio", blob, "voice.wav");
    fd.append("language", lang.toLowerCase().startsWith("ta") ? "ta" : "en");
    const j = await apiJson("/api/voice-transcribe", {method:"POST",body:fd});
    $("#voiceTranscript").value = j.text || "";
    $("#voiceStatus").textContent = j.text ? "Offline Vosk transcription complete" : "No speech recognized";
    $("#recordIndicator").textContent = "Ready";
  } catch (e) {
    $("#recordIndicator").textContent = "Ready";
    $("#recordVoice").textContent = "● Start Recording";
    $("#voiceWave").classList.remove("recording");
    $("#voiceStatus").textContent = e.message || "Voice recognition failed.";
  }
};

$("#voiceMatchBtn").onclick = async () => {
  const text = $("#voiceTranscript").value.trim();
  if (!text) return alert("Record or enter a voice description first.");
  await rankText(text, "Voice Detection");
};

loadVoiceStatus();

// ---------------------------------------------------------------------------
// Species Database — search only; selected details go to the same full-width
// output surface as image/text/voice.
// ---------------------------------------------------------------------------
async function searchSpecies() {
  const q = $("#speciesSearch").value.trim();
  $("#databaseResults").innerHTML = "Searching local Phase 14…";
  try {
    const j = await apiJson(`/api/species?q=${encodeURIComponent(q)}`);
    if (!j.records?.length) {
      $("#databaseResults").innerHTML = "<span>No matching species found in the local Phase 14 database.</span>";
      if (onlineMode && q) await runOnlineFallback(q, "This species/query is not present in the local Phase 14 database.");
      return;
    }
    $("#databaseResults").innerHTML = j.records.map((r,i)=>`<button class="db-row" data-name="${esc(r.scientific_name)}"><img src="${esc(r.sample_image_url || '/static/assets/sample_snake.svg')}" alt=""><div><b>${esc(r.scientific_name)}</b><span>${esc(r.common_name || "Common name not recorded")}</span></div><strong>›</strong></button>`).join("");
    $$(".db-row").forEach(b => b.onclick = () => selectDatabaseSpecies(b.dataset.name));
  } catch(e) { $("#databaseResults").textContent = e.message; }
}
$("#speciesSearchBtn").onclick = searchSpecies;
$("#speciesSearch").onkeydown = e => { if(e.key === "Enter") searchSpecies(); };

async function selectDatabaseSpecies(name) {
  try {
    const j = await apiJson(`/api/species/${encodeURIComponent(name)}`);
    renderFullOutput(j.record, {mode:"SEARCHED SPECIES",pill:"✓ Selected from Database",source:"Species Database search",mainImage:j.record.sample_image_url,datasetText:"Phase 14 validated knowledge record"});
    populateTalk(j.record, buildSpokenSummary(j.record));
    $("#outputSection").scrollIntoView({behavior:"smooth",block:"start"});
  } catch(e) { alert(e.message); }
}

// ---------------------------------------------------------------------------
// Batch
// ---------------------------------------------------------------------------
$("#batchInput").onchange = () => $("#batchStatus").textContent = `${$("#batchInput").files.length} image(s) selected`;
$("#batchBtn").onclick = async () => {
  const files = [...$("#batchInput").files];
  if (!files.length) return alert("Choose multiple images first.");
  $("#batchResults").innerHTML = ""; let done=0;
  for (const f of files) {
    const fd=new FormData();fd.append("image",f);
    try {
      const j=await apiJson("/api/analyze",{method:"POST",body:fd});
      const r=j.result;
      $("#batchResults").innerHTML += `<div class="batch-result"><b>${esc(f.name)}</b><span>${esc(r.scientific_name||"Unknown")}</span><span>${Number(r.confidence_percent||0).toFixed(1)}%</span><span>${esc(r.venom_status||"UNRESOLVED")}</span></div>`;
    } catch(e) { $("#batchResults").innerHTML += `<div class="batch-result error"><b>${esc(f.name)}</b><span>Failed: ${esc(e.message)}</span></div>`; }
    done++; $("#batchProgress").style.width=`${done/files.length*100}%`; $("#batchStatus").textContent=`Processed ${done}/${files.length}`;
  }
};

// ---------------------------------------------------------------------------
// AI Voice Talk — neural voices in Online Mode, installed voices in Offline Mode.
// ---------------------------------------------------------------------------
const neuralProfiles = [
  ["en-female-aashi","English • Female • Aashi","en"],
  ["en-female-aarti","English • Female • Aarti","en"],
  ["en-female-ananya","English • Female • Ananya","en"],
  ["en-female-kavya","English • Female • Kavya","en"],
  ["en-female-neerja","English • Female • Neerja","en"],
  ["en-male-aarav","English • Male • Aarav","en"],
  ["en-male-arjun","English • Male • Arjun","en"],
  ["en-male-kunal","English • Male • Kunal","en"],
  ["en-male-prabhat","English • Male • Prabhat","en"],
  ["en-male-rehaan","English • Male • Rehaan","en"],
  ["ta-female-pallavi","தமிழ் • Female • Pallavi (India)","ta"],
  ["ta-male-valluvar","தமிழ் • Male • Valluvar (India)","ta"],
  ["ta-female-kani","தமிழ் • Female • Kani (Malaysia)","ta"],
  ["ta-male-surya","தமிழ் • Male • Surya (Malaysia)","ta"],
  ["ta-female-venba","தமிழ் • Female • Venba (Singapore)","ta"],
  ["ta-male-anbu","தமிழ் • Male • Anbu (Singapore)","ta"],
  ["ta-female-saranya","தமிழ் • Female • Saranya (Sri Lanka)","ta"],
  ["ta-male-kumar","தமிழ் • Male • Kumar (Sri Lanka)","ta"]
];

function loadVoices() {
  const select = $("#voiceProfile");
  if (!select) return;
  const profiles = onlineMode ? neuralProfiles : [
    ["en-male-1","English • Male (installed)","en"],
    ["en-female-1","English • Female (installed)","en"],
    ["ta-male-1","தமிழ் • Male (installed)","ta"],
    ["ta-female-1","தமிழ் • Female (installed)","ta"]
  ];
  select.innerHTML = profiles.map(([id,label,lang])=>`<option value="${id}" data-lang="${lang}">${label}</option>`).join("");
}
if (window.speechSynthesis) { speechSynthesis.onvoiceschanged = loadVoices; }

function chooseInstalledVoice(profile, language) {
  const lang = language.toLowerCase().startsWith("ta") ? "ta" : "en";
  const candidates = voices.filter(v => v.lang.toLowerCase().startsWith(lang));
  if (!candidates.length) return null;
  const p = profile.toLowerCase();
  const femaleWords = ["female","woman","zira","susan","heera","kalpana","veena","pallavi","google us english female"];
  const maleWords = ["male","man","david","mark","ravi","prabhat","valluvar","guy"];
  const female = p.includes("female");
  const male = p.includes("male");
  const preferred = candidates.find(v => {
    const n=v.name.toLowerCase();
    if(female) return femaleWords.some(w=>n.includes(w));
    if(male) return maleWords.some(w=>n.includes(w));
    return false;
  });
  return preferred || candidates[0];
}

function syncVoiceProfiles() {
  loadVoices();
  const want = $("#talkLanguage")?.value?.startsWith("ta") ? "ta" : "en";
  const opts=[...($("#voiceProfile")?.options || [])];
  const first=opts.find(o=>o.dataset.lang===want);
  if(first) $("#voiceProfile").value=first.value;
}

$("#talkLanguage").onchange = syncVoiceProfiles;

async function speakNeuralOnline(text, profile) {
  const attempts = [profile];
  const lang = profile.startsWith("ta-") ? "ta" : "en";
  const fallbacks = lang === "ta"
    ? ["ta-female-pallavi","ta-male-valluvar","ta-female-kani","ta-male-surya","ta-female-venba","ta-male-anbu","ta-female-saranya","ta-male-kumar"]
    : ["en-female-aarti","en-female-neerja","en-female-ananya","en-female-kavya","en-male-arjun","en-male-prabhat","en-male-kunal"];
  for (const id of fallbacks) if (!attempts.includes(id)) attempts.push(id);
  let lastError = null;
  for (const id of attempts) {
    try {
      const r = await fetch("/api/tts", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({text, profile:id})
      });
      if (!r.ok) {
        let msg = "Neural voice failed.";
        try { const x=await r.json(); msg=x.error || msg; } catch (_) {}
        throw new Error(msg);
      }
      const blob = await r.blob();
      if (!blob.size) throw new Error("No audio was returned.");
      if (window.__vgAudioUrl) URL.revokeObjectURL(window.__vgAudioUrl);
      window.__vgAudioUrl = URL.createObjectURL(blob);
      const audio = new Audio(window.__vgAudioUrl);
      window.__vgAudio = audio;
      await audio.play();
      return id;
    } catch (e) { lastError=e; }
  }
  throw lastError || new Error("No neural voice could be generated.");
}

$("#speakResult").onclick = async () => {
  const text=$("#talkText").value.trim();
  if(!text) return alert("There is no result text to speak.");
  const lang=$("#talkLanguage").value;
  const profile=$("#voiceProfile").value;
  $("#speakResult").disabled=true;
  try {
    if (onlineMode) {
      $("#voiceCatalogNote").textContent = "Using Microsoft neural voice in Online Mode…";
      const usedProfile = await speakNeuralOnline(text, profile);
      const label = neuralProfiles.find(x => x[0] === usedProfile)?.[1] || usedProfile;
      $("#voiceCatalogNote").textContent = `Natural neural voice: ${label}`;
    } else {
      if(!window.speechSynthesis) throw new Error("Speech synthesis is not available in this browser.");
      speechSynthesis.cancel();
      const voice=chooseInstalledVoice(profile,lang);
      const u=new SpeechSynthesisUtterance(text);
      u.lang=lang;
      u.rate=profile.includes("young") ? .98 : .92;
      u.pitch=profile.includes("female") ? 1.02 : profile.includes("young") ? 1.08 : .98;
      if(voice)u.voice=voice;
      speechSynthesis.speak(u);
      $("#voiceCatalogNote").textContent = voice ? `Offline voice: ${voice.name}` : "No matching installed voice was found; browser default voice is being used.";
    }
  } catch(e) {
    alert(e.message || "Voice playback failed.");
  } finally {
    $("#speakResult").disabled=false;
  }
};

$("#stopSpeech").onclick=()=>{
  try { speechSynthesis?.cancel(); } catch (_) {}
  try { window.__vgAudio?.pause(); window.__vgAudio.currentTime=0; } catch (_) {}
};

// Initial state.
$("#voiceOfflineStatus").textContent = "Offline Vosk: add voice_models/en or voice_models/ta for fully local transcription. Browser fallback may use the browser's speech service.";
