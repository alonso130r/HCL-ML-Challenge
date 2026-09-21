const cameraCard = document.querySelector("#camera-card");
const cameraPreview = document.querySelector("#camera-preview");
const cameraToggle = document.querySelector("#camera-toggle");
const recordButton = document.querySelector("#record-button");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const intro = document.querySelector("#intro");
const statusText = document.querySelector("#status-text");
const emotionReadout = document.querySelector("#emotion-readout");
const emotionDiagnostic = document.querySelector("#emotion-diagnostic");
const streamState = document.querySelector("#stream-state");

let cameraStream = null;
let recorder = null;
let microphoneStream = null;
let recordedAudio = null;
let recordedChunks = [];

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
}

function addMessage(role, text = "") {
  intro.classList.add("hidden");
  const message = document.createElement("article");
  message.className = `message ${role}`;
  if (role === "assistant") {
    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = "MODEL OUTPUT";
    message.append(label);
  }
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  message.append(body);
  messages.append(message);
  message.scrollIntoView({ behavior: "smooth", block: "end" });
  return body;
}

cameraToggle.addEventListener("click", async () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
    cameraPreview.srcObject = null;
    cameraCard.classList.remove("active");
    cameraToggle.textContent = "Enable camera";
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    cameraPreview.srcObject = cameraStream;
    cameraCard.classList.add("active");
    cameraToggle.textContent = "Disable camera";
  } catch (_error) {
    setStatus("Camera permission was not granted", true);
  }
});

recordButton.addEventListener("click", async () => {
  if (recorder?.state === "recording") {
    recorder.stop();
    recordButton.classList.remove("recording");
    recordButton.setAttribute("aria-pressed", "false");
    recordButton.setAttribute("aria-label", "Start recording");
    setStatus("Preparing audio…");
    return;
  }
  try {
    microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(microphoneStream);
    recordedChunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) recordedChunks.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      recordedAudio = new Blob(recordedChunks, { type: recorder.mimeType });
      microphoneStream.getTracks().forEach((track) => track.stop());
      recorder = null;
      microphoneStream = null;
      setStatus("Audio ready · enter the matching transcript");
    });
    recorder.start();
    recordButton.classList.add("recording");
    recordButton.setAttribute("aria-pressed", "true");
    recordButton.setAttribute("aria-label", "Stop recording");
    setStatus("Recording… press again to stop");
  } catch (_error) {
    setStatus("Microphone permission was not granted", true);
  }
});

async function encodeAudio(blob) {
  if (!blob) return null;
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return btoa(binary);
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || sendButton.disabled) {
    if (!text) setStatus("Type a message to try the stream", true);
    return;
  }
  addMessage("user", text);
  input.value = "";
  resizeInput();
  sendButton.disabled = true;
  setStatus("Receiving stream…");
  streamState.textContent = "STREAMING";
  streamState.classList.add("active");

  const assistantBody = addMessage("assistant");
  assistantBody.classList.add("cursor");
  try {
    const audio = await encodeAudio(recordedAudio);
    recordedAudio = null;
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, audio }),
    });
    if (!response.ok || !response.body) throw new Error("Response could not be started.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = pending.split("\n");
      pending = done ? "" : lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "delta") assistantBody.textContent += event.text;
        if (event.type === "metadata") {
          emotionReadout.classList.add("ready");
          const confidence = Math.round(event.confidence * 100);
          emotionReadout.lastElementChild.textContent = `${event.emotion} · ${confidence}%`;
          const textConfidence = Math.round(event.text.confidence * 100);
          const audioConfidence = Math.round(event.audio.confidence * 100);
          const gate = Math.round(event.audio_gate * 100);
          emotionDiagnostic.textContent =
            `TEXT ${event.text.emotion} ${textConfidence}% · ` +
            `AUDIO ${event.audio.emotion} ${audioConfidence}% · ` +
            `AUDIO GATE ${gate}%`;
        }
        if (event.type === "error") throw new Error(event.message);
      }
      if (done) break;
    }
    setStatus("Ready");
  } catch (error) {
    assistantBody.textContent ||= "The response was interrupted. Please try again.";
    setStatus(error.message || "The response was interrupted", true);
  } finally {
    assistantBody.classList.remove("cursor");
    sendButton.disabled = false;
    streamState.textContent = "IDLE";
    streamState.classList.remove("active");
    input.focus();
  }
}

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});
sendButton.addEventListener("click", sendMessage);
