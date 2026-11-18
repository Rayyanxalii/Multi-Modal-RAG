// Frontend application logic for the document research assistant.

const savedActiveDoc = typeof localStorage !== "undefined" ? localStorage.getItem("rag_active_doc_id") : null;
let currentDocId = savedActiveDoc || null;
let allDocuments = [];
const chatHistories = {};
let isChatBusy = false;

// Creates or retrieves a persistent anonymous user ID from local storage.
function getOrCreateUserId() {
  const STORAGE_KEY = "rag_user_id";
  let uid = null;
  try {
    uid = localStorage.getItem(STORAGE_KEY);
  } catch (_) {}

  if (!uid || typeof uid !== "string" || !uid.startsWith("user_")) {
    const rand = Math.random().toString(36).substring(2, 10);
    const ts = Date.now().toString(36);
    uid = `user_${ts}_${rand}`;
    try {
      localStorage.setItem(STORAGE_KEY, uid);
    } catch (_) {}
  }
  return uid;
}

const currentUserId = getOrCreateUserId();

// Builds request headers including the user ID and optional Groq API key.
function getAuthHeaders(extraHeaders = {}) {
  const headers = {
    "X-User-ID": currentUserId,
    ...extraHeaders,
  };
  const groqKey = typeof localStorage !== "undefined" ? (localStorage.getItem("user_groq_key") || "") : "";
  if (groqKey) {
    headers["X-Groq-Api-Key"] = groqKey;
  }
  return headers;
}

let isDocViewerOpen = false;
let currentViewerDoc = currentDocId;
let currentPage = 1;
let totalPages = 1;
let docPagesList = [];
let currentZoom = 1.0;
let viewerMode = "pages";

document.addEventListener("DOMContentLoaded", () => {
  initApiKeyModal();
  initDocumentLedger();
  initChatForm();
  initSummarizeButton();
  initExportChat();
  initClearChatButton();
  initUploadModal();
  initDocViewer();

  if (window.renderMathInElement) {
    document.querySelectorAll(".answer-body").forEach((el) => {
      try {
        window.renderMathInElement(el, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "\\[", right: "\\]", display: true },
            { left: "\\(", right: "\\)", display: false },
          ],
          throwOnError: false,
        });
      } catch (e) {}
    });
  }
});

// Initializes the document ledger and restores history for the active document.
async function initDocumentLedger() {
  await loadDocuments();

  await loadDocumentHistory(currentDocId);
}

// Fetches the user's documents and renders them in the sidebar.
async function loadDocuments() {
  const docList = document.getElementById("docList");
  try {
    const res = await fetch(`/api/documents?user_id=${encodeURIComponent(currentUserId)}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const docs = await res.json();
    allDocuments = docs || [];

    docList.innerHTML = "";
    if (!docs || docs.length === 0) {
      docList.innerHTML = `<li style="font-size: 13px; color: var(--dp-color-ink-muted); padding: 8px 4px;">No documents yet.</li>`;
      currentDocId = null;
      if (typeof localStorage !== "undefined") localStorage.removeItem("rag_active_doc_id");
      updateInputPlaceholder();
      updateActiveBadge();
      renderInitialGreeting(null);
      return;
    }

    const docExists = currentDocId ? docs.some((d) => d.doc_id.toLowerCase() === currentDocId.toLowerCase()) : false;
    if (!docExists && docs.length > 0) {
      currentDocId = docs[0].doc_id;
      if (typeof localStorage !== "undefined") localStorage.setItem("rag_active_doc_id", currentDocId);
    }

    docs.forEach((doc) => {
      const li = document.createElement("li");
      const isSelected = currentDocId && doc.doc_id.toLowerCase() === currentDocId.toLowerCase();
      li.className = `doc-item-scrap ${isSelected ? "active" : ""}`;
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", isSelected ? "true" : "false");

      li.innerHTML = `
        <div class="doc-info" onclick="switchDocument('${doc.doc_id}')">
          <div class="doc-name">${escapeHtml(doc.title || doc.doc_id)}</div>
          <div class="doc-meta">${doc.pages || 0} pages</div>
        </div>
        <button class="btn-delete-scrap" title="Delete document" onclick="handleDeleteDoc(event, '${doc.doc_id}')" aria-label="Delete ${doc.doc_id}">
          ✕
        </button>
      `;
      docList.appendChild(li);
    });

    updateInputPlaceholder();
    updateActiveBadge();
  } catch (err) {
    console.error("Failed to load documents:", err);
  }
}

// Loads and renders saved conversation history for a document.
async function loadDocumentHistory(docId) {
  const chatStream = document.getElementById("chatStream");
  if (!chatStream) return;

  if (!docId) {
    renderInitialGreeting(null);
    return;
  }

  const displayName = getDocDisplayName(docId);
  chatStream.innerHTML = `
    <article class="message-entry assistant" id="historyLoader">
      <div class="assistant-working-sheet">
        <div class="answer-body" style="color: var(--dp-color-secondary); font-style: italic;">
          Loading conversation history for <strong>${escapeHtml(displayName)}</strong>...
        </div>
      </div>
    </article>
  `;

  try {
    const res = await fetch(`/api/documents/${docId}/history?user_id=${encodeURIComponent(currentUserId)}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    chatStream.innerHTML = "";
    if (data.messages && data.messages.length > 0) {
      data.messages.forEach((msg) => {
        if (msg.role === "user") {
          appendUserMessage(msg.content);
        } else if (msg.role === "assistant") {
          appendAssistantMessage(
            { answer: msg.content, sources: msg.sources || [], doc_id: data.doc_id || docId },
            { immediate: true }
          );
        }
      });
      chatHistories[docId] = chatStream.innerHTML;
    } else {
      renderInitialGreeting(docId);
    }
  } catch (err) {
    console.warn(`Failed to restore history for ${docId}:`, err);
    renderInitialGreeting(docId);
  }

  chatStream.scrollTop = chatStream.scrollHeight;
}

// Returns a human-readable display name for a document ID.
function getDocDisplayName(docId) {
  if (!docId) return "";
  const found = allDocuments.find((d) => d.doc_id.toLowerCase() === docId.toLowerCase());
  if (found && found.title) return found.title;
  return docId.replace(/[_-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Resolves the stored filename for a document.
function getDocFileName(docId) {
  const clean = (docId || "").trim();
  if (!clean) return "";
  if (/\.(pdf|txt)$/i.test(clean)) return clean;
  const found = allDocuments.find((d) => d.doc_id.toLowerCase() === clean.toLowerCase());
  if (found && found.filename) return found.filename;
  return `${clean}.pdf`;
}

// Enables or disables workspace actions based on document and busy state.
function updateWorkspaceActions() {
  const hasDoc = Boolean(currentDocId);
  const btnSummarize = document.getElementById("btnSummarize");
  const btnExport = document.getElementById("btnExport");

  if (btnSummarize) btnSummarize.disabled = !hasDoc || isChatBusy;
  if (btnExport) btnExport.disabled = !hasDoc || isChatBusy;

  if (!hasDoc) closeExportMenu();
}

// Sets the shared busy state and updates dependent controls.
function setChatBusy(busy) {
  isChatBusy = Boolean(busy);

  const btnSubmit = document.getElementById("btnSubmit");
  if (btnSubmit) btnSubmit.disabled = isChatBusy;

  updateWorkspaceActions();
}

// Updates the question input placeholder for the active document.
function updateInputPlaceholder() {
  const input = document.getElementById("questionInput");
  if (!input) return;
  if (currentDocId) {
    input.placeholder = `Ask a question about ${getDocDisplayName(currentDocId)}...`;
  } else {
    input.placeholder = "Upload or select a document to start asking questions...";
  }
}

// Switches the active document and restores its chat view.
async function switchDocument(docId, forceReload = false) {
  if (docId === currentDocId && !forceReload) return;

  const chatStream = document.getElementById("chatStream");
  if (currentDocId && chatStream) {
    chatHistories[currentDocId] = chatStream.innerHTML;
  }

  currentDocId = docId;
  if (typeof localStorage !== "undefined") {
    if (docId) localStorage.setItem("rag_active_doc_id", docId);
    else localStorage.removeItem("rag_active_doc_id");
  }
  updateActiveBadge();
  updateInputPlaceholder();
  closeExportMenu();

  if (chatHistories[docId] && !forceReload) {
    chatStream.innerHTML = chatHistories[docId];
    chatStream.scrollTop = chatStream.scrollHeight;
  } else {
    await loadDocumentHistory(docId);
  }

  updateDocListHighlight();

  if (docId) {
    if (isDocViewerOpen) {
      loadDocumentPages(docId);
    } else {
      currentViewerDoc = docId;
      currentPage = 1;
    }
  }
}

// Updates sidebar highlighting to reflect the active document.
function updateDocListHighlight() {
  document.querySelectorAll(".doc-item-scrap").forEach((el) => {
    const docName = el.querySelector(".doc-name")?.textContent.trim().toLowerCase();
    const isTarget = currentDocId && (docName === currentDocId.toLowerCase() || docName?.startsWith(currentDocId.toLowerCase()));
    el.classList.toggle("active", Boolean(isTarget));
    el.setAttribute("aria-selected", isTarget ? "true" : "false");
  });
}

// Wires up the clear chat button and its confirmation flow.
function initClearChatButton() {
  const btn = document.getElementById("btnClearChat");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    if (!currentDocId) {
      alert("No active document selected.");
      return;
    }

    const displayName = getDocDisplayName(currentDocId);
    const confirmed = confirm(
      `Clear chat history for "${displayName}"?\n\nThis will permanently delete all saved messages for this document from the database.`
    );
    if (!confirmed) return;

    btn.disabled = true;
    const originalHtml = btn.innerHTML;
    btn.innerHTML = `<span class="btn-clear-icon" aria-hidden="true">⏳</span><span>Clearing...</span>`;

    try {
      const res = await fetch(`/api/documents/${currentDocId}/history?user_id=${encodeURIComponent(currentUserId)}`, {
        method: "DELETE",
        headers: getAuthHeaders(),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }
      delete chatHistories[currentDocId];
      renderInitialGreeting(currentDocId);
    } catch (err) {
      console.error("Failed to clear chat history:", err);
      alert(`Could not clear chat history: ${err.message || err}`);
    } finally {
      btn.disabled = false;
      btn.innerHTML = originalHtml;
    }
  });
}

// Renders the initial greeting or empty state for a document.
function renderInitialGreeting(docId) {
  const chatStream = document.getElementById("chatStream");
  if (!chatStream) return;

  if (!docId) {
    chatStream.innerHTML = `
      <div class="chat-empty-state">
        <div class="empty-icon">📄</div>
        <h2 class="empty-title">Welcome to Omni Retrieve</h2>
        <p class="empty-subtitle">Upload any document (PDF) containing text, equations, figures, or tables using the <strong>+ Upload Document</strong> button on the left to begin asking questions.</p>
        <button class="btn-sketch-action" type="button" onclick="document.getElementById('btnOpenUpload').click()" style="margin-top: 4px;">
          <span>+ Upload Document</span>
        </button>
      </div>
    `;
    return;
  }

  const displayName = getDocDisplayName(docId);
  chatStream.innerHTML = `
    <div class="chat-empty-state">
      <div class="empty-icon">📖</div>
      <h2 class="empty-title">Ask any question about <strong>${escapeHtml(displayName)}</strong></h2>
      <p class="empty-subtitle">Explore text passages, mathematical equations, figures, diagrams, and tables with verifiable page citations.</p>
      <div class="empty-doc-pill">
        <span>📄 Active Document: ${escapeHtml(displayName)}</span>
      </div>
    </div>
  `;
  chatHistories[docId] = chatStream.innerHTML;
}

// Updates the active document badge in the header.
function updateActiveBadge() {
  const badge = document.getElementById("activeDocBadge");
  if (badge) {
    badge.textContent = currentDocId ? `DOC: ${currentDocId.toUpperCase()}` : "";
    badge.style.display = currentDocId ? "inline-block" : "none";
  }

  updateWorkspaceActions();
}

// Deletes a document after confirmation and refreshes the list.
async function handleDeleteDoc(event, docId) {
  event.stopPropagation();
  const displayName = getDocDisplayName(docId);
  const confirmed = confirm(`Are you sure you want to delete "${displayName}"?`);
  if (!confirmed) return;

  try {
    const res = await fetch(`/api/documents/${docId}?user_id=${encodeURIComponent(currentUserId)}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
    const data = await res.json();
    if (res.ok) {
      delete chatHistories[docId];
      if (currentDocId === docId) {
        currentDocId = null;
        if (typeof localStorage !== "undefined") localStorage.removeItem("rag_active_doc_id");
      }
      await loadDocuments();
      if (currentDocId) {
        await switchDocument(currentDocId, true);
      } else {
        renderInitialGreeting(null);
      }
    } else {
      alert(`Deletion notice: ${data.message || "Failed"}`);
    }
  } catch (err) {
    alert(`Error deleting document: ${err}`);
  }
}

// Initializes the chat form submission handler.
function initChatForm() {
  const chatForm = document.getElementById("chatForm");
  const questionInput = document.getElementById("questionInput");

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (isChatBusy) return;
    if (!currentDocId) {
      alert("Please upload or select a document from the left sidebar before asking questions.");
      return;
    }
    const query = questionInput.value.trim();
    if (!query) return;

    questionInput.value = "";
    await submitQuestion(query);
  });
}

// Submits a question through the streaming RAG pipeline.
async function submitQuestion(query, displayText = null) {
  if (!currentDocId) {
    alert("Please upload or select a document from the left sidebar before asking questions.");
    return;
  }
  if (isChatBusy) return;

  const questionInput = document.getElementById("questionInput");
  setChatBusy(true);
  appendUserMessage(displayText || query);

  try {
    await handleStreamingChat(query);
  } finally {
    setChatBusy(false);
    if (questionInput) questionInput.focus();
  }
}

// Wires up the one-click document summarize button.
function initSummarizeButton() {
  const btn = document.getElementById("btnSummarize");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    if (!currentDocId) {
      alert("Please select a document before requesting a summary.");
      return;
    }
    if (isChatBusy) return;

    const prompt =
      "Summarize this document with a structured executive summary covering its core objective, " +
      "key highlights and methodology, major results or financial highlights, and conclusion.";

    await submitQuestion(prompt, "⚡ Summarize this document");
  });
}

let isExportMenuOpen = false;

// Initializes export menu interactions.
function initExportChat() {
  const btnExport = document.getElementById("btnExport");
  const btnExportMd = document.getElementById("btnExportMd");
  const btnExportJson = document.getElementById("btnExportJson");
  const dropdown = document.getElementById("exportDropdown");

  if (!btnExport || !dropdown) return;

  btnExport.addEventListener("click", (e) => {
    e.stopPropagation();
    setExportMenuOpen(!isExportMenuOpen);
  });

  dropdown.addEventListener("click", (e) => e.stopPropagation());

  if (btnExportMd) {
    btnExportMd.addEventListener("click", () => {
      closeExportMenu();
      exportChat("markdown");
    });
  }

  if (btnExportJson) {
    btnExportJson.addEventListener("click", () => {
      closeExportMenu();
      exportChat("json");
    });
  }

  document.addEventListener("click", () => closeExportMenu());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeExportMenu();
  });
}

// Opens or closes the export dropdown menu.
function setExportMenuOpen(open) {
  const menu = document.getElementById("exportMenu");
  const btnExport = document.getElementById("btnExport");
  if (!menu || !btnExport) return;

  isExportMenuOpen = Boolean(open);
  menu.style.display = isExportMenuOpen ? "block" : "none";
  btnExport.setAttribute("aria-expanded", isExportMenuOpen ? "true" : "false");
}

// Closes the export dropdown if it is open.
function closeExportMenu() {
  if (isExportMenuOpen) setExportMenuOpen(false);
}

// Fetches conversation messages for a document.
async function fetchConversationMessages(docId) {
  const res = await fetch(
    `/api/documents/${encodeURIComponent(docId)}/history?user_id=${encodeURIComponent(currentUserId)}`,
    { headers: getAuthHeaders() }
  );
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Server error: ${res.status}`);
  }
  const data = await res.json();
  return Array.isArray(data.messages) ? data.messages : [];
}

// Builds a timestamped filename for an exported chat.
function buildExportFilename(docId, extension) {
  const stamp = new Date()
    .toISOString()
    .replace(/[:.]/g, "-")
    .replace("T", "_")
    .slice(0, 19);
  const stem = getDocFileName(docId)
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-zA-Z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "document";
  return `${stem}_chat_${stamp}.${extension}`;
}

// Builds a Markdown representation of the conversation.
function buildChatMarkdown(docId, messages, exportedAt) {
  const filename = getDocFileName(docId);
  const title = getDocDisplayName(docId);
  const lines = [
    "# Omni Retrieve — Chat Export",
    "",
    `- **Document:** ${filename}`,
    `- **Title:** ${title}`,
    `- **Document ID:** \`${docId}\``,
    `- **Exported:** ${exportedAt.toISOString()}`,
    `- **Messages:** ${messages.length}`,
    "",
    "---",
    "",
  ];

  messages.forEach((msg) => {
    const role = msg.role === "user" ? "You" : "Assistant";
    lines.push(`## ${role}`, "");

    if (msg.role === "user") {
      lines.push(String(msg.content || "").trim() || "_(empty message)_");
    } else {
      const { cleanText, pages } = cleanAnswerTextAndExtractPages(
        msg.content || "",
        msg.sources || []
      );
      lines.push(cleanText || "_(no answer content)_");

      if (pages.length > 0) {
        lines.push("", `**Sources:** [${filename}] Page ${pages.join(", ")}`);
      } else if (Array.isArray(msg.sources) && msg.sources.length > 0) {
        lines.push("", `**Sources:** [${filename}]`);
      }

      const visualSources = (msg.sources || []).filter((src) => {
        const type = String(src.type || "").toLowerCase();
        return type === "figure" || type === "table";
      });
      if (visualSources.length > 0) {
        const refs = visualSources.map((src) => {
          const label = String(src.type || "visual").toUpperCase();
          const page = (src.pages || []).join(", ") || "?";
          return `${label} (Page ${page})`;
        });
        lines.push("", `**Figures & Tables:** ${refs.join("; ")}`);
      }
    }

    lines.push("", "---", "");
  });

  return lines.join("\n");
}

// Builds a JSON payload of the conversation.
function buildChatJsonPayload(docId, messages, exportedAt) {
  return {
    application: "Omni Retrieve",
    exported_at: exportedAt.toISOString(),
    document: {
      doc_id: docId,
      filename: getDocFileName(docId),
      title: getDocDisplayName(docId),
    },
    message_count: messages.length,
    messages: messages.map((msg) => {
      if (msg.role === "user") {
        return { role: "user", content: String(msg.content || "") };
      }

      const { cleanText, pages } = cleanAnswerTextAndExtractPages(
        msg.content || "",
        msg.sources || []
      );
      return {
        role: msg.role || "assistant",
        content: cleanText,
        cited_pages: pages,
        sources: Array.isArray(msg.sources) ? msg.sources : [],
      };
    }),
  };
}

// Downloads text content as a file in the browser.
function downloadTextFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: `${mimeType};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

// Exports the current conversation in the requested format.
async function exportChat(format) {
  if (!currentDocId) {
    alert("Please select a document before exporting a conversation.");
    return;
  }

  const docId = currentDocId;
  const btnExport = document.getElementById("btnExport");
  if (btnExport) btnExport.disabled = true;

  try {
    const messages = await fetchConversationMessages(docId);
    if (messages.length === 0) {
      alert("There is no conversation to export yet. Ask a question first.");
      return;
    }

    const exportedAt = new Date();
    if (format === "json") {
      const payload = buildChatJsonPayload(docId, messages, exportedAt);
      downloadTextFile(
        JSON.stringify(payload, null, 2),
        buildExportFilename(docId, "json"),
        "application/json"
      );
    } else {
      downloadTextFile(
        buildChatMarkdown(docId, messages, exportedAt),
        buildExportFilename(docId, "md"),
        "text/markdown"
      );
    }
  } catch (err) {
    console.error("Failed to export chat:", err);
    alert(`Could not export the conversation: ${err.message || err}`);
  } finally {
    updateWorkspaceActions();
  }
}

// Handles streaming chat responses and renders the answer.
async function handleStreamingChat(query) {
  const stream = document.getElementById("chatStream");

  const entry = document.createElement("article");
  entry.className = "message-entry assistant";

  const workingSheet = document.createElement("div");
  workingSheet.className = "assistant-working-sheet";

  const statusDiv = document.createElement("div");
  statusDiv.className = "streaming-status-indicator";
  statusDiv.style.color = "var(--dp-color-secondary)";
  statusDiv.style.fontStyle = "italic";
  statusDiv.style.fontSize = "0.95rem";
  statusDiv.style.marginBottom = "8px";
  statusDiv.innerHTML = `<span class="progress-spinner" aria-hidden="true">⏳</span> Looking into document...`;
  workingSheet.appendChild(statusDiv);

  const answerBody = document.createElement("div");
  answerBody.className = "answer-body";
  workingSheet.appendChild(answerBody);

  entry.appendChild(workingSheet);
  stream.appendChild(entry);
  stream.scrollTop = stream.scrollHeight;

  let rawAnswer = "";
  let metadata = null;
  let hasStartedTokens = false;

  try {
    const userGroqKey = (typeof localStorage !== "undefined" ? localStorage.getItem("user_groq_key") : "") || "";
    const headers = getAuthHeaders({ "Content-Type": "application/json" });

    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers,
      body: JSON.stringify({
        question: query,
        doc_id: currentDocId,
        user_id: currentUserId,
        groq_api_key: userGroqKey || undefined,
      }),
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      const errorMsg = errJson.detail || `Server error: ${res.status}`;
      if (res.status === 401 || errorMsg.toLowerCase().includes("api key") || errorMsg.toLowerCase().includes("vision")) {
        openApiKeyModal("Please add a Vision LLM key to ask questions. (Hint: get a free key from Groq at console.groq.com)");
      }
      throw new Error(errorMsg);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          let data = null;
          try {
            data = JSON.parse(trimmed.slice(6));
          } catch (pErr) {
            console.warn("Error parsing chat SSE chunk:", pErr);
            continue;
          }

          if (data.type === "status") {
            if (!hasStartedTokens) {
              statusDiv.innerHTML = `<span class="progress-spinner" aria-hidden="true">⏳</span> ${escapeHtml(data.message)}`;
            }
          } else if (data.type === "token") {
            if (!hasStartedTokens) {
              hasStartedTokens = true;
              statusDiv.remove();
            }
            rawAnswer += data.delta;
            answerBody.innerHTML = `<span class="streaming-text">${escapeHtml(rawAnswer)}</span><span class="streaming-cursor"></span>`;
            stream.scrollTop = stream.scrollHeight;
          } else if (data.type === "metadata") {
            metadata = data;
          } else if (data.type === "error") {
            const errMsg = data.content || data.message || "Streaming error";
            if (errMsg.toLowerCase().includes("api key") || errMsg.includes("401") || errMsg.toLowerCase().includes("vision")) {
              openApiKeyModal("Please add a Vision LLM key to ask questions. (Hint: get a free key from Groq at console.groq.com)");
            }
            throw new Error(errMsg);
          }
        }
      }
    }

    if (statusDiv.parentNode) statusDiv.remove();

    const fullAnswerText = (metadata && metadata.full_answer) || rawAnswer;
    const sources = (metadata && metadata.sources) || [];

    const { cleanText, pages } = cleanAnswerTextAndExtractPages(fullAnswerText, sources);
    answerBody.innerHTML = parseMarkdownToHtml(cleanText);
    renderMathInElementSafely(answerBody);

    if (pages.length > 0) {
      const docLabel = (metadata && metadata.doc_id) || (sources[0] && sources[0].doc_id) || currentDocId;
      workingSheet.appendChild(renderSourcesBar(pages, docLabel));
    }

    const visualSources = getRelevantVisualSources(sources, fullAnswerText);
    if (visualSources.length > 0) {
      const deckContainer = document.createElement("div");
      deckContainer.innerHTML = renderVisualDeckHtml(visualSources);
      if (deckContainer.firstElementChild) {
        workingSheet.appendChild(deckContainer.firstElementChild);
      }
    }

    const routeBadge = document.getElementById("routeBadge");
    if (routeBadge && metadata && metadata.route) {
      routeBadge.textContent = `ROUTE: ${metadata.route.toUpperCase()}`;
    }

    stream.scrollTop = stream.scrollHeight;

  } catch (err) {
    if (statusDiv.parentNode) statusDiv.remove();
    answerBody.innerHTML = `<span style="color: #a00;">✕ Error: ${escapeHtml(err.message || String(err))}</span>`;
    if (err.message && (err.message.includes("401") || err.message.toLowerCase().includes("api key") || err.message.toLowerCase().includes("vision"))) {
      openApiKeyModal("Please add a Vision LLM key to ask questions. (Hint: get a free key from Groq at console.groq.com)");
    }
  }
}

// Appends a user message to the chat stream.
function appendUserMessage(text) {
  const stream = document.getElementById("chatStream");
  const emptyState = stream.querySelector(".chat-empty-state");
  if (emptyState) {
    emptyState.remove();
  }
  const entry = document.createElement("article");
  entry.className = "message-entry user";
  entry.innerHTML = `
    <div class="user-thought-scrap">
      <div>${escapeHtml(text)}</div>
    </div>
  `;
  stream.appendChild(entry);
  stream.scrollTop = stream.scrollHeight;
}

// Appends a temporary thinking indicator to the chat.
function appendThinkingIndicator() {
  const stream = document.getElementById("chatStream");
  const entry = document.createElement("article");
  entry.className = "message-entry assistant";
  entry.innerHTML = `
    <div class="assistant-working-sheet">
      <div class="answer-body" style="color: var(--dp-color-secondary); font-style: italic;">
        Searching document...
      </div>
    </div>
  `;
  stream.appendChild(entry);
  stream.scrollTop = stream.scrollHeight;
  return entry;
}

// Filters and deduplicates visual sources for display.
function getRelevantVisualSources(sources = [], answerText = "") {
  if (!Array.isArray(sources) || sources.length === 0) return [];

  const candidates = sources.filter(
    (s) => s.web_image_url || s.image_path || s.table_markdown
  );
  if (candidates.length === 0) return [];

  const seen = new Set();
  const deduped = [];
  for (const c of candidates) {
    const key = c.id || c.web_image_url || c.image_path;
    if (key && !seen.has(key)) {
      seen.add(key);
      deduped.push(c);
    }
  }

  return deduped.slice(0, 2);
}

let activeStreamingTimer = null;

// Appends an assistant message with optional streaming.
function appendAssistantMessage(data, options = {}) {
  if (!options.immediate) {
    return streamAssistantResponse(data);
  }

  const stream = document.getElementById("chatStream");
  const entry = document.createElement("article");
  entry.className = "message-entry assistant";

  const workingSheet = document.createElement("div");
  workingSheet.className = "assistant-working-sheet";

  const answerBody = document.createElement("div");
  answerBody.className = "answer-body";

  const { cleanText, pages } = cleanAnswerTextAndExtractPages(data.answer || "", data.sources || []);
  answerBody.innerHTML = parseMarkdownToHtml(cleanText);
  workingSheet.appendChild(answerBody);

  renderMathInElementSafely(answerBody);

  if (pages.length > 0) {
    const docLabel = (data && data.doc_id) || (data.sources && data.sources[0] && data.sources[0].doc_id) || currentDocId;
    workingSheet.appendChild(renderSourcesBar(pages, docLabel));
  }

  const visualSources = getRelevantVisualSources(data.sources, data.answer);
  if (visualSources.length > 0) {
    const deckContainer = document.createElement("div");
    deckContainer.innerHTML = renderVisualDeckHtml(visualSources);
    if (deckContainer.firstElementChild) {
      workingSheet.appendChild(deckContainer.firstElementChild);
    }
  }

  entry.appendChild(workingSheet);
  stream.appendChild(entry);
  stream.scrollTop = stream.scrollHeight;

  const routeBadge = document.getElementById("routeBadge");
  if (routeBadge) routeBadge.textContent = `ROUTE: ${(data.route || "both").toUpperCase()}`;
}

// Streams an assistant response with a typewriter effect.
function streamAssistantResponse(data) {
  const stream = document.getElementById("chatStream");
  const entry = document.createElement("article");
  entry.className = "message-entry assistant";

  const workingSheet = document.createElement("div");
  workingSheet.className = "assistant-working-sheet";
  workingSheet.style.cursor = "pointer";

  const answerBody = document.createElement("div");
  answerBody.className = "answer-body";
  workingSheet.appendChild(answerBody);

  entry.appendChild(workingSheet);
  stream.appendChild(entry);
  stream.scrollTop = stream.scrollHeight;

  const { cleanText, pages } = cleanAnswerTextAndExtractPages(data.answer || "", data.sources || []);

  const visualSources = getRelevantVisualSources(data.sources, data.answer);

  let isFinalized = false;

  // Finalizes the streamed assistant response rendering.
  const finalize = () => {
    if (isFinalized) return;
    isFinalized = true;

    if (activeStreamingTimer) {
      clearInterval(activeStreamingTimer);
      activeStreamingTimer = null;
    }

    workingSheet.style.cursor = "default";

    answerBody.innerHTML = parseMarkdownToHtml(cleanText);
    renderMathInElementSafely(answerBody);

    if (pages.length > 0) {
      const docLabel = (data && data.doc_id) || (data.sources && data.sources[0] && data.sources[0].doc_id) || currentDocId;
      workingSheet.appendChild(renderSourcesBar(pages, docLabel));
    }

    if (visualSources.length > 0) {
      const deckContainer = document.createElement("div");
      deckContainer.innerHTML = renderVisualDeckHtml(visualSources);
      if (deckContainer.firstElementChild) {
        workingSheet.appendChild(deckContainer.firstElementChild);
      }
    }

    const routeBadge = document.getElementById("routeBadge");
    if (routeBadge) routeBadge.textContent = `ROUTE: ${(data.route || "both").toUpperCase()}`;

    stream.scrollTop = stream.scrollHeight;
  };

  workingSheet.addEventListener("click", finalize, { once: true });

  const tokens = cleanText.split(/(\s+)/);
  if (tokens.length <= 1) {
    finalize();
    return;
  }

  let tokenIdx = 0;
  let displayedText = "";

  const batchSize = tokens.length > 350 ? 3 : (tokens.length > 180 ? 2 : 1);
  const intervalMs = 16;

  activeStreamingTimer = setInterval(() => {
    for (let b = 0; b < batchSize && tokenIdx < tokens.length; b++) {
      displayedText += tokens[tokenIdx];
      tokenIdx++;
    }

    answerBody.innerHTML = `<span class="streaming-text">${escapeHtml(displayedText)}</span><span class="streaming-cursor"></span>`;
    stream.scrollTop = stream.scrollHeight;

    if (tokenIdx >= tokens.length) {
      clearInterval(activeStreamingTimer);
      activeStreamingTimer = null;
      setTimeout(finalize, 80);
    }
  }, intervalMs);
}

// Cleans answer text and extracts cited page numbers.
function cleanAnswerTextAndExtractPages(rawText, sources = []) {
  if (!rawText) return { cleanText: "", pages: [] };

  const citedPages = new Set();
  const mathPlaceholders = [];
  let text = rawText;

  text = text.replace(/(?<![\\$])\$(?!\$)(?=\s*\(?[\d\.])/g, "___CURRENCY_DOLLAR___");

  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (m, eq) => {
    const idx = mathPlaceholders.length;
    mathPlaceholders.push({ type: "block", code: eq.trim() });
    return `___MATH_BLOCK_${idx}___`;
  });
  text = text.replace(/\\\[([\s\S]+?)\\\]/g, (m, eq) => {
    const idx = mathPlaceholders.length;
    mathPlaceholders.push({ type: "block", code: eq.trim() });
    return `___MATH_BLOCK_${idx}___`;
  });
  text = text.replace(/(?<!\$)\$([^\$\n]+?)\$(?!\$)/g, (m, eq) => {
    const trimmed = eq.trim();
    if (/\*\*|__/.test(trimmed)) return m;
    const idx = mathPlaceholders.length;
    mathPlaceholders.push({ type: "inline", code: trimmed });
    return `___MATH_INLINE_${idx}___`;
  });
  text = text.replace(/\\\(([\s\S]+?)\\\)/g, (m, eq) => {
    const idx = mathPlaceholders.length;
    mathPlaceholders.push({ type: "inline", code: eq.trim() });
    return `___MATH_INLINE_${idx}___`;
  });

  text = text.replace(/\[([^\]]*?Page[s]?\s*\d+[^\]]*?)\]/gi, (match) => {
    const pageMatches = [...match.matchAll(/Page[s]?\s*(\d+)/gi)];
    pageMatches.forEach((m) => {
      const num = parseInt(m[1], 10);
      if (!isNaN(num)) citedPages.add(num);
    });
    return "";
  });

  text = text.replace(/\[Source\s*(\d+)\]/gi, (match, srcNum) => {
    const idx = parseInt(srcNum, 10) - 1;
    if (sources && sources[idx] && Array.isArray(sources[idx].pages)) {
      sources[idx].pages.forEach((p) => {
        const num = parseInt(p, 10);
        if (!isNaN(num)) citedPages.add(num);
      });
    }
    return "";
  });

  text = text.replace(/\[(?:Source\s*)?\d+(?:\s*,\s*\d+)*\]/gi, "");

  text = text.replace(/\s+([.,;:!?])/g, "$1");

  text = text.replace(/[ \t]{2,}/g, " ");

  if (citedPages.size === 0 && Array.isArray(sources)) {
    sources.forEach((src) => {
      if (Array.isArray(src.pages)) {
        src.pages.forEach((p) => {
          const num = parseInt(p, 10);
          if (!isNaN(num)) citedPages.add(num);
        });
      }
    });
  }

  text = text.replace(/___MATH_BLOCK_(\d+)___/g, (m, idx) => `$$${mathPlaceholders[idx].code}$$`);
  text = text.replace(/___MATH_INLINE_(\d+)___/g, (m, idx) => `$${mathPlaceholders[idx].code}$`);
  text = text.replace(/___CURRENCY_DOLLAR___/g, "$");

  const sortedPages = [...citedPages].sort((a, b) => a - b);
  return { cleanText: text.trim(), pages: sortedPages };
}

// Parses Markdown and LaTeX into HTML.
function parseMarkdownToHtml(rawText) {
  if (!rawText) return "";

  const mathTokens = [];
  let s = rawText;

  s = s.replace(/_{1,3}MATH_TOKEN_\d+_{1,3}/g, "$");
  s = s.replace(/_{1,3}CURRENCY_DOLLAR_{1,3}/g, "$");

  s = s.replace(/(?<![\\$])\$(?!\$)(?=\s*\(?[\d\.])/g, "@@@CURRENCY_DOLLAR@@@");

  s = s.replace(/\$\$([\s\S]+?)\$\$/g, (m, eq) => {
    const idx = mathTokens.length;
    mathTokens.push({ type: "block", code: eq.trim() });
    return `@@@MATH_TOKEN_${idx}@@@`;
  });
  s = s.replace(/\\\[([\s\S]+?)\\\]/g, (m, eq) => {
    const idx = mathTokens.length;
    mathTokens.push({ type: "block", code: eq.trim() });
    return `@@@MATH_TOKEN_${idx}@@@`;
  });

  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (m, eq) => {
    const idx = mathTokens.length;
    mathTokens.push({ type: "inline", code: eq.trim() });
    return `@@@MATH_TOKEN_${idx}@@@`;
  });

  s = s.replace(/(?<!\$)\$([^\$\n]+?)\$(?!\$)/g, (fullMatch, content) => {
    const trimmed = content.trim();
    if (/\*\*|__/.test(trimmed)) {
      return fullMatch;
    }
    const words = trimmed.split(/\s+/);

    const hasMathSymbols = /[\\=+^_{}[\]<>~/|]/.test(trimmed) || /^[a-zA-Z]$/.test(trimmed);
    if (!hasMathSymbols && words.length > 2) {
      return fullMatch;
    }
    const idx = mathTokens.length;
    mathTokens.push({ type: "inline", code: trimmed });
    return `@@@MATH_TOKEN_${idx}@@@`;
  });

  s = s.replace(/([.:;])\s*[\*\-]\s+(?=\*\*|[a-zA-Z])/g, "$1\n   * ");

  s = escapeHtml(s);

  s = s.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]+?)```/g, (m, lang, code) => {
    return `<pre class="md-code-block"><code class="language-${lang}">${code.trim()}</code></pre>`;
  });

  s = s.replace(/`([^`\n]+)`/g, '<code class="md-code">$1</code>');

  s = s.replace(/^####\s+(.+)$/gm, '<h5 class="md-subheading">$1</h5>');
  s = s.replace(/^###\s+(.+)$/gm, '<h4 class="md-subheading">$1</h4>');
  s = s.replace(/^##\s+(.+)$/gm, '<h3 class="md-heading">$1</h3>');
  s = s.replace(/^#\s+(.+)$/gm, '<h2 class="md-title">$1</h2>');

  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/__(.+?)__/g, '<strong>$1</strong>');

  s = s.replace(/(?<!\*)\*([^\*\n]+?)\*(?!\*)/g, '<em>$1</em>');
  s = s.replace(/(^|[\s(])_([^\s_][^_]*?[^\s_]|[^_\s])_([)\s.,;:!?]|$)/g, '$1<em>$2</em>$3');

  s = s.replace(/^[ \t]{2,}[\*\-]\s+(.+)$/gm, '<li class="md-sublist-item">$1</li>');

  s = s.replace(/^[ \t]*[\*\-]\s+(.+)$/gm, '<li class="md-list-item">$1</li>');

  s = s.replace(/^[ \t]*(\d+)\.\s+(.+)$/gm, '<li class="md-ordered-item" data-num="$1">$2</li>');

  s = s.replace(/((?:<li class="md-sublist-item">[\s\S]+?<\/li>\s*)+)/g, '<ul class="md-sublist">$1</ul>');
  s = s.replace(/((?:<li class="md-list-item">[\s\S]+?<\/li>\s*)+)/g, '<ul class="md-list">$1</ul>');
  s = s.replace(/((?:<li class="md-ordered-item"[^>]*>[\s\S]+?<\/li>\s*)+)/g, '<ol class="md-ordered-list">$1</ol>');

  s = s.replace(/<\/ul>\s*<ul class="md-list">/g, '');
  s = s.replace(/<\/ol>\s*<ol class="md-ordered-list">/g, '');

  s = s.replace(/\n{2,}/g, '<br><br>');

  s = s.replace(/([^>])\n([^<])/g, '$1<br>$2');

  s = s.replace(/@@@MATH_TOKEN_(\d+)@@@/g, (match, idxStr) => {
    const token = mathTokens[parseInt(idxStr, 10)];
    if (!token) return match;

    if (/^[\d,.]+$/.test(token.code)) {
      return `$${token.code}`;
    }

    if (window.katex && typeof window.katex.renderToString === "function") {
      try {
        const isBlock = token.type === "block";
        const rendered = window.katex.renderToString(token.code, {
          displayMode: isBlock,
          throwOnError: false,
        });
        if (isBlock) {
          return `<div class="math-block">${rendered}</div>`;
        }
        return `<span class="math-inline">${rendered}</span>`;
      } catch (err) {
        console.warn("KaTeX direct render error:", err);
      }
    }
    return formatMathFallback(token.code, token.type === "block");
  });

  s = s.replace(/@@@CURRENCY_DOLLAR@@@/g, "$");

  return s;
}

// Renders the sources bar with page jump buttons.
function renderSourcesBar(pages, docName) {
  const bar = document.createElement("div");
  bar.className = "sources-provenance-bar";

  const label = document.createElement("span");
  label.className = "sources-label";
  label.textContent = "Sources:";
  bar.appendChild(label);

  const cleanDoc = (docName || currentDocId || "").trim();
  if (cleanDoc) {
    const docSpan = document.createElement("span");
    docSpan.className = "source-doc-pill";
    const displayDoc = getDocFileName(cleanDoc);
    docSpan.textContent = `[${displayDoc}]`;
    docSpan.title = `Document: ${displayDoc}`;
    docSpan.setAttribute("data-doc-id", cleanDoc);
    bar.appendChild(docSpan);
  }

  if (pages && pages.length > 0) {
    const prefix = document.createElement("span");
    prefix.className = "sources-prefix";
    prefix.textContent = "Page";
    bar.appendChild(prefix);

    const listSpan = document.createElement("span");
    listSpan.className = "sources-list";

    pages.forEach((p, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "source-page-pill";
      btn.textContent = `${p}`;
      btn.title = `View Page ${p} beside chat`;
      btn.onclick = (e) => {
        e.stopPropagation();
        openAndJumpToPage(p);
      };
      listSpan.appendChild(btn);

      if (idx < pages.length - 1) {
        const sep = document.createTextNode(", ");
        listSpan.appendChild(sep);
      }
    });

    bar.appendChild(listSpan);
  }

  return bar;
}

// Renders HTML for the visual references deck.
function renderVisualDeckHtml(visualSources) {
  if (!visualSources || visualSources.length === 0) return "";
  return `
    <div class="visual-references-deck">
      <div class="deck-label">
        <span>Figures &amp; Tables</span>
      </div>
      <div class="taped-references-grid">
        ${visualSources.map((src) => renderTapedMount(src)).join("")}
      </div>
    </div>
  `;
}

// Safely renders math in an element with KaTeX.
function renderMathInElementSafely(element) {
  if (!element || !window.renderMathInElement) return;
  try {
    window.renderMathInElement(element, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  } catch (err) {
    console.warn("KaTeX post-render notice:", err);
  }
}

// Formats a math expression as fallback text.
function formatMathFallback(code, isBlock) {
  let s = code
    .replace(/\\text\{([^\}]+)\}/g, "$1")
    .replace(/\\mathrm\{([^\}]+)\}/g, "$1")
    .replace(/\\left\(/g, "(")
    .replace(/\\right\)/g, ")")
    .replace(/\\left\[/g, "[")
    .replace(/\\right\]/g, "]")
    .replace(/\\frac\{([^\}]+)\}\{([^\}]+)\}/g, "($1 / $2)")
    .replace(/\\sqrt\{([^\}]+)\}/g, "√($1)")
    .replace(/\\sqrt\s*([a-zA-Z0-9]+)/g, "√$1")
    .replace(/\\cdot/g, "·")
    .replace(/\\times/g, "×")
    .replace(/\\le/g, "≤")
    .replace(/\\ge/g, "≥")
    .replace(/\\neq/g, "≠")
    .replace(/\\approx/g, "≈")
    .replace(/\\in/g, "∈")
    .replace(/_k\b/g, "ₖ")
    .replace(/_v\b/g, "ᵥ")
    .replace(/_i\b/g, "ᵢ")
    .replace(/_j\b/g, "ⱼ")
    .replace(/\^T\b/g, "ᵀ")
    .replace(/\^2\b/g, "²")
    .replace(/\^d\b/g, "ᵈ");

  s = escapeHtml(s);
  if (isBlock) {
    return `<div class="math-block math-fallback">${s}</div>`;
  }
  return `<span class="math-inline math-fallback">${s}</span>`;
}

// Renders a taped mount card for a visual source.
function renderTapedMount(src) {
  const isTable = (src.type || "").toLowerCase() === "table" || Boolean(src.table_markdown);
  const pages = (src.pages || []).join(", ") || "?";
  const docTag = (src.doc_id || currentDocId) ? `[${getDocFileName(src.doc_id || currentDocId)}] ` : "";
  const badgeText = isTable ? `${docTag}TABLE [PAGE ${pages}]` : `${docTag}FIGURE [PAGE ${pages}]`;
  const caption = src.caption || src.panel_caption || "Visual asset extracted from document";
  const targetPage = (src.pages && src.pages.length > 0 && parseInt(src.pages[0], 10)) || 1;

  let imgUrl = src.web_image_url;
  if (!imgUrl && src.image_path) {
    const rawPath = String(src.image_path).replace(/\\/g, "/");
    const filename = rawPath.split("/").pop();
    const folder = isTable || filename.toLowerCase().includes("tab") ? "tables" : "figures";
    imgUrl = `/data/${folder}/${filename}`;
  } else if (imgUrl && imgUrl.includes("src/pdf/")) {
    const filename = imgUrl.split("/").pop();
    const folder = isTable || filename.toLowerCase().includes("tab") ? "tables" : "figures";
    imgUrl = `/data/${folder}/${filename}`;
  }

  let bodyHtml = "";
  if (!isTable && imgUrl) {
    bodyHtml = `<img class="mount-image" src="${escapeHtml(imgUrl)}" alt="${escapeHtml(caption)}" loading="lazy" onerror="this.style.display='none'">`;
  } else if (isTable) {
    if (src.table_markdown) {
      bodyHtml = `<pre class="table-markdown-box">${escapeHtml(src.table_markdown)}</pre>`;
    } else if (imgUrl) {
      bodyHtml = `<img class="mount-image" src="${escapeHtml(imgUrl)}" alt="${escapeHtml(caption)}" loading="lazy" onerror="this.style.display='none'">`;
    }
  }

  return `
    <div class="taped-mount clickable-mount" onclick="openAndJumpToPage(${targetPage})" title="Click to view Page ${targetPage} beside chat">
      <div class="mount-badge">${escapeHtml(badgeText)}</div>
      ${bodyHtml}
      <div class="mount-caption">${escapeHtml(caption)}</div>
    </div>
  `;
}

// Appends an error message to the chat stream.
function appendErrorMessage(errText) {
  const stream = document.getElementById("chatStream");
  const entry = document.createElement("article");
  entry.className = "message-entry assistant";
  entry.innerHTML = `
    <div class="assistant-working-sheet professional-card" style="border-left: 3px solid var(--dp-color-accent-ink);">
      <div class="answer-body" style="color: var(--dp-color-accent-ink);">
        ${escapeHtml(errText)}
      </div>
    </div>
  `;
  stream.appendChild(entry);
  stream.scrollTop = stream.scrollHeight;
}

// Updates ingestion progress UI elements.
function updateIngestionUI(pct, text, filename) {
  const box = document.getElementById("uploadProgressBox");
  const stageText = document.getElementById("progressStageText");
  const percentText = document.getElementById("progressPercent");
  const fill = document.getElementById("progressBarFill");
  const globalBar = document.getElementById("globalIngestBar");
  const globalText = document.getElementById("globalIngestText");
  const globalPercent = document.getElementById("globalIngestPercent");

  if (box) box.style.display = "block";
  if (globalBar) globalBar.style.display = "flex";

  const rounded = Math.min(Math.round(pct), 100);
  if (fill) fill.style.width = `${rounded}%`;
  if (percentText) percentText.textContent = `${rounded}%`;
  if (stageText) {
    stageText.innerHTML = `<span class="progress-spinner" aria-hidden="true">⏳</span><span>${escapeHtml(text)}</span>`;
  }
  if (globalText) {
    globalText.textContent = `INGESTING [${(filename || "").toUpperCase()}]: ${text}`;
  }
  if (globalPercent) {
    globalPercent.textContent = `${rounded}%`;
  }
}

// Initializes ingestion progress UI.
function startIngestionProgress(filename) {
  updateIngestionUI(2, "Uploading PDF & initializing workspace...", filename);
}

// Marks ingestion progress as complete or failed.
function completeIngestionProgress(success = true, message = "") {
  const stageText = document.getElementById("progressStageText");
  const percentText = document.getElementById("progressPercent");
  const fill = document.getElementById("progressBarFill");
  const globalBar = document.getElementById("globalIngestBar");
  const globalText = document.getElementById("globalIngestText");
  const globalPercent = document.getElementById("globalIngestPercent");

  if (success) {
    if (fill) fill.style.width = "100%";
    if (percentText) percentText.textContent = "100%";
    const successMsg = message || "Ingestion complete! Opening workspace...";
    if (stageText) {
      stageText.innerHTML = `<span style="font-weight: 700;">✓</span><span>${escapeHtml(successMsg)}</span>`;
    }
    if (globalText) {
      globalText.textContent = `✓ INGESTION COMPLETE: ${successMsg}`;
    }
    if (globalPercent) {
      globalPercent.textContent = "100%";
    }
  } else {
    const errorMsg = message || "Ingestion failed.";
    if (stageText) {
      stageText.innerHTML = `<span style="font-weight: 700; color: #a00;">✕</span><span>${escapeHtml(errorMsg)}</span>`;
    }
    if (globalText) {
      globalText.textContent = `✕ INGESTION FAILED: ${errorMsg}`;
    }
  }
}

let isApiKeyModalOpen = false;

// Opens the API key modal.
function openApiKeyModal(noticeMessage = "") {
  const modal = document.getElementById("apiKeyModal");
  const input = document.getElementById("groqApiKeyInput");
  const notice = document.getElementById("apiKeyNotice");
  const btnClear = document.getElementById("btnClearApiKey");

  if (!modal) return;

  const currentKey = localStorage.getItem("user_groq_key") || "";
  if (input) {
    input.value = currentKey;
  }

  if (btnClear) {
    btnClear.style.display = currentKey ? "inline-block" : "none";
  }

  if (notice) {
    if (noticeMessage) {
      notice.textContent = noticeMessage;
      notice.className = "api-key-notice error";
      notice.style.display = "block";
    } else if (currentKey) {
      const tail = currentKey.length >= 4 ? currentKey.slice(-4) : "****";
      notice.textContent = `✓ Active in browser (ending in ...${tail}). Sent automatically with every question.`;
      notice.className = "api-key-notice success";
      notice.style.display = "block";
    } else {
      notice.style.display = "none";
    }
  }

  modal.style.display = "flex";
  isApiKeyModalOpen = true;
  if (input) {
    setTimeout(() => input.focus(), 50);
  }
}

// Closes the API key modal.
function closeApiKeyModal() {
  const modal = document.getElementById("apiKeyModal");
  if (modal) {
    modal.style.display = "none";
    isApiKeyModalOpen = false;
  }
}

// Updates the API key status indicator.
async function updateApiKeyStatusUI() {
  const statusDot = document.getElementById("apiKeyStatusDot");
  const btnLabel = document.getElementById("btnApiKeyLabel");
  const userKey = localStorage.getItem("user_groq_key") || "";

  if (userKey) {
    if (statusDot) {
      statusDot.className = "status-dot-active";
      statusDot.title = "Personal Vision LLM Key Active (Groq)";
    }
    if (btnLabel) {
      btnLabel.textContent = "Vision Key ✓";
    }
    return;
  }

  try {
    const res = await fetch("/api/auth/status");
    if (res.ok) {
      const data = await res.json();
      if (data.has_server_key) {
        if (statusDot) {
          statusDot.className = "status-dot-active";
          statusDot.title = "Server Default Key Active";
        }
        if (btnLabel) {
          btnLabel.textContent = "Vision Key";
        }
        return;
      }
    }
  } catch (err) {
    console.warn("Could not check /api/auth/status:", err);
  }

  if (statusDot) {
    statusDot.className = "status-dot-inactive";
    statusDot.title = "Vision LLM Key Not Configured (Click to add free Groq key)";
  }
  if (btnLabel) {
    btnLabel.textContent = "Vision Key !";
  }
}

// Initializes API key modal interactions.
function initApiKeyModal() {
  const modal = document.getElementById("apiKeyModal");
  const btnOpen = document.getElementById("btnOpenApiKeyModal");
  const btnClose = document.getElementById("btnCloseApiKeyModal");
  const form = document.getElementById("apiKeyForm");
  const input = document.getElementById("groqApiKeyInput");
  const btnToggle = document.getElementById("btnToggleKeyVisibility");
  const btnClear = document.getElementById("btnClearApiKey");
  const notice = document.getElementById("apiKeyNotice");

  if (!modal) return;

  if (btnOpen) {
    btnOpen.addEventListener("click", () => openApiKeyModal());
  }

  if (btnClose) {
    btnClose.addEventListener("click", () => closeApiKeyModal());
  }

  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeApiKeyModal();
  });

  if (btnToggle && input) {
    btnToggle.addEventListener("click", () => {
      const isPassword = input.type === "password";
      input.type = isPassword ? "text" : "password";
      btnToggle.textContent = isPassword ? "🔒" : "👁️";
    });
  }

  if (btnClear) {
    btnClear.addEventListener("click", () => {
      localStorage.removeItem("user_groq_key");
      if (input) input.value = "";
      if (notice) {
        notice.textContent = "Personal API key removed. Using server default if configured.";
        notice.className = "api-key-notice success";
        notice.style.display = "block";
      }
      btnClear.style.display = "none";
      updateApiKeyStatusUI();
      setTimeout(() => closeApiKeyModal(), 1200);
    });
  }

  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const key = (input ? input.value : "").trim();
      if (!key) {
        if (notice) {
          notice.textContent = "Please enter an API key or click Cancel.";
          notice.className = "api-key-notice error";
          notice.style.display = "block";
        }
        return;
      }

      if (!key.startsWith("gsk_") && key.length < 20) {
        if (notice) {
          notice.textContent = "Warning: Groq API keys usually start with 'gsk_'. Please ensure your key is valid.";
          notice.className = "api-key-notice error";
          notice.style.display = "block";
        }
      }

      localStorage.setItem("user_groq_key", key);
      updateApiKeyStatusUI();

      if (notice) {
        notice.textContent = "✓ Groq API Key saved successfully in your browser.";
        notice.className = "api-key-notice success";
        notice.style.display = "block";
      }

      setTimeout(() => {
        closeApiKeyModal();
      }, 700);
    });
  }

  updateApiKeyStatusUI();
}

// Initializes the upload modal and ingestion flow.
function initUploadModal() {
  const modal = document.getElementById("uploadModal");
  const btnOpen = document.getElementById("btnOpenUpload");
  const btnClose = document.getElementById("btnCloseUpload");
  const uploadForm = document.getElementById("uploadForm");
  const fileInput = document.getElementById("pdfFileInput");
  const btnSubmitUpload = document.getElementById("btnSubmitUpload");
  const errorNotice = document.getElementById("uploadErrorNotice");
  const limitHint = document.getElementById("uploadLimitHint");

  let maxUploadMb = 10;
  let maxUploadPages = 25;

  // Shows an upload error message.
  const showUploadError = (msg) => {
    if (errorNotice) {
      errorNotice.textContent = msg;
      errorNotice.style.display = "block";
    }
  };

  // Clears the upload error message.
  const clearUploadError = () => {
    if (errorNotice) {
      errorNotice.style.display = "none";
      errorNotice.textContent = "";
    }
  };

  // Syncs upload size and page limits from the server.
  const syncUploadLimits = async () => {
    try {
      const res = await fetch("/api/auth/status");
      if (res.ok) {
        const data = await res.json();
        if (typeof data.max_upload_mb === "number") maxUploadMb = data.max_upload_mb;
        if (typeof data.max_upload_pages === "number") maxUploadPages = data.max_upload_pages;
        if (limitHint) {
          const parserLabel =
            data.parser_mode === "docling"
              ? "Docling"
              : data.parser_mode === "pymupdf_lite"
                ? "PyMuPDF Lite"
                : "";
          const parserNote = parserLabel ? ` Parser: ${parserLabel}.` : "";
          limitHint.textContent = `Upload a PDF or TXT document (max ${maxUploadMb} MB, ${maxUploadPages} pages) to analyze its text, figures, and tables.${parserNote}`;
        }
      }
    } catch (_) {}
  };

  syncUploadLimits();

  if (fileInput) {
    fileInput.addEventListener("change", () => clearUploadError());
  }

  btnOpen.addEventListener("click", () => {
    clearUploadError();
    syncUploadLimits();
    modal.classList.add("open");
    fileInput.focus();
  });

  btnClose.addEventListener("click", () => {
    modal.classList.remove("open");
  });

  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("open");
  });

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearUploadError();
    if (!fileInput.files || fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    if (file.size > maxUploadMb * 1024 * 1024) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      const errMsg = `File exceeds the ${maxUploadMb} MB limit (${sizeMb} MB selected). Please choose a smaller file.`;
      showUploadError(errMsg);
      return;
    }

    // Delays execution for the given milliseconds.
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    // Builds form data for an upload request.
    const buildUploadFormData = () => {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("user_id", currentUserId);
      return fd;
    };

    // Starts an ingestion job with retries.
    const startIngestJob = async () => {
      let lastError = null;
      for (let attempt = 0; attempt < 30; attempt += 1) {
        try {
          const res = await fetch("/api/upload", {
            method: "POST",
            headers: { "X-User-ID": currentUserId },
            body: buildUploadFormData(),
          });
          if (res.ok) return await res.json();

          if (res.status >= 500) {
            lastError = new Error(`Upload service returned HTTP ${res.status}.`);
          } else {
            const errorData = await res.json().catch(() => ({}));
            const error = new Error(errorData.detail || `Upload failed: ${res.status}`);
            error.userFacing = true;
            throw error;
          }
        } catch (err) {
          if (err && err.userFacing) throw err;
          lastError = err;
        }

        if (attempt < 29) {
          updateIngestionUI(4, "Waiting for the server to come back online...", file.name);
          await sleep(3000);
        }
      }
      throw new Error(
        `Could not reach the upload service${lastError ? `: ${lastError.message}` : "."}`
      );
    };

    // Polls an ingestion job until completion.
    const pollIngestJob = async (jobId) => {
      let transientErrors = 0;
      while (true) {
        await sleep(900);

        let statusRes;
        try {
          statusRes = await fetch(`/api/upload/status/${encodeURIComponent(jobId)}`);
        } catch (netErr) {
          transientErrors += 1;
          if (transientErrors >= 15) {
            const error = new Error(`Network connection lost while checking ingestion status: ${netErr.message}`);
            error.serverRestarted = true;
            throw error;
          }
          continue;
        }

        if (statusRes.status === 404) {
          const errJson = await statusRes.json().catch(() => ({}));
          const error = new Error(
            errJson.detail || "The server restarted while ingesting this document."
          );
          error.serverRestarted = true;
          throw error;
        }

        if (!statusRes.ok) {
          transientErrors += 1;
          if (transientErrors >= 15) {
            if (statusRes.status === 502 || statusRes.status === 503 || statusRes.status === 504) {
              const error = new Error(
                `The server became unreachable (HTTP ${statusRes.status}) while ingesting this document.`
              );
              error.serverRestarted = true;
              throw error;
            }
            throw new Error(`Server status check failed (HTTP ${statusRes.status}).`);
          }
          continue;
        }

        transientErrors = 0;
        const job = await statusRes.json();

        if (job.progress !== undefined) {
          updateIngestionUI(job.progress, job.message || "Processing...", file.name);
        }

        if (job.stage === "error" || job.error) {
          throw new Error(job.error || job.message || "Ingestion error encountered on backend.");
        }

        if (job.done || job.stage === "complete") {
          return job;
        }
      }
    };

    startIngestionProgress(file.name);
    btnSubmitUpload.disabled = true;
    btnSubmitUpload.innerHTML = `<span>Ingesting...</span> <span aria-hidden="true">⏳</span>`;
    btnCloseUpload.disabled = true;

    try {

      let initData = await startIngestJob();
      let finalDocId = initData.doc_id || null;
      let finalMessage = "";
      let colpaliExisted = false;
      let alreadyExisted = false;

      if (!initData.job_id) {
        throw new Error("Server did not return an ingestion job ID.");
      }

      updateIngestionUI(5, `Uploaded ${file.name}. Starting pipeline...`, file.name);

      for (let restartRetries = 0; ; restartRetries += 1) {
        try {
          const job = await pollIngestJob(initData.job_id);
          finalDocId = job.doc_id || finalDocId;
          finalMessage = job.message || `Successfully ingested ${file.name}!`;
          colpaliExisted = Boolean(job.colpali_existed);
          alreadyExisted = Boolean(job.already_existed);
          break;
        } catch (err) {
          if (err && err.serverRestarted && restartRetries < 1) {
            updateIngestionUI(
              5,
              "The server restarted during ingestion. Resuming automatically — completed steps are cached...",
              file.name
            );
            await sleep(3000);
            initData = await startIngestJob();
            if (!initData.job_id) {
              throw new Error("Server did not return an ingestion job ID after restart.");
            }
            finalDocId = initData.doc_id || finalDocId;
            continue;
          }
          if (err && err.serverRestarted) {
            throw new Error(
              "The server restarted again while ingesting this document, most likely because the instance ran out of memory. " +
              "Completed steps are cached, so retrying the upload will resume from the last finished stage. " +
              "If this repeats, upgrade the Render instance to more RAM."
            );
          }
          throw err;
        }
      }

      completeIngestionProgress(true, finalMessage || `Successfully ingested ${file.name}`);

      const delayMs = alreadyExisted ? 1800 : 750;
      await new Promise((resolve) => setTimeout(resolve, delayMs));

      modal.classList.remove("open");
      uploadForm.reset();

      await loadDocuments();
      if (finalDocId) {
        await switchDocument(finalDocId, true);
        if (colpaliExisted) {
          const chatStream = document.getElementById("chatStream");
          if (chatStream) {
            const noticeEl = document.createElement("div");
            noticeEl.className = "colpali-reuse-notice";
            noticeEl.innerHTML = `<span aria-hidden="true">ℹ️</span> <span><strong>Existing ColPali Pages Detected:</strong> Visual embeddings for this document already existed on ColPali and were linked cleanly. Ask any question below!</span>`;
            chatStream.prepend(noticeEl);
          }
        }
        const qInput = document.getElementById("questionInput");
        if (qInput) qInput.focus();
      }

      setTimeout(() => {
        const box = document.getElementById("uploadProgressBox");
        const globalBar = document.getElementById("globalIngestBar");
        if (box) box.style.display = "none";
        if (globalBar) globalBar.style.display = "none";
      }, alreadyExisted ? 5500 : 1800);
    } catch (err) {
      const errMsg = err.message || String(err);
      completeIngestionProgress(false, errMsg);
      showUploadError(errMsg);
      setTimeout(() => {
        const box = document.getElementById("uploadProgressBox");
        const globalBar = document.getElementById("globalIngestBar");
        if (box) box.style.display = "none";
        if (globalBar) globalBar.style.display = "none";
      }, 2000);
    } finally {
      btnSubmitUpload.disabled = false;
      btnSubmitUpload.innerHTML = `Upload &amp; Ingest`;
      btnCloseUpload.disabled = false;
    }
  });
}

// Escapes HTML special characters in text.
function escapeHtml(text) {
  if (!text) return "";
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

// Initializes the document viewer controls.
function initDocViewer() {
  const btnToggle = document.getElementById("btnToggleDocViewer");
  const btnClose = document.getElementById("btnCloseViewer");
  const btnPrev = document.getElementById("btnPrevPage");
  const btnNext = document.getElementById("btnNextPage");
  const inputJump = document.getElementById("inputPageJump");
  const btnZoomIn = document.getElementById("btnZoomIn");
  const btnZoomOut = document.getElementById("btnZoomOut");
  const btnResetZoom = document.getElementById("btnResetZoom");
  const tabPages = document.getElementById("tabPagesMode");
  const tabPdf = document.getElementById("tabPdfMode");

  if (btnToggle) {
    btnToggle.addEventListener("click", () => toggleDocViewer());
  }

  if (btnClose) {
    btnClose.addEventListener("click", () => toggleDocViewer(false));
  }

  if (btnPrev) {
    btnPrev.addEventListener("click", () => goToPage(currentPage - 1));
  }

  if (btnNext) {
    btnNext.addEventListener("click", () => goToPage(currentPage + 1));
  }

  if (inputJump) {
    inputJump.addEventListener("change", (e) => {
      const target = parseInt(e.target.value, 10);
      if (!isNaN(target)) goToPage(target);
    });
    inputJump.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        const target = parseInt(e.target.value, 10);
        if (!isNaN(target)) goToPage(target);
      }
    });
  }

  if (btnZoomIn) {
    btnZoomIn.addEventListener("click", () => setZoom(currentZoom + 0.15));
  }

  if (btnZoomOut) {
    btnZoomOut.addEventListener("click", () => setZoom(currentZoom - 0.15));
  }

  if (btnResetZoom) {
    btnResetZoom.addEventListener("click", () => setZoom(1.0));
  }

  if (tabPages) {
    tabPages.addEventListener("click", () => setViewerMode("pages"));
  }

  if (tabPdf) {
    tabPdf.addEventListener("click", () => setViewerMode("pdf"));
  }

  const btnFallbackPdf = document.getElementById("btnFallbackPdfView");
  if (btnFallbackPdf) {
    btnFallbackPdf.addEventListener("click", () => setViewerMode("pdf"));
  }
}

// Toggles the document viewer panel.
async function toggleDocViewer(forceState) {
  const column = document.getElementById("docViewerColumn");
  const btn = document.getElementById("btnToggleDocViewer");
  const btnText = document.getElementById("btnToggleDocViewerText");
  const container = document.querySelector(".working-sheet-container");

  const newState = forceState !== undefined ? forceState : !isDocViewerOpen;
  isDocViewerOpen = newState;

  if (isDocViewerOpen) {
    column.style.display = "flex";
    btn.classList.add("active");
    if (btnText) btnText.textContent = "Close Document Viewer";
    container.classList.add("split-mode");
    await loadDocumentPages(currentDocId);
  } else {
    column.style.display = "none";
    btn.classList.remove("active");
    if (btnText) btnText.textContent = "Open Document Beside Chat";
    container.classList.remove("split-mode");
  }
}

// Loads page images and metadata for the viewer.
async function loadDocumentPages(docId) {
  currentViewerDoc = docId;
  const titleEl = document.getElementById("viewerDocTitle");
  if (titleEl) titleEl.textContent = `${docId}.pdf`;

  try {
    const res = await fetch(`/api/documents/${docId}/pages?user_id=${encodeURIComponent(currentUserId)}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    totalPages = data.total_pages || 1;
    docPagesList = data.pages || [];

    const inputJump = document.getElementById("inputPageJump");
    const lblTotal = document.getElementById("lblTotalPages");
    if (inputJump) inputJump.max = totalPages;
    if (lblTotal) lblTotal.textContent = `/ ${totalPages}`;

    const pdfIframe = document.getElementById("pdfIframe");
    if (pdfIframe) {
      pdfIframe.src = data.pdf_url || `/api/documents/${docId}/pdf?user_id=${encodeURIComponent(currentUserId)}`;
    }

    renderThumbnails();
    goToPage(currentPage > totalPages ? 1 : currentPage);
  } catch (err) {
    console.error("Failed to load document pages:", err);
  }
}

// Navigates the viewer to a specific page.
function goToPage(pageNum) {
  if (totalPages <= 0) return;
  currentPage = Math.max(1, Math.min(pageNum, totalPages));

  const imgEl = document.getElementById("viewerPageImg");
  const fallback = document.getElementById("viewerFallbackCard");

  if (imgEl && docPagesList.length >= currentPage) {
    const pageItem = docPagesList[currentPage - 1];

    imgEl.style.display = "none";
    if (fallback) fallback.style.display = "none";

    let triedStatic = false;

    imgEl.onload = () => {
      if (fallback) fallback.style.display = "none";
      imgEl.style.display = "block";
    };
    imgEl.onerror = () => {
      console.warn(`[Viewer] Failed loading page preview: ${imgEl.src}`);

      if (!triedStatic && pageItem.static_url) {
        triedStatic = true;
        imgEl.src = pageItem.static_url;
        return;
      }
      imgEl.style.display = "none";
      if (fallback) fallback.style.display = "block";
    };

    imgEl.src = pageItem.image_url;
  }

  const badge = document.getElementById("viewerPageBadge");
  if (badge) badge.textContent = `Page ${currentPage} / ${totalPages}`;

  const inputJump = document.getElementById("inputPageJump");
  if (inputJump) inputJump.value = currentPage;

  const btnPrev = document.getElementById("btnPrevPage");
  const btnNext = document.getElementById("btnNextPage");
  if (btnPrev) btnPrev.disabled = currentPage <= 1;
  if (btnNext) btnNext.disabled = currentPage >= totalPages;

  document.querySelectorAll(".thumb-chip").forEach((chip, idx) => {
    chip.classList.toggle("active", idx + 1 === currentPage);
  });
}

// Opens the viewer and jumps to a page.
async function openAndJumpToPage(pageNum) {
  if (!isDocViewerOpen) {
    currentPage = pageNum;
    await toggleDocViewer(true);
  }
  setViewerMode("pages");
  goToPage(pageNum);

  const container = document.getElementById("pageViewContainer");
  if (container) container.scrollTop = 0;
}
window.openAndJumpToPage = openAndJumpToPage;

// Sets the viewer zoom level.
function setZoom(level) {
  currentZoom = Math.max(0.5, Math.min(2.5, Math.round(level * 100) / 100));
  const wrapper = document.getElementById("pageImageWrapper");
  if (wrapper) {
    wrapper.style.transform = `scale(${currentZoom})`;
  }
  const badge = document.getElementById("lblZoomLevel");
  if (badge) {
    badge.textContent = `${Math.round(currentZoom * 100)}%`;
  }
}

// Switches the viewer between page and PDF modes.
function setViewerMode(mode) {
  viewerMode = mode;
  const tabPages = document.getElementById("tabPagesMode");
  const tabPdf = document.getElementById("tabPdfMode");
  const pagesContainer = document.getElementById("pageViewContainer");
  const pagesToolbar = document.getElementById("pagesToolbar");
  const thumbStrip = document.getElementById("thumbnailStrip");
  const pdfContainer = document.getElementById("pdfViewContainer");

  if (mode === "pages") {
    tabPages.classList.add("active");
    tabPdf.classList.remove("active");
    pagesContainer.style.display = "flex";
    pagesToolbar.style.display = "flex";
    thumbStrip.style.display = "flex";
    pdfContainer.style.display = "none";
  } else {
    tabPages.classList.remove("active");
    tabPdf.classList.add("active");
    pagesContainer.style.display = "none";
    pagesToolbar.style.display = "none";
    thumbStrip.style.display = "none";
    pdfContainer.style.display = "flex";
  }
}

// Renders thumbnail chips for document pages.
function renderThumbnails() {
  const strip = document.getElementById("thumbnailStrip");
  if (!strip) return;
  strip.innerHTML = "";

  for (let i = 1; i <= totalPages; i++) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `thumb-chip ${i === currentPage ? "active" : ""}`;
    chip.textContent = `P.${i}`;
    chip.title = `Page ${i}`;
    chip.addEventListener("click", () => goToPage(i));
    strip.appendChild(chip);
  }
}