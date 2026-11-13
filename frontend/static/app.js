/**
 * Document Research Assistant
 * Handles document switching, side-by-side inspection, streaming Q&A, and uploads.
 */

const savedActiveDoc = typeof localStorage !== "undefined" ? localStorage.getItem("rag_active_doc_id") : null;
let currentDocId = savedActiveDoc || null;
let allDocuments = [];
const chatHistories = {}; // doc_id -> in-memory HTML snapshot

// Document Viewer State
let isDocViewerOpen = false;
let currentViewerDoc = currentDocId;
let currentPage = 1;
let totalPages = 1;
let docPagesList = [];
let currentZoom = 1.0;
let viewerMode = "pages"; // "pages" or "pdf"

document.addEventListener("DOMContentLoaded", () => {
  initApiKeyModal();
  initDocumentLedger();
  initChatForm();
  initClearChatButton();
  initUploadModal();
  initDocViewer();

  // Re-sweep any initial answer bodies with KaTeX
  if (window.renderMathInElement) {
    document.querySelectorAll(".answer-body").forEach((el) => {
      try {
        window.renderMathInElement(el, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "\\[", right: "\\]", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false },
          ],
          throwOnError: false,
        });
      } catch (e) {}
    });
  }
});

// ============================================================================
// 1. Document Ledger & Switching
// ============================================================================
async function initDocumentLedger() {
  await loadDocuments();
  // Restore conversational history from SQLite checkpointer for active document
  await loadDocumentHistory(currentDocId);
}

async function loadDocuments() {
  const docList = document.getElementById("docList");
  try {
    const res = await fetch("/api/documents");
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

    // Verify currentDocId exists among docs, otherwise fallback to first
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

/**
 * Load and restore conversation history from SQLite checkpointer.
 */
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
    const res = await fetch(`/api/documents/${docId}/history`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    chatStream.innerHTML = "";
    if (data.messages && data.messages.length > 0) {
      data.messages.forEach((msg) => {
        if (msg.role === "user") {
          appendUserMessage(msg.content);
        } else if (msg.role === "assistant") {
          appendAssistantMessage(
            { answer: msg.content, sources: msg.sources || [] },
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

function getDocDisplayName(docId) {
  if (!docId) return "";
  const found = allDocuments.find((d) => d.doc_id.toLowerCase() === docId.toLowerCase());
  if (found && found.title) return found.title;
  return docId.replace(/[_-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function updateInputPlaceholder() {
  const input = document.getElementById("questionInput");
  if (!input) return;
  if (currentDocId) {
    input.placeholder = `Ask a question about ${getDocDisplayName(currentDocId)}...`;
  } else {
    input.placeholder = "Upload or select a document to start asking questions...";
  }
}

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

  if (chatHistories[docId] && !forceReload) {
    chatStream.innerHTML = chatHistories[docId];
    chatStream.scrollTop = chatStream.scrollHeight;
  } else {
    await loadDocumentHistory(docId);
  }

  updateDocListHighlight();

  // If document viewer is currently open beside chat, update its content
  if (docId) {
    if (isDocViewerOpen) {
      loadDocumentPages(docId);
    } else {
      currentViewerDoc = docId;
      currentPage = 1;
    }
  }
}

function updateDocListHighlight() {
  document.querySelectorAll(".doc-item-scrap").forEach((el) => {
    const docName = el.querySelector(".doc-name")?.textContent.trim().toLowerCase();
    const isTarget = currentDocId && (docName === currentDocId.toLowerCase() || docName?.startsWith(currentDocId.toLowerCase()));
    el.classList.toggle("active", Boolean(isTarget));
    el.setAttribute("aria-selected", isTarget ? "true" : "false");
  });
}

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
      const res = await fetch(`/api/documents/${currentDocId}/history`, { method: "DELETE" });
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

function updateActiveBadge() {
  const badge = document.getElementById("activeDocBadge");
  if (badge) {
    badge.textContent = currentDocId ? `DOC: ${currentDocId.toUpperCase()}` : "";
    badge.style.display = currentDocId ? "inline-block" : "none";
  }
}

async function handleDeleteDoc(event, docId) {
  event.stopPropagation();
  const displayName = getDocDisplayName(docId);
  const confirmed = confirm(`Are you sure you want to delete "${displayName}"?`);
  if (!confirmed) return;

  try {
    const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
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

// ============================================================================
// 2. Chat Q&A Interaction
// ============================================================================
function initChatForm() {
  const chatForm = document.getElementById("chatForm");
  const questionInput = document.getElementById("questionInput");
  const btnSubmit = document.getElementById("btnSubmit");

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentDocId) {
      alert("Please upload or select a document from the left sidebar before asking questions.");
      return;
    }
    const query = questionInput.value.trim();
    if (!query) return;

    questionInput.value = "";
    btnSubmit.disabled = true;

    appendUserMessage(query);
    await handleStreamingChat(query);
    btnSubmit.disabled = false;
    questionInput.focus();
  });
}

/**
 * Real-time SSE Token Streaming & Status Pipeline Handler
 */
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
    const headers = { "Content-Type": "application/json" };
    if (userGroqKey) {
      headers["X-Groq-Api-Key"] = userGroqKey;
    }

    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers,
      body: JSON.stringify({
        question: query,
        doc_id: currentDocId,
        thread_id: `doc_${currentDocId}`,
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
      workingSheet.appendChild(renderSourcesBar(pages));
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

function getRelevantVisualSources(sources = [], answerText = "") {
  if (!Array.isArray(sources) || sources.length === 0) return [];

  const candidates = sources.filter(
    (s) => s.web_image_url || s.image_path || s.table_markdown
  );
  if (candidates.length === 0) return [];

  // Deduplicate by unique asset ID or URL
  const seen = new Set();
  const deduped = [];
  for (const c of candidates) {
    const key = c.id || c.web_image_url || c.image_path;
    if (key && !seen.has(key)) {
      seen.add(key);
      deduped.push(c);
    }
  }

  // Strictly cap at at most 2 relevant visual assets to avoid visual clutter
  return deduped.slice(0, 2);
}

let activeStreamingTimer = null;

function appendAssistantMessage(data, options = {}) {
  if (!options.immediate) {
    return streamAssistantResponse(data);
  }

  // Immediate rendering path (e.g. for restored history)
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
    workingSheet.appendChild(renderSourcesBar(pages));
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

/**
 * Progressive Typewriter Streaming Effect
 * Streams tokens word-by-word with a blinking cursor before settling into rich markdown.
 */
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

  // 1. Clean answer text and extract cited pages
  const { cleanText, pages } = cleanAnswerTextAndExtractPages(data.answer || "", data.sources || []);

  const visualSources = getRelevantVisualSources(data.sources, data.answer);

  let isFinalized = false;

  const finalize = () => {
    if (isFinalized) return;
    isFinalized = true;

    if (activeStreamingTimer) {
      clearInterval(activeStreamingTimer);
      activeStreamingTimer = null;
    }

    workingSheet.style.cursor = "default";

    // 1. Render complete rich markdown with KaTeX
    answerBody.innerHTML = parseMarkdownToHtml(cleanText);
    renderMathInElementSafely(answerBody);

    // 2. Attach unified end-of-answer sources provenance bar: Sources: Page 10, 12, 15...
    if (pages.length > 0) {
      workingSheet.appendChild(renderSourcesBar(pages));
    }

    // 3. Attach visual references deck
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

  // Allow clicking anywhere on the card to instantly skip animation
  workingSheet.addEventListener("click", finalize, { once: true });

  // Tokenize text into words and spaces for progressive streaming
  const tokens = cleanText.split(/(\s+)/);
  if (tokens.length <= 1) {
    finalize();
    return;
  }

  let tokenIdx = 0;
  let displayedText = "";

  // Dynamic pacing: ~15-20ms per tick, batch tokens if response is long
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

/**
 * Robust Citation Extractor & Body Cleaner
 * Strips all inline bracket tags [Source N, Page P] and extracts unique pages.
 */
function cleanAnswerTextAndExtractPages(rawText, sources = []) {
  if (!rawText) return { cleanText: "", pages: [] };

  const citedPages = new Set();
  const mathPlaceholders = [];
  let text = rawText;

  // Stash math blocks and inlines so regexes don't touch them
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
    const idx = mathPlaceholders.length;
    mathPlaceholders.push({ type: "inline", code: eq.trim() });
    return `___MATH_INLINE_${idx}___`;
  });
  text = text.replace(/\\\(([\s\S]+?)\\\)/g, (m, eq) => {
    const idx = mathPlaceholders.length;
    mathPlaceholders.push({ type: "inline", code: eq.trim() });
    return `___MATH_INLINE_${idx}___`;
  });

  // Extract page numbers from compound or single citations e.g. [Source 1, Page 1; Source 2, Page 10], [Page 4]
  text = text.replace(/\[([^\]]*?Page[s]?\s*\d+[^\]]*?)\]/gi, (match) => {
    const pageMatches = [...match.matchAll(/Page[s]?\s*(\d+)/gi)];
    pageMatches.forEach((m) => {
      const num = parseInt(m[1], 10);
      if (!isNaN(num)) citedPages.add(num);
    });
    return "";
  });

  // Extract from [Source N] using sources metadata
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

  // Strip standalone brackets like [1, 2] or [Source 1]
  text = text.replace(/\[(?:Source\s*)?\d+(?:\s*,\s*\d+)*\]/gi, "");

  // Clean trailing spaces before punctuation: " (£) ." -> " (£)."
  text = text.replace(/\s+([.,;:!?])/g, "$1");
  // Normalize whitespace
  text = text.replace(/[ \t]{2,}/g, " ");

  // Fallback to sources metadata if no pages cited in text
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

  // Restore math placeholders
  text = text.replace(/___MATH_BLOCK_(\d+)___/g, (m, idx) => `$$${mathPlaceholders[idx].code}$$`);
  text = text.replace(/___MATH_INLINE_(\d+)___/g, (m, idx) => `$${mathPlaceholders[idx].code}$`);

  const sortedPages = [...citedPages].sort((a, b) => a - b);
  return { cleanText: text.trim(), pages: sortedPages };
}

/**
 * Rich Semantic Markdown & LaTeX Math Parser
 * Converts **bold**, headings, lists, inline code, and formulas into structured semantic HTML.
 */
function parseMarkdownToHtml(rawText) {
  if (!rawText) return "";

  const mathTokens = [];
  let s = rawText;

  // Defensive pre-clean: Repair any legacy or corrupted math token remnants from previous turns
  s = s.replace(/_{1,3}MATH_TOKEN_\d+_{1,3}/g, "$");

  // Stash display/block math ($$ ... $$ and \[ ... \])
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

  // Stash LaTeX parentheses \( ... \)
  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (m, eq) => {
    const idx = mathTokens.length;
    mathTokens.push({ type: "inline", code: eq.trim() });
    return `@@@MATH_TOKEN_${idx}@@@`;
  });

  // Stash inline math ($ ... $) with strict currency disambiguation
  s = s.replace(/(?<!\$)\$([^\$\n]+?)\$(?!\$)/g, (fullMatch, content) => {
    const trimmed = content.trim();
    const words = trimmed.split(/\s+/);
    // If the content contains multiple English words without LaTeX commands or operators,
    // it represents separate currency values (e.g. "$1,773 million and $2,938 million"), not a formula
    const hasMathSymbols = /[\\=+^_{}[\]<>*/~|]/.test(trimmed) || /^[a-zA-Z]$/.test(trimmed);
    if (!hasMathSymbols && words.length > 2) {
      return fullMatch;
    }
    const idx = mathTokens.length;
    mathTokens.push({ type: "inline", code: trimmed });
    return `@@@MATH_TOKEN_${idx}@@@`;
  });

  // Pre-process inline sub-bullets like ": * **" or ". * **" -> newline with indent
  s = s.replace(/([.:;])\s*[\*\-]\s+(?=\*\*|[a-zA-Z])/g, "$1\n   * ");

  // Escape HTML
  s = escapeHtml(s);

  // Fenced Code Blocks (```lang ... ```)
  s = s.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]+?)```/g, (m, lang, code) => {
    return `<pre class="md-code-block"><code class="language-${lang}">${code.trim()}</code></pre>`;
  });

  // Inline Code (`code`)
  s = s.replace(/`([^`\n]+)`/g, '<code class="md-code">$1</code>');

  // Headings
  s = s.replace(/^####\s+(.+)$/gm, '<h5 class="md-subheading">$1</h5>');
  s = s.replace(/^###\s+(.+)$/gm, '<h4 class="md-subheading">$1</h4>');
  s = s.replace(/^##\s+(.+)$/gm, '<h3 class="md-heading">$1</h3>');
  s = s.replace(/^#\s+(.+)$/gm, '<h2 class="md-title">$1</h2>');

  // Bold (**text** or __text__)
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/__(.+?)__/g, '<strong>$1</strong>');

  // Italic (*text* or _text_)
  s = s.replace(/(?<!\*)\*([^\*\n]+?)\*(?!\*)/g, '<em>$1</em>');
  s = s.replace(/(^|[\s(])_([^\s_][^_]*?[^\s_]|[^_\s])_([)\s.,;:!?]|$)/g, '$1<em>$2</em>$3');

  // Lists
  // Sub-bullets (indented with 2+ spaces: "   * item")
  s = s.replace(/^[ \t]{2,}[\*\-]\s+(.+)$/gm, '<li class="md-sublist-item">$1</li>');
  // Top-level bullets ("* item" or "- item")
  s = s.replace(/^[ \t]*[\*\-]\s+(.+)$/gm, '<li class="md-list-item">$1</li>');
  // Numbered items ("1. item")
  s = s.replace(/^[ \t]*(\d+)\.\s+(.+)$/gm, '<li class="md-ordered-item" data-num="$1">$2</li>');

  // Wrap lists
  s = s.replace(/((?:<li class="md-sublist-item">[\s\S]+?<\/li>\s*)+)/g, '<ul class="md-sublist">$1</ul>');
  s = s.replace(/((?:<li class="md-list-item">[\s\S]+?<\/li>\s*)+)/g, '<ul class="md-list">$1</ul>');
  s = s.replace(/((?:<li class="md-ordered-item"[^>]*>[\s\S]+?<\/li>\s*)+)/g, '<ol class="md-ordered-list">$1</ol>');

  s = s.replace(/<\/ul>\s*<ul class="md-list">/g, '');
  s = s.replace(/<\/ol>\s*<ol class="md-ordered-list">/g, '');

  // Paragraph breaks
  s = s.replace(/\n{2,}/g, '<br><br>');
  // Single newlines
  s = s.replace(/([^>])\n([^<])/g, '$1<br>$2');

  // Render & Re-inject math expressions (using safe @@@ tokens unaffected by markdown bold/italic)
  s = s.replace(/@@@MATH_TOKEN_(\d+)@@@/g, (match, idxStr) => {
    const token = mathTokens[parseInt(idxStr, 10)];
    if (!token) return match;

    // Bare numeric values like "$1,773$" or "$1.70$" restore as clean currency amounts
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

  return s;
}

/**
 * End-of-Answer Sources Bar: "Sources: Page 10, 12, 15..."
 * Formatted with interactive jump pills for each referenced page.
 */
function renderSourcesBar(pages) {
  const bar = document.createElement("div");
  bar.className = "sources-provenance-bar";

  const label = document.createElement("span");
  label.className = "sources-label";
  label.textContent = "Sources:";
  bar.appendChild(label);

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
  return bar;
}

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

function renderMathInElementSafely(element) {
  if (!element || !window.renderMathInElement) return;
  try {
    window.renderMathInElement(element, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  } catch (err) {
    console.warn("KaTeX post-render notice:", err);
  }
}

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

function renderTapedMount(src) {
  const isTable = (src.type || "").toLowerCase() === "table" || Boolean(src.table_markdown);
  const pages = (src.pages || []).join(", ") || "?";
  const badgeText = isTable ? `TABLE [PAGE ${pages}]` : `FIGURE [PAGE ${pages}]`;
  const caption = src.caption || src.panel_caption || "Visual asset extracted from document";
  const targetPage = (src.pages && src.pages.length > 0 && parseInt(src.pages[0], 10)) || 1;

  // Resolve image URL with resilient fallback
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

// ============================================================================
// 3. Upload Document Modal & Ingestion Progress Bar
// ============================================================================
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

function startIngestionProgress(filename) {
  updateIngestionUI(2, "Uploading PDF & initializing workspace...", filename);
}

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

// ============================================================================
// API Key Management (BYOK - Bring Your Own Groq Key)
// ============================================================================
let isApiKeyModalOpen = false;

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

function closeApiKeyModal() {
  const modal = document.getElementById("apiKeyModal");
  if (modal) {
    modal.style.display = "none";
    isApiKeyModalOpen = false;
  }
}

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

  // Check if backend has a server-side default key
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

  const showUploadError = (msg) => {
    if (errorNotice) {
      errorNotice.textContent = msg;
      errorNotice.style.display = "block";
    }
  };

  const clearUploadError = () => {
    if (errorNotice) {
      errorNotice.style.display = "none";
      errorNotice.textContent = "";
    }
  };

  const syncUploadLimits = async () => {
    try {
      const res = await fetch("/api/auth/status");
      if (res.ok) {
        const data = await res.json();
        if (typeof data.max_upload_mb === "number") maxUploadMb = data.max_upload_mb;
        if (typeof data.max_upload_pages === "number") maxUploadPages = data.max_upload_pages;
        if (limitHint) {
          limitHint.textContent = `Upload a PDF document (max ${maxUploadMb} MB, ${maxUploadPages} pages) to analyze its text, figures, and tables.`;
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
      const errMsg = `PDF exceeds the ${maxUploadMb} MB limit (${sizeMb} MB selected). Please choose a smaller PDF.`;
      showUploadError(errMsg);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    startIngestionProgress(file.name);
    btnSubmitUpload.disabled = true;
    btnSubmitUpload.innerHTML = `<span>Ingesting...</span> <span aria-hidden="true">⏳</span>`;
    btnCloseUpload.disabled = true;

    try {
      // Launch asynchronous background ingestion job (avoids Cloudflare/Render 100s timeouts)
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed: ${res.status}`);
      }

      const initData = await res.json();
      const jobId = initData.job_id;
      let finalDocId = initData.doc_id || null;
      let finalMessage = "";
      let colpaliExisted = false;
      let alreadyExisted = false;

      if (!jobId) {
        throw new Error("Server did not return an ingestion job ID.");
      }

      updateIngestionUI(5, `Uploaded ${file.name}. Starting pipeline...`, file.name);

      let transientErrors = 0;
      while (true) {
        await new Promise((resolve) => setTimeout(resolve, 900));

        let statusRes;
        try {
          statusRes = await fetch(`/api/upload/status/${encodeURIComponent(jobId)}`);
        } catch (netErr) {
          transientErrors += 1;
          if (transientErrors >= 15) {
            throw new Error(`Network connection lost while checking ingestion status: ${netErr.message}`);
          }
          continue;
        }

        if (statusRes.status === 404) {
          const errJson = await statusRes.json().catch(() => ({}));
          throw new Error(errJson.detail || "Ingestion job expired or server restarted (memory limit reached).");
        }

        if (!statusRes.ok) {
          transientErrors += 1;
          if (transientErrors >= 15) {
            if (statusRes.status === 502 || statusRes.status === 503 || statusRes.status === 504) {
              throw new Error(`Server temporarily unreachable (HTTP ${statusRes.status} Bad Gateway). The container may have restarted.`);
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
          finalDocId = job.doc_id || finalDocId;
          finalMessage = job.message || `Successfully ingested ${file.name}!`;
          colpaliExisted = Boolean(job.colpali_existed);
          alreadyExisted = Boolean(job.already_existed);
          break;
        }
      }

      completeIngestionProgress(true, finalMessage || `Successfully ingested ${file.name}`);

      // Allow extra time for user to read notice when document/pages already existed
      const delayMs = alreadyExisted ? 1800 : 750;
      await new Promise((resolve) => setTimeout(resolve, delayMs));

      modal.classList.remove("open");
      uploadForm.reset();

      // Refresh documents and switch to newly ingested document
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

// ============================================================================
// Utility Helpers
// ============================================================================
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

// ============================================================================
// 4. Side-by-Side Document Inspection Panel
// ============================================================================
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

async function loadDocumentPages(docId) {
  currentViewerDoc = docId;
  const titleEl = document.getElementById("viewerDocTitle");
  if (titleEl) titleEl.textContent = `${docId}.pdf`;

  try {
    const res = await fetch(`/api/documents/${docId}/pages`);
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
      pdfIframe.src = data.pdf_url || `/api/documents/${docId}/pdf`;
    }

    renderThumbnails();
    goToPage(currentPage > totalPages ? 1 : currentPage);
  } catch (err) {
    console.error("Failed to load document pages:", err);
  }
}

function goToPage(pageNum) {
  if (totalPages <= 0) return;
  currentPage = Math.max(1, Math.min(pageNum, totalPages));

  const imgEl = document.getElementById("viewerPageImg");
  const fallback = document.getElementById("viewerFallbackCard");

  if (imgEl && docPagesList.length >= currentPage) {
    const pageItem = docPagesList[currentPage - 1];
    imgEl.onload = () => {
      if (fallback) fallback.style.display = "none";
      imgEl.style.display = "block";
    };
    imgEl.onerror = () => {
      console.warn(`[Viewer] Failed loading page preview: ${imgEl.src}`);
      imgEl.style.display = "none";
      if (fallback) fallback.style.display = "block";
    };

    if (fallback) fallback.style.display = "none";
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

async function openAndJumpToPage(pageNum) {
  if (!isDocViewerOpen) {
    await toggleDocViewer(true);
  }
  setViewerMode("pages");
  goToPage(pageNum);

  const container = document.getElementById("pageViewContainer");
  if (container) container.scrollTop = 0;
}
window.openAndJumpToPage = openAndJumpToPage;

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
