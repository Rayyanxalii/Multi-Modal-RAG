/**
 * Document Research Assistant
 * Handles document switching, side-by-side inspection, streaming Q&A, and uploads.
 */

const savedActiveDoc = typeof localStorage !== "undefined" ? localStorage.getItem("rag_active_doc_id") : null;
let currentDocId = savedActiveDoc || "attention";
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

    docList.innerHTML = "";
    if (!docs || docs.length === 0) {
      docList.innerHTML = `<li style="font-size: 13px; color: var(--dp-color-ink-muted); padding: 8px 4px;">No documents yet.</li>`;
      return;
    }

    // Verify currentDocId exists among docs, otherwise fallback to first
    const docExists = docs.some((d) => d.doc_id.toLowerCase() === currentDocId.toLowerCase());
    if (!docExists && docs.length > 0) {
      currentDocId = docs[0].doc_id;
      if (typeof localStorage !== "undefined") localStorage.setItem("rag_active_doc_id", currentDocId);
    }

    docs.forEach((doc) => {
      const li = document.createElement("li");
      const isSelected = doc.doc_id.toLowerCase() === currentDocId.toLowerCase();
      li.className = `doc-item-scrap ${isSelected ? "active" : ""}`;
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", isSelected ? "true" : "false");

      li.innerHTML = `
        <div class="doc-info" onclick="switchDocument('${doc.doc_id}')">
          <div class="doc-name">${escapeHtml(doc.title || doc.doc_id)}</div>
          <div class="doc-meta">${doc.pages || 0} pages</div>
        </div>
        ${doc.doc_id !== "attention" ? `
          <button class="btn-delete-scrap" title="Delete document" onclick="handleDeleteDoc(event, '${doc.doc_id}')" aria-label="Delete ${doc.doc_id}">
            ✕
          </button>
        ` : `
          <span style="font-size: 12px; color: var(--dp-color-accent-ink); opacity: 0.85; font-weight: 500;">Default</span>
        `}
      `;
      docList.appendChild(li);
    });

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

  chatStream.innerHTML = `
    <article class="message-entry assistant" id="historyLoader">
      <div class="assistant-working-sheet">
        <div class="answer-body" style="color: var(--dp-color-secondary); font-style: italic;">
          Loading conversation history for <strong>${escapeHtml(docId)}.pdf</strong>...
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

async function switchDocument(docId, forceReload = false) {
  if (docId === currentDocId && !forceReload) return;

  const chatStream = document.getElementById("chatStream");
  if (currentDocId && chatStream) {
    chatHistories[currentDocId] = chatStream.innerHTML;
  }

  currentDocId = docId;
  if (typeof localStorage !== "undefined") localStorage.setItem("rag_active_doc_id", docId);
  updateActiveBadge();

  if (chatHistories[docId] && !forceReload) {
    chatStream.innerHTML = chatHistories[docId];
    chatStream.scrollTop = chatStream.scrollHeight;
  } else {
    await loadDocumentHistory(docId);
  }

  updateDocListHighlight();

  // If document viewer is currently open beside chat, update its content
  if (isDocViewerOpen) {
    loadDocumentPages(docId);
  } else {
    currentViewerDoc = docId;
    currentPage = 1;
  }
}

function updateDocListHighlight() {
  document.querySelectorAll(".doc-item-scrap").forEach((el) => {
    const docName = el.querySelector(".doc-name")?.textContent.trim().toLowerCase();
    const isTarget = docName === currentDocId.toLowerCase() || docName?.startsWith(currentDocId.toLowerCase());
    el.classList.toggle("active", isTarget);
    el.setAttribute("aria-selected", isTarget ? "true" : "false");
  });
}

function initClearChatButton() {
  const btn = document.getElementById("btnClearChat");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    const confirmed = confirm(
      `Clear chat history for "${currentDocId}.pdf"?\n\nThis will permanently delete all saved messages for this document from the database.`
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
  chatStream.innerHTML = `
    <article class="message-entry assistant">
      <div class="assistant-working-sheet">
        <div class="answer-body">
          Ask any question about <strong>${escapeHtml(docId)}.pdf</strong>'s text, equations, figures, or tables.
        </div>
      </div>
    </article>
  `;
  chatHistories[docId] = chatStream.innerHTML;
}

function updateActiveBadge() {
  const badge = document.getElementById("activeDocBadge");
  if (badge) badge.textContent = `DOC: ${currentDocId.toUpperCase()}`;
}

async function handleDeleteDoc(event, docId) {
  event.stopPropagation();
  const confirmed = confirm(`Are you sure you want to delete "${docId}"?`);
  if (!confirmed) return;

  try {
    const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
    const data = await res.json();
    if (res.ok) {
      delete chatHistories[docId];
      if (currentDocId === docId) {
        if (typeof localStorage !== "undefined") localStorage.setItem("rag_active_doc_id", "attention");
        await switchDocument("attention", true);
      }
      await loadDocuments();
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
    const query = questionInput.value.trim();
    if (!query) return;

    questionInput.value = "";
    btnSubmit.disabled = true;

    appendUserMessage(query);
    const thinkingEl = appendThinkingIndicator();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: query,
          doc_id: currentDocId,
          thread_id: `doc_${currentDocId}`,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${res.status}`);
      }

      const result = await res.json();
      thinkingEl.remove();
      appendAssistantMessage(result);
    } catch (err) {
      thinkingEl.remove();
      appendErrorMessage(err.message || String(err));
    } finally {
      btnSubmit.disabled = false;
      questionInput.focus();
    }
  });
}

function appendUserMessage(text) {
  const stream = document.getElementById("chatStream");
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

  // Stash math expressions
  s = s.replace(/\$\$([\s\S]+?)\$\$/g, (m, eq) => {
    const idx = mathTokens.length;
    mathTokens.push({ type: "block", code: eq.trim() });
    return `___MATH_TOKEN_${idx}___`;
  });
  s = s.replace(/\\\[([\s\S]+?)\\\]/g, (m, eq) => {
    const idx = mathTokens.length;
    mathTokens.push({ type: "block", code: eq.trim() });
    return `___MATH_TOKEN_${idx}___`;
  });
  s = s.replace(/(?<!\$)\$([^\$\n]+?)\$(?!\$)/g, (m, eq) => {
    const idx = mathTokens.length;
    mathTokens.push({ type: "inline", code: eq.trim() });
    return `___MATH_TOKEN_${idx}___`;
  });
  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (m, eq) => {
    const idx = mathTokens.length;
    mathTokens.push({ type: "inline", code: eq.trim() });
    return `___MATH_TOKEN_${idx}___`;
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

  // Render & Re-inject math expressions
  s = s.replace(/___MATH_TOKEN_(\d+)___/g, (match, idxStr) => {
    const token = mathTokens[parseInt(idxStr, 10)];
    if (!token) return match;

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
let ingestionProgressTimer = null;

function startIngestionProgress(filename) {
  const box = document.getElementById("uploadProgressBox");
  const stageText = document.getElementById("progressStageText");
  const percentText = document.getElementById("progressPercent");
  const fill = document.getElementById("progressBarFill");
  const globalBar = document.getElementById("globalIngestBar");
  const globalText = document.getElementById("globalIngestText");
  const globalPercent = document.getElementById("globalIngestPercent");

  if (box) box.style.display = "block";
  if (globalBar) globalBar.style.display = "flex";

  const startTime = Date.now();
  const stages = [
    { threshold: 18, text: "Uploading PDF & initializing workspace..." },
    { threshold: 38, text: "Rendering high-res document pages..." },
    { threshold: 62, text: "Parsing layout & extracting text (IBM Docling)..." },
    { threshold: 82, text: "Cropping figures & structured data tables..." },
    { threshold: 95, text: "Indexing semantic vectors & multimodal embeddings..." },
  ];

  const updateUI = (pct, text) => {
    const rounded = Math.min(Math.round(pct), 98);
    if (fill) fill.style.width = `${rounded}%`;
    if (percentText) percentText.textContent = `${rounded}%`;
    if (stageText) {
      stageText.innerHTML = `<span class="progress-spinner" aria-hidden="true">⏳</span><span>${escapeHtml(text)}</span>`;
    }
    if (globalText) {
      globalText.textContent = `INGESTING [${filename.toUpperCase()}]: ${text}`;
    }
    if (globalPercent) {
      globalPercent.textContent = `${rounded}%`;
    }
  };

  updateUI(2, stages[0].text);

  if (ingestionProgressTimer) clearInterval(ingestionProgressTimer);

  ingestionProgressTimer = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;

    let currentPercent = 0;
    if (elapsed < 3) {
      currentPercent = (elapsed / 3) * 25;
    } else if (elapsed < 10) {
      currentPercent = 25 + ((elapsed - 3) / 7) * 35;
    } else if (elapsed < 22) {
      currentPercent = 60 + ((elapsed - 10) / 12) * 25;
    } else {
      currentPercent = 85 + (1 - Math.exp(-(elapsed - 22) / 15)) * 11;
    }

    let activeStage = stages[stages.length - 1].text;
    for (const stage of stages) {
      if (currentPercent < stage.threshold) {
        activeStage = stage.text;
        break;
      }
    }

    updateUI(currentPercent, activeStage);
  }, 200);
}

function completeIngestionProgress(success = true, message = "") {
  if (ingestionProgressTimer) {
    clearInterval(ingestionProgressTimer);
    ingestionProgressTimer = null;
  }

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

function initUploadModal() {
  const modal = document.getElementById("uploadModal");
  const btnOpen = document.getElementById("btnOpenUpload");
  const btnClose = document.getElementById("btnCloseUpload");
  const uploadForm = document.getElementById("uploadForm");
  const fileInput = document.getElementById("pdfFileInput");
  const btnSubmitUpload = document.getElementById("btnSubmitUpload");

  btnOpen.addEventListener("click", () => {
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
    if (!fileInput.files || fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    startIngestionProgress(file.name);
    btnSubmitUpload.disabled = true;
    btnSubmitUpload.innerHTML = `<span>Ingesting...</span> <span aria-hidden="true">⏳</span>`;
    btnCloseUpload.disabled = true;

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed: ${res.status}`);
      }

      const result = await res.json();
      completeIngestionProgress(true, `Successfully ingested ${file.name}`);

      // Allow 750ms for user to see the 100% complete state
      await new Promise((resolve) => setTimeout(resolve, 750));

      modal.classList.remove("open");
      uploadForm.reset();

      // Refresh documents and switch to new document
      await loadDocuments();
      if (result.doc_id) {
        await switchDocument(result.doc_id, true);
      }
    } catch (err) {
      completeIngestionProgress(false, err.message || String(err));
      alert(`Error during ingestion: ${err.message || err}`);
    } finally {
      setTimeout(() => {
        const box = document.getElementById("uploadProgressBox");
        const globalBar = document.getElementById("globalIngestBar");
        if (box) box.style.display = "none";
        if (globalBar) globalBar.style.display = "none";
      }, 1500);

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
  if (imgEl && docPagesList.length >= currentPage) {
    imgEl.src = docPagesList[currentPage - 1].image_url;
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
