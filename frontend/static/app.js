/**
 * Multimodal Agentic RAG — Working Sheet Frontend
 * Handles document switching, taped visual reference mounts, streaming Q&A, and uploads.
 */

let currentDocId = "attention";
const chatHistories = {}; // doc_id -> list of DOM elements or HTML

// Document Viewer State
let isDocViewerOpen = false;
let currentViewerDoc = "attention";
let currentPage = 1;
let totalPages = 1;
let docPagesList = [];
let currentZoom = 1.0;
let viewerMode = "pages"; // "pages" or "pdf"

document.addEventListener("DOMContentLoaded", () => {
  initDocumentLedger();
  initChatForm();
  initUploadModal();
  initDocViewer();
});

// ============================================================================
// 1. Document Ledger & Switching
// ============================================================================
async function initDocumentLedger() {
  await loadDocuments();
}

async function loadDocuments() {
  const docList = document.getElementById("docList");
  try {
    const res = await fetch("/api/documents");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const docs = await res.json();

    docList.innerHTML = "";
    if (!docs || docs.length === 0) {
      docList.innerHTML = `<li style="font-size: 19px; color: var(--dp-color-ink-muted);">No documents yet.</li>`;
      return;
    }

    docs.forEach((doc) => {
      const li = document.createElement("li");
      li.className = `doc-item-scrap ${doc.doc_id === currentDocId ? "active" : ""}`;
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", doc.doc_id === currentDocId ? "true" : "false");

      li.innerHTML = `
        <div class="doc-info" onclick="switchDocument('${doc.doc_id}')">
          <div class="doc-name">${escapeHtml(doc.title || doc.doc_id)}</div>
          <div class="doc-meta">${doc.pages || 0} pages • ${doc.figures || 0} figs • ${doc.tables || 0} tbls</div>
        </div>
        ${doc.doc_id !== "attention" ? `
          <button class="btn-delete-scrap" title="Delete document & chat" onclick="handleDeleteDoc(event, '${doc.doc_id}')" aria-label="Delete ${doc.doc_id}">
            ✕
          </button>
        ` : `
          <span style="font-size: 16px; color: var(--dp-color-annotation); opacity: 0.7;">Default</span>
        `}
      `;
      docList.appendChild(li);
    });

    updateActiveBadge();
  } catch (err) {
    console.error("Failed to load documents:", err);
  }
}

function switchDocument(docId) {
  if (docId === currentDocId) return;

  // Save current chat stream HTML into cache
  const chatStream = document.getElementById("chatStream");
  chatHistories[currentDocId] = chatStream.innerHTML;

  currentDocId = docId;
  updateActiveBadge();

  // Restore or reset chat stream for new document
  if (chatHistories[docId]) {
    chatStream.innerHTML = chatHistories[docId];
  } else {
    chatStream.innerHTML = `
      <article class="message-entry assistant">
        <div class="assistant-working-sheet">
          <div class="assistant-header-meta">
            <span class="route-pill">ARCHIVE SESSION</span>
            <span class="reasoning-text">Active document set to "${escapeHtml(docId)}"</span>
          </div>
          <div class="answer-body">
            Switched session context to: <strong>${escapeHtml(docId)}</strong>.
            <br>
            All subsequent multimodal queries will isolate retrieval strictly to this paper's text passages, diagrams, and tables.
          </div>
        </div>
      </article>
    `;
  }

  // Refresh active styles in list
  document.querySelectorAll(".doc-item-scrap").forEach((el) => {
    const isTarget = el.querySelector(".doc-name")?.textContent.toLowerCase().includes(docId.toLowerCase());
    el.classList.toggle("active", isTarget);
  });

  chatStream.scrollTop = chatStream.scrollHeight;

  // If document viewer is currently open beside chat, update its content
  if (isDocViewerOpen) {
    loadDocumentPages(docId);
  } else {
    currentViewerDoc = docId;
    currentPage = 1;
  }
}

function updateActiveBadge() {
  const badge = document.getElementById("activeDocBadge");
  if (badge) badge.textContent = `DOC: ${currentDocId.toUpperCase()}`;
}

async function handleDeleteDoc(event, docId) {
  event.stopPropagation();
  const confirmed = confirm(`Are you sure you want to delete "${docId}"?\nThis will permanently purge all local figures, chunks, Qdrant vectors, and cloud embeddings.`);
  if (!confirmed) return;

  try {
    const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
    const data = await res.json();
    if (res.ok) {
      delete chatHistories[docId];
      if (currentDocId === docId) {
        switchDocument("attention");
      }
      await loadDocuments();
    } else {
      alert(`Deletion note: ${data.message || "Failed"}`);
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
      <div class="user-tag">INQUIRY / RESEARCHER</div>
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
    <div class="assistant-working-sheet" style="opacity: 0.85;">
      <div class="assistant-header-meta">
        <span class="route-pill">PROCESSING</span>
        <span class="reasoning-text">Analyzing query intent and executing multi-vector retrieval</span>
      </div>
      <div class="answer-body" style="color: var(--dp-color-secondary); font-style: italic;">
        Consulting text chunks, layout bounding boxes, and visual late-interaction vectors...
      </div>
    </div>
  `;
  stream.appendChild(entry);
  stream.scrollTop = stream.scrollHeight;
  return entry;
}

function appendAssistantMessage(data) {
  const stream = document.getElementById("chatStream");
  const entry = document.createElement("article");
  entry.className = "message-entry assistant";

  // Format citations [Source N, Page P] into interactive buttons that jump beside chat
  let formattedAnswer = escapeHtml(data.answer || "");
  formattedAnswer = formattedAnswer.replace(/\[Source\s+(\d+),\s*Page\s+(\d+)\]/gi, (match, srcNum, pageNum) => {
    return `<button type="button" class="citation-highlight-btn" onclick="openAndJumpToPage(${pageNum})" title="Inspect Page ${pageNum} in document view"><span>${match}</span><span style="font-size:11px;">→</span></button>`;
  });

  // Extract visual sources with images or table markdown
  const visualSources = (data.sources || []).filter(
    (s) => s.web_image_url || s.image_path || s.table_markdown
  );

  let visualDeckHtml = "";
  if (visualSources.length > 0) {
    visualDeckHtml = `
      <div class="visual-references-deck">
        <div class="deck-label">
          <span>Figure &amp; Table Evidence</span>
          <span style="font-family: var(--dp-typography-mono); font-size: 13px; color: var(--dp-color-secondary);">(${visualSources.length} items retrieved)</span>
        </div>
        <div class="taped-references-grid">
          ${visualSources.map((src) => renderTapedMount(src)).join("")}
        </div>
      </div>
    `;
  }

  const routeName = (data.route || "both").toUpperCase();
  const reasoning = data.routing_reasoning || "Multimodal context synthesis";

  entry.innerHTML = `
    <div class="assistant-working-sheet">
      <div class="assistant-header-meta">
        <span class="route-pill">ROUTE: ${routeName}</span>
        <span class="reasoning-text">${escapeHtml(reasoning)}</span>
      </div>
      <div class="answer-body">
        ${formattedAnswer}
      </div>
      ${visualDeckHtml}
    </div>
  `;

  stream.appendChild(entry);
  stream.scrollTop = stream.scrollHeight;

  // Update status badge
  const routeBadge = document.getElementById("routeBadge");
  if (routeBadge) routeBadge.textContent = `ROUTE: ${routeName}`;
}

function renderTapedMount(src) {
  const isTable = (src.type || "").toLowerCase() === "table" || !!src.table_markdown;
  const pages = (src.pages || []).join(", ") || "?";
  const badgeText = isTable ? `TABLE [PAGE ${pages}]` : `FIGURE [PAGE ${pages}]`;
  const caption = src.caption || src.panel_caption || "Visual asset extracted from document";
  const targetPage = (src.pages && src.pages.length > 0 && parseInt(src.pages[0], 10)) || 1;

  let bodyHtml = "";
  if (src.web_image_url) {
    bodyHtml = `<img class="mount-image" src="${escapeHtml(src.web_image_url)}" alt="${escapeHtml(caption)}" loading="lazy">`;
  } else if (src.table_markdown) {
    bodyHtml = `<pre class="table-markdown-box">${escapeHtml(src.table_markdown)}</pre>`;
  }

  return `
    <div class="taped-mount clickable-mount" onclick="openAndJumpToPage(${targetPage})" title="Click to view Page ${targetPage} beside chat">
      <div class="mount-badge">${escapeHtml(badgeText)}</div>
      ${bodyHtml}
      <div class="mount-caption">${escapeHtml(caption)}</div>
      <div class="mount-inspect-hint"><span>INSPECT PAGE ${targetPage} IN DOCUMENT →</span></div>
    </div>
  `;
}

function appendErrorMessage(errText) {
  const stream = document.getElementById("chatStream");
  const entry = document.createElement("article");
  entry.className = "message-entry assistant";
  entry.innerHTML = `
    <div class="assistant-working-sheet" style="border-top-color: var(--dp-color-ink);">
      <div class="assistant-header-meta">
        <span class="route-pill" style="background: var(--dp-color-ink); color: #fff;">SYSTEM NOTICE</span>
      </div>
      <div class="answer-body" style="color: var(--dp-color-secondary);">
        ${escapeHtml(errText)}
      </div>
    </div>
  `;
  stream.appendChild(entry);
  stream.scrollTop = stream.scrollHeight;
}

// ============================================================================
// 3. Upload Document Modal
// ============================================================================
function initUploadModal() {
  const modal = document.getElementById("uploadModal");
  const btnOpen = document.getElementById("btnOpenUpload");
  const btnClose = document.getElementById("btnCloseUpload");
  const uploadForm = document.getElementById("uploadForm");
  const fileInput = document.getElementById("pdfFileInput");
  const progress = document.getElementById("uploadProgress");
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

    progress.style.display = "block";
    btnSubmitUpload.disabled = true;

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
      modal.classList.remove("open");
      uploadForm.reset();

      // Refresh documents and switch to new document
      await loadDocuments();
      if (result.doc_id) {
        switchDocument(result.doc_id);
      }
    } catch (err) {
      alert(`Error during ingestion: ${err.message || err}`);
    } finally {
      progress.style.display = "none";
      btnSubmitUpload.disabled = false;
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
    btnText.textContent = "Close Document View";
    container.classList.add("split-mode");
    await loadDocumentPages(currentDocId);
  } else {
    column.style.display = "none";
    btn.classList.remove("active");
    btnText.textContent = "Open Document Beside Chat";
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
    chip.textContent = `p. ${i}`;
    chip.title = `Jump to Page ${i}`;
    chip.addEventListener("click", () => goToPage(i));
    strip.appendChild(chip);
  }
}
