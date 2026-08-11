/* SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pathAtOffset(ranges, offset) {
  let best = "";
  let bestSpan = Number.POSITIVE_INFINITY;
  for (const [path, range] of Object.entries(ranges || {})) {
    if (range.start <= offset && offset < range.end) {
      const span = range.end - range.start;
      if (span < bestSpan) {
        bestSpan = span;
        best = path;
      }
    }
  }
  return best;
}

function parentPath(path) {
  if (!path) {
    return "";
  }
  const bracket = path.lastIndexOf("[");
  const dot = path.lastIndexOf(".");
  if (bracket > dot) {
    return path.slice(0, bracket);
  }
  if (dot >= 0) {
    return path.slice(0, dot);
  }
  return "";
}

/** Walk up from a YAML leaf path until we find a diagram node with that data-path. */
function nearestDiagramPath(root, path) {
  let candidate = path || "";
  while (candidate) {
    if (root.querySelector(`[data-path="${CSS.escape(candidate)}"]`)) {
      return candidate;
    }
    candidate = parentPath(candidate);
  }
  return "";
}

function buildHighlightHtml(text, ranges, activePath) {
  const range = activePath && ranges ? ranges[activePath] : null;
  if (!range) {
    return escapeHtml(text);
  }
  const start = Math.max(0, Math.min(text.length, range.start));
  const end = Math.max(start, Math.min(text.length, range.end));
  return (
    escapeHtml(text.slice(0, start)) +
    '<mark class="nss-pii-yaml-mark">' +
    escapeHtml(text.slice(start, end)) +
    "</mark>" +
    escapeHtml(text.slice(end))
  );
}

function renderRows(rows) {
  if (!rows || rows.length === 0) {
    return '<div class="nss-pii-row nss-pii-row-empty">(none)</div>';
  }
  return rows
    .map((row) => {
      const secondary = row.secondary
        ? `<span class="nss-pii-secondary">${escapeHtml(row.secondary)}</span>`
        : "";
      return (
        `<div class="nss-pii-row" data-path="${escapeHtml(row.path)}" tabindex="0">` +
        `<span class="nss-pii-primary">${escapeHtml(row.primary)}</span>` +
        secondary +
        `</div>`
      );
    })
    .join("");
}

// The button and its panel can sit in different wrappers (a card header keeps
// both; the scope pill keeps the button but puts the panel underneath), so the
// click handler finds the pair via the nearest .nss-pii-help-holder.
function helpButton(text, label) {
  if (!text) {
    return "";
  }
  return (
    `<button type="button" class="nss-pii-help-btn" aria-expanded="false" ` +
    `aria-label="Explain ${escapeHtml(label)}" title="What is this?">?</button>`
  );
}

function helpPanel(text) {
  return text ? `<div class="nss-pii-help-text" hidden>${escapeHtml(text)}</div>` : "";
}

function renderCard(card) {
  const compartments = (card.compartments || [])
    .map((compartment) => {
      const hint = compartment.hint
        ? `<span class="nss-pii-compartment-hint">${escapeHtml(compartment.hint)}</span>`
        : "";
      return (
        `<section class="nss-pii-compartment" data-path="${escapeHtml(compartment.path)}">` +
        `<header class="nss-pii-compartment-label">${escapeHtml(compartment.label)}${hint}</header>` +
        `<div class="nss-pii-rows">${renderRows(compartment.rows)}</div>` +
        `</section>`
      );
    })
    .join("");
  return (
    `<article class="nss-pii-card nss-pii-card-${escapeHtml(card.kind)}" data-path="${escapeHtml(card.path)}" tabindex="0">` +
    `<header class="nss-pii-card-header nss-pii-help-holder">` +
    `<div class="nss-pii-card-titlerow">` +
    `<span class="nss-pii-card-title">${escapeHtml(card.title)}</span>` +
    helpButton(card.help, card.title) +
    `</div>` +
    helpPanel(card.help) +
    `</header>` +
    compartments +
    `</article>`
  );
}

function renderDiagram(diagram) {
  const scope = diagram && diagram.scope ? diagram.scope : "dataframe";
  const scopePath = (diagram && diagram.scope_path) || "scope";
  const scopeHelp = (diagram && diagram.scope_help) || "";
  const cards = ((diagram && diagram.cards) || []).map(renderCard).join("");
  return (
    `<div class="nss-pii-diagram">` +
    `<div class="nss-pii-scope-row nss-pii-help-holder">` +
    `<div class="nss-pii-scope" data-path="${escapeHtml(scopePath)}" tabindex="0">scope: <strong>${escapeHtml(scope)}</strong>` +
    helpButton(scopeHelp, "scope") +
    `</div>` +
    helpPanel(scopeHelp) +
    `</div>` +
    `<div class="nss-pii-cards">${cards}</div>` +
    `</div>`
  );
}

function setActivePath(model, root, path, { scrollYaml = false } = {}) {
  model.set("active_path", path || "");
  model.save_changes();
  root.querySelectorAll(".nss-pii-active").forEach((node) => node.classList.remove("nss-pii-active"));
  if (!path) {
    return;
  }
  root.querySelectorAll(`[data-path="${CSS.escape(path)}"]`).forEach((node) => {
    node.classList.add("nss-pii-active");
  });
  if (scrollYaml) {
    const mark = root.querySelector(".nss-pii-yaml-mark");
    const editor = root.querySelector(".nss-pii-yaml-editor");
    const backdrop = root.querySelector(".nss-pii-yaml-backdrop");
    if (mark && editor && backdrop) {
      root._nssSuppressScrollSync = true;
      editor.scrollTop = Math.max(0, mark.offsetTop - 48);
      backdrop.scrollTop = editor.scrollTop;
      requestAnimationFrame(() => {
        root._nssSuppressScrollSync = false;
      });
    }
  }
}

function render({ model, el }) {
  el.innerHTML = `
    <div class="nss-pii-preview">
      <header class="nss-pii-header">
        <h2 class="nss-pii-title">PII Replacement Plan Preview</h2>
      </header>
      <div class="nss-pii-panes">
        <div class="nss-pii-yaml-pane">
          <div class="nss-pii-pane-label">YAML</div>
          <div class="nss-pii-yaml-shell">
            <pre class="nss-pii-yaml-backdrop" aria-hidden="true"></pre>
            <textarea class="nss-pii-yaml-editor" spellcheck="false" aria-label="PII replacement plan YAML"></textarea>
          </div>
        </div>
        <div class="nss-pii-diagram-pane">
          <div class="nss-pii-pane-label">Diagram</div>
          <div class="nss-pii-diagram-host"></div>
        </div>
      </div>
      <div class="nss-pii-footer">
        <button type="button" class="nss-pii-render-btn">Save and render diagram</button>
        <div class="nss-pii-status"></div>
      </div>
      <div class="nss-pii-error" hidden></div>
      <div class="nss-pii-warnings" hidden></div>
    </div>
  `;

  const root = el.querySelector(".nss-pii-preview");
  const editor = el.querySelector(".nss-pii-yaml-editor");
  const backdrop = el.querySelector(".nss-pii-yaml-backdrop");
  const diagramHost = el.querySelector(".nss-pii-diagram-host");
  const errorBox = el.querySelector(".nss-pii-error");
  const warningsBox = el.querySelector(".nss-pii-warnings");
  const statusBox = el.querySelector(".nss-pii-status");
  const renderBtn = el.querySelector(".nss-pii-render-btn");

  function syncEditorFromModel() {
    const text = model.get("yaml_text") || "";
    if (editor.value !== text) {
      editor.value = text;
    }
    backdrop.innerHTML = buildHighlightHtml(text, model.get("ranges"), model.get("active_path"));
  }

  function syncStatusFromModel() {
    const err = model.get("error") || "";
    if (err) {
      errorBox.hidden = false;
      errorBox.textContent = err;
    } else {
      errorBox.hidden = true;
      errorBox.textContent = "";
    }

    const warnings = model.get("warnings") || [];
    if (warnings.length > 0) {
      warningsBox.hidden = false;
      warningsBox.innerHTML =
        `<div class="nss-pii-warnings-title">Placement warnings</div>` +
        `<ul class="nss-pii-warnings-list">` +
        warnings.map((msg) => `<li>${escapeHtml(msg)}</li>`).join("") +
        `</ul>`;
    } else {
      warningsBox.hidden = true;
      warningsBox.innerHTML = "";
    }
    statusBox.textContent = model.get("status") || "";
  }

  function syncDiagramFromModel() {
    diagramHost.innerHTML = renderDiagram(model.get("diagram") || {});
    syncStatusFromModel();
    setActivePath(model, root, model.get("active_path") || "");
  }

  function refreshHighlight() {
    backdrop.innerHTML = buildHighlightHtml(
      editor.value,
      model.get("ranges"),
      model.get("active_path"),
    );
  }

  syncEditorFromModel();
  syncDiagramFromModel();

  model.on("change:yaml_text", syncEditorFromModel);
  model.on("change:diagram", syncDiagramFromModel);
  model.on("change:ranges", () => {
    refreshHighlight();
  });
  model.on("change:error", syncStatusFromModel);
  model.on("change:warnings", syncStatusFromModel);
  model.on("change:status", syncStatusFromModel);
  model.on("change:active_path", () => {
    setActivePath(model, root, model.get("active_path") || "");
    refreshHighlight();
  });

  renderBtn.addEventListener("click", () => {
    model.set("yaml_text", editor.value);
    model.set("active_path", "");
    model.set("render_nonce", (model.get("render_nonce") || 0) + 1);
    model.save_changes();
  });

  let syncingScroll = false;

  function maxScroll(node) {
    return Math.max(0, node.scrollHeight - node.clientHeight);
  }

  function syncScrollFrom(source, target) {
    if (syncingScroll || root._nssSuppressScrollSync) {
      if (source === editor) {
        backdrop.scrollTop = editor.scrollTop;
        backdrop.scrollLeft = editor.scrollLeft;
      }
      return;
    }
    const sourceMax = maxScroll(source);
    const targetMax = maxScroll(target);
    syncingScroll = true;
    try {
      target.scrollTop = sourceMax <= 0 || targetMax <= 0 ? 0 : (source.scrollTop / sourceMax) * targetMax;
      if (source === editor || target === editor) {
        backdrop.scrollTop = editor.scrollTop;
        backdrop.scrollLeft = editor.scrollLeft;
      }
    } finally {
      requestAnimationFrame(() => {
        syncingScroll = false;
      });
    }
  }

  editor.addEventListener("scroll", () => {
    syncScrollFrom(editor, diagramHost);
  });

  diagramHost.addEventListener("scroll", () => {
    syncScrollFrom(diagramHost, editor);
  });

  editor.addEventListener("input", () => {
    backdrop.innerHTML = buildHighlightHtml(editor.value, model.get("ranges"), "");
  });

  function syncPathFromEditorCaret() {
    // YAML leaves (e.g. ...columns_to_replace[0].column_name) are finer than
    // diagram nodes (rows/cards). Map to the nearest ancestor that exists in
    // the diagram so both panes highlight together.
    const yamlPath = pathAtOffset(model.get("ranges"), editor.selectionStart);
    const path = nearestDiagramPath(root, yamlPath) || yamlPath;
    if (path !== (model.get("active_path") || "")) {
      setActivePath(model, root, path);
      refreshHighlight();
    }
  }

  editor.addEventListener("keyup", syncPathFromEditorCaret);
  editor.addEventListener("click", syncPathFromEditorCaret);
  editor.addEventListener("select", syncPathFromEditorCaret);

  diagramHost.addEventListener("mousemove", (event) => {
    const target = event.target.closest("[data-path]");
    const path = target ? target.getAttribute("data-path") : "";
    if ((path || "") !== (model.get("active_path") || "")) {
      setActivePath(model, root, path || "", { scrollYaml: true });
      refreshHighlight();
    }
  });

  diagramHost.addEventListener("mouseleave", () => {
    setActivePath(model, root, "");
    refreshHighlight();
  });

  diagramHost.addEventListener("focusin", (event) => {
    const target = event.target.closest("[data-path]");
    if (target) {
      setActivePath(model, root, target.getAttribute("data-path") || "");
      refreshHighlight();
    }
  });

  // Delegated so it survives every diagram re-render.
  diagramHost.addEventListener("click", (event) => {
    const btn = event.target.closest(".nss-pii-help-btn");
    if (!btn) {
      return;
    }
    event.preventDefault();
    const holder = btn.closest(".nss-pii-help-holder");
    const panel = holder && holder.querySelector(".nss-pii-help-text");
    if (!panel) {
      return;
    }
    const show = panel.hidden;
    panel.hidden = !show;
    btn.setAttribute("aria-expanded", show ? "true" : "false");
  });
}

export default { render };
