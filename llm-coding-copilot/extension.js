// extension.js
const vscode = require("vscode");

const API_URL = "http://127.0.0.1:8005/generate";

const PREFIX_MAX_CHARS = 12000;
const SUFFIX_MAX_CHARS = 4000;

const DEFAULT_GEN = {
  max_new_tokens: 96,
  temperature: 0.0,
  top_p: 1.0,
  do_sample: false,
  extra_stop: ["\n\n\ndef ", "\n\nclass ", "```"],
};

let lastAbort = null;

// Manual gating: only call API when user presses Alt+S
let manualArmed = false;
// Optional: expire manual mode quickly (avoid accidental spamming)
let manualTimer = null;

function armManualTrigger(ms = 2000) {
  manualArmed = true;
  if (manualTimer) clearTimeout(manualTimer);
  manualTimer = setTimeout(() => {
    manualArmed = false;
    manualTimer = null;
  }, ms);
}

function clampFromEnd(text, maxChars) {
  if (!text) return "";
  return text.length > maxChars ? text.slice(text.length - maxChars) : text;
}

function clampFromStart(text, maxChars) {
  if (!text) return "";
  return text.length > maxChars ? text.slice(0, maxChars) : text;
}

function getPrefixSuffix(document, position) {
  const full = document.getText();
  const offset = document.offsetAt(position);
  const prefix = clampFromEnd(full.slice(0, offset), PREFIX_MAX_CHARS);
  const suffix = clampFromStart(full.slice(offset), SUFFIX_MAX_CHARS);
  return { prefix, suffix };
}

async function callApi(payload, signal) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${txt}`);
  }
  return res.json();
}

function cleanCompletion(s) {
  if (!s) return "";
  // remove trailing spaces that can cause odd rendering
  return String(s).replace(/\r/g, "").replace(/[ \t]+$/gm, "").trimEnd();
}

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  console.log('Extension "llm-coding-agent" activated');

  const provider = {
    async provideInlineCompletionItems(document, position, inlineContext, token) {
      try {
        // Gate: only serve when user armed manual trigger
        if (!manualArmed) {
          return [];
        }

        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.uri.toString() !== document.uri.toString()) {
          return [];
        }

        // Cancel previous request
        if (lastAbort) lastAbort.abort();
        lastAbort = new AbortController();
        token.onCancellationRequested(() => {
          try { lastAbort.abort(); } catch {}
        });

        const { prefix, suffix } = getPrefixSuffix(document, position);
        if (!prefix || prefix.trim().length === 0) return [];

        const payload = { prefix, suffix, ...DEFAULT_GEN };

        console.log("[llm] calling API…");
        const data = await callApi(payload, lastAbort.signal);
        const completion = cleanCompletion(data?.completion || "");

        // Disarm after one response attempt (prevents repeated calls)
        manualArmed = false;

        if (!completion || completion.trim().length === 0) return [];

        const range = new vscode.Range(position, position);
        return [new vscode.InlineCompletionItem(completion, range)];
      } catch (err) {
        if (err && (err.name === "AbortError" || String(err).includes("AbortError"))) {
          return [];
        }
        console.error("[llm] inline error:", err);
        // Disarm on error too
        manualArmed = false;
        return [];
      }
    },
  };

  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider({ pattern: "**" }, provider)
  );

  // Manual trigger command
  context.subscriptions.push(
    vscode.commands.registerCommand("llm-coding-agent.inlineSuggest", async () => {
      armManualTrigger(2000);
      await vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("llm-coding-agent.helloWorld", () => {
      vscode.window.showInformationMessage("Hello World from llm-coding-agent!");
    })
  );
}

function deactivate() {
  if (lastAbort) {
    try { lastAbort.abort(); } catch {}
  }
  if (manualTimer) clearTimeout(manualTimer);
}

module.exports = { activate, deactivate };
