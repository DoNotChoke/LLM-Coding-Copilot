// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode');

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {  // Create and register a new command
	const provider = {
		provideInlineCompletionItems: async (
			document, position, context, token
		) => {
			const editor = vscode.window.activeTextEditor;
			const selection = editor.selection;
			const manualKind = 0;
			const manuallyTriggered = context.triggerKind == manualKind;

			if (manuallyTriggered && position.isEqual(selection.start)) {
				editor.selection = new vscode.Selection(
					selection.start, selection.end
				);
				vscode.commands.executeCommand(
					"editor.action.inlineSuggest.trigger"
				);
				return []
			}

			if (manuallyTriggered && selection && !selection.isEmpty) {
				const selectionRange = new vscode.Range(
					selection.start, selection.end
				);
				const highlighted = editor.document.getText(selectionRange);

				var payload = {
					prompt: highlighted
				};

				const response = await fetch(
					"http://localhost:8000/generate", {
						method: "POST",
						headers: {
							"Content-Type": "application/json"
						},
						body: JSON.stringify(payload)
					}
				);

				var responseText = await response.text();

				range = new vscode.Range(selection.end, selection.end);
				return new Promise(resolve => {
					resolve([{insertText: responseText, range}])
				});
			}
		}
	};

	vscode.languages.registerInlineCompletionItemProvider (
		{scheme: 'file', language: 'python'},
		provider
	);
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
	activate,
	deactivate
}
