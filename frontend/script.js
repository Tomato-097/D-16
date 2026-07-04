/**
 * Frontend script for the Text Generator UI.
 *
 * This file:
 *   1. Reads the prompt from the textarea
 *   2. Sends it to the FastAPI backend with fetch()
 *   3. Displays the generated text (or an error message)
 */

// Backend URL — must match where uvicorn runs (see backend/main.py).
const API_URL = "http://127.0.0.1:8000/generate";

// Grab references to HTML elements once (reuse them in our functions).
const promptInput = document.getElementById("prompt");
const generateBtn = document.getElementById("generate-btn");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");

/**
 * Show a short status message under the button.
 * @param {string} message - Text to display
 * @param {boolean} isError - If true, style as an error
 */
function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

/**
 * Call the backend /generate endpoint and return the JSON response.
 * @param {string} prompt - User's input text
 * @returns {Promise<object>} Parsed JSON from the server
 */
async function callGenerateApi(prompt) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      // Tell the server we are sending JSON
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt: prompt,
      // Optional: you can change max_new_tokens here
      max_new_tokens: 128,
    }),
  });

  // If the server returned 4xx/5xx, try to read the error detail
  if (!response.ok) {
    let errorMessage = `Request failed (${response.status})`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage =
          typeof errorData.detail === "string"
            ? errorData.detail
            : JSON.stringify(errorData.detail);
      }
    } catch {
      // Response body was not JSON — keep the generic message
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Main handler when the user clicks "Generate".
 */
async function handleGenerate() {
  const prompt = promptInput.value.trim();

  // Basic client-side validation
  if (!prompt) {
    setStatus("Please enter a prompt first.", true);
    return;
  }

  // Disable the button while waiting so users do not double-submit
  generateBtn.disabled = true;
  setStatus("Generating...");
  outputEl.textContent = "";

  try {
    const data = await callGenerateApi(prompt);

    // The backend returns: { "generated_text": "..." }
    outputEl.textContent = data.generated_text || "(No text returned)";
    setStatus("Done.");
  } catch (error) {
    setStatus(error.message || "Something went wrong.", true);
    outputEl.textContent = "Could not get a response from the server.";
  } finally {
    generateBtn.disabled = false;
  }
}

// Wire up the button click to our handler
generateBtn.addEventListener("click", handleGenerate);
