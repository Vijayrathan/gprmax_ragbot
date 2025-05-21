document.addEventListener("DOMContentLoaded", () => {
  const messagesContainer = document.querySelector(".messages");
  const messageInput = document.querySelector(".message-input");
  const sendButton = document.querySelector(".send-button");
  const modeButtons = document.querySelectorAll(".mode-button");
  let currentMode = "qa";

  // Function to switch modes
  function switchMode(mode) {
    currentMode = mode;
    // Update button styles
    modeButtons.forEach((button) => {
      if (button.dataset.mode === mode) {
        button.classList.add("active");
      } else {
        button.classList.remove("active");
      }
    });
    // Update input placeholder
    messageInput.placeholder =
      mode === "qa"
        ? "Ask a question about GPRMax..."
        : "What kind of simulation would you like to create?";
    // Send mode switch command
    sendMessage(`/mode ${mode}`);
  }

  // Add click handlers for mode buttons
  modeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      switchMode(button.dataset.mode);
    });
  });

  // Function to add a message to the chat
  function addMessage(message, isUser = false) {
    const messageElement = document.createElement("div");
    messageElement.classList.add("message");
    messageElement.classList.add(isUser ? "user" : "bot");

    // Create the message content
    const messageContent = document.createElement("div");
    messageContent.classList.add("message-content");

    // Check if the message contains a code block
    const codeBlockRegex = /```gprmax\n([\s\S]*?)```/g;
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(message)) !== null) {
      // Add text before the code block
      if (match.index > lastIndex) {
        const textContent = document.createElement("p");
        textContent.textContent = message.slice(lastIndex, match.index);
        messageContent.appendChild(textContent);
      }

      // Create code block container
      const codeContainer = document.createElement("div");
      codeContainer.classList.add("code-block-container");

      // Create code block
      const codeBlock = document.createElement("pre");
      codeBlock.classList.add("code-block");

      // Get the code content
      const codeContent = match[1];
      codeBlock.textContent = codeContent;

      // Create download button
      const downloadButton = document.createElement("button");
      downloadButton.classList.add("download-button");
      downloadButton.innerHTML = '<i class="fas fa-download"></i> Download';

      // Extract filename from the first line if it exists
      const lines = codeContent.split("\n");
      const firstLine = lines[0];
      const filenameMatch = firstLine.match(/#\s*filename:\s*(.+)/);
      const filename = filenameMatch
        ? filenameMatch[1].trim()
        : "simulation.in";

      // Add download functionality
      downloadButton.addEventListener("click", () => {
        const blob = new Blob([codeContent], { type: "text/plain" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      });

      codeContainer.appendChild(codeBlock);
      codeContainer.appendChild(downloadButton);
      messageContent.appendChild(codeContainer);

      lastIndex = match.index + match[0].length;
    }

    // Add any remaining text
    if (lastIndex < message.length) {
      const textContent = document.createElement("p");
      textContent.textContent = message.slice(lastIndex);
      messageContent.appendChild(textContent);
    }

    messageElement.appendChild(messageContent);
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    return messageElement;
  }

  // Function to show typing indicator
  function showTypingIndicator() {
    const indicatorElement = document.createElement("div");
    indicatorElement.classList.add(
      "message",
      "bot",
      "typing-indicator-container"
    );

    const indicator = document.createElement("div");
    indicator.classList.add("typing-indicator");

    for (let i = 0; i < 3; i++) {
      const bubble = document.createElement("div");
      bubble.classList.add("typing-bubble");
      indicator.appendChild(bubble);
    }

    indicatorElement.appendChild(indicator);
    messagesContainer.appendChild(indicatorElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    return indicatorElement;
  }

  // Function to send a message to the bot
  async function sendMessage(message) {
    if (!message.trim()) return;

    // Add the user message to the chat
    addMessage(message, true);

    // Clear the input field
    messageInput.value = "";

    // Reset textarea height
    messageInput.style.height = "auto";

    // Show typing indicator
    const typingIndicator = showTypingIndicator();

    try {
      // Send the message to the server
      const response = await fetch("/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: message }),
      });

      const data = await response.json();

      // Remove typing indicator
      typingIndicator.remove();

      if (data.error) {
        addMessage(`Error: ${data.error}`, false);
      } else {
        // Add the bot's response to the chat with a small delay for a more natural feel
        addMessage(data.answer, false);
      }
    } catch (error) {
      // Remove typing indicator
      typingIndicator.remove();

      // Add an error message
      addMessage(
        "Sorry, there was an error processing your request. Please try again.",
        false
      );
      console.error("Error:", error);
    }
  }

  // Add event listeners
  sendButton.addEventListener("click", () => {
    sendMessage(messageInput.value);
  });

  // Function to auto-resize textarea
  function autoResizeTextarea() {
    messageInput.style.height = "auto";
    messageInput.style.height = messageInput.scrollHeight + "px";

    // Limit max height to prevent excessive growth
    const maxHeight = 150; // Maximum height in pixels
    if (messageInput.scrollHeight > maxHeight) {
      messageInput.style.height = maxHeight + "px";
      messageInput.style.overflowY = "auto";
    } else {
      messageInput.style.overflowY = "hidden";
    }
  }

  // Initialize textarea height
  messageInput.addEventListener("input", autoResizeTextarea);

  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      if (e.shiftKey) {
        // Allow Shift+Enter for new line
        // The default behavior will insert a newline
        setTimeout(autoResizeTextarea, 0);
      } else {
        // Regular Enter sends the message
        e.preventDefault();
        sendMessage(messageInput.value);
      }
    }
  });

  // Initialize with QA mode
  switchMode("qa");

  // Focus the input field when the page loads
  messageInput.focus();

  // Add a welcome message
  setTimeout(() => {
    addMessage(
      "Hello! I am the GPRMax Assistant. Ask me anything about Ground Penetrating Radar (GPR) simulations and the GPRMax software.",
      false
    );
  }, 500);
});
