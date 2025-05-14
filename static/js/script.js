document.addEventListener("DOMContentLoaded", () => {
  const messagesContainer = document.querySelector(".messages");
  const messageInput = document.querySelector(".message-input");
  const sendButton = document.querySelector(".send-button");

  // Function to add a message to the chat
  function addMessage(message, isUser = false) {
    const messageElement = document.createElement("div");
    messageElement.classList.add("message");
    messageElement.classList.add(isUser ? "user" : "bot");

    // Create the message content
    const messageContent = document.createElement("p");
    messageContent.textContent = message;
    messageElement.appendChild(messageContent);

    messagesContainer.appendChild(messageElement);

    // Scroll to the bottom of the chat
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

  // Event listener for the send button
  sendButton.addEventListener("click", () => {
    sendMessage(messageInput.value);
  });

  // Event listener for pressing Enter in the input field
  messageInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      sendMessage(messageInput.value);
    }
  });

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
