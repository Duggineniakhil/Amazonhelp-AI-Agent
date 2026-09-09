document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const chatWindow = document.getElementById("chatWindow");
    
    // Diagnostic Panel Elements
    const intentLabel = document.getElementById("intentLabel");
    const intentConfidenceBar = document.getElementById("intentConfidenceBar");
    const intentConfidenceText = document.getElementById("intentConfidenceText");
    
    const escalationDecision = document.getElementById("escalationDecision");
    const escalationReason = document.getElementById("escalationReason");
    
    const retrievalConfidenceBar = document.getElementById("retrievalConfidenceBar");
    const retrievalConfidenceText = document.getElementById("retrievalConfidenceText");
    const retrievedExample = document.getElementById("retrievedExample");
    
    const processingTime = document.getElementById("processingTime");

    let threadLength = 0;

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        // 1. Add Customer Message to UI
        appendMessage(message, "customer");
        userInput.value = "";
        threadLength++;

        // 2. Add Typing Indicator
        const typingId = showTypingIndicator();
        scrollToBottom();

        // 3. Reset Diagnostics to loading state
        resetDiagnostics();

        try {
            // 4. Send API Request
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message, thread_length: threadLength })
            });

            if (!response.ok) throw new Error("API Error");
            
            const data = await response.json();

            // 5. Remove Typing Indicator
            removeElement(typingId);

            // 6. Add Agent Message to UI
            appendMessage(data.reply, "agent");

            // 7. Update Diagnostics Panel
            updateDiagnostics(data);

        } catch (error) {
            removeElement(typingId);
            appendMessage("Sorry, I encountered an error connecting to the server.", "system");
            console.error(error);
        }
        
        scrollToBottom();
    });

    function appendMessage(text, role) {
        const msgDiv = document.createElement("div");
        msgDiv.classList.add("message", role);
        
        const contentDiv = document.createElement("div");
        contentDiv.classList.add("message-content");
        
        const headerDiv = document.createElement("div");
        headerDiv.classList.add("message-header");
        
        if (role === "agent") {
            headerDiv.innerHTML = `<i class="fa-brands fa-amazon"></i> AmazonHelp`;
        } else if (role === "customer") {
            headerDiv.innerHTML = `<i class="fa-solid fa-user"></i> You`;
        }
        
        const textElement = document.createElement("p");
        textElement.textContent = text;
        
        if (role !== "system") {
            contentDiv.appendChild(headerDiv);
        }
        contentDiv.appendChild(textElement);
        msgDiv.appendChild(contentDiv);
        
        chatWindow.appendChild(msgDiv);
    }

    function showTypingIndicator() {
        const id = "typing-" + Date.now();
        const msgDiv = document.createElement("div");
        msgDiv.classList.add("message", "agent");
        msgDiv.id = id;
        
        const contentDiv = document.createElement("div");
        contentDiv.classList.add("message-content");
        
        const typingDiv = document.createElement("div");
        typingDiv.classList.add("typing-indicator");
        typingDiv.innerHTML = `<span></span><span></span><span></span>`;
        
        contentDiv.appendChild(typingDiv);
        msgDiv.appendChild(contentDiv);
        chatWindow.appendChild(msgDiv);
        
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function resetDiagnostics() {
        intentLabel.textContent = "Processing...";
        intentLabel.className = "value badge neutral";
        intentConfidenceBar.style.width = "0%";
        intentConfidenceText.textContent = "0%";
        
        escalationDecision.textContent = "Processing...";
        escalationDecision.className = "value badge neutral";
        escalationReason.textContent = "Analyzing signals...";
        
        retrievalConfidenceBar.style.width = "0%";
        retrievalConfidenceText.textContent = "0%";
        retrievedExample.innerHTML = `<p class="context-hint">Searching knowledge base...</p>`;
    }

    function updateDiagnostics(data) {
        // Intent
        const intent = data.intent.label;
        const conf = (data.intent.confidence * 100).toFixed(0);
        
        intentLabel.textContent = intent.replace("_", " ").toUpperCase();
        
        // Color code based on confidence
        if (data.intent.confidence > 0.7) intentLabel.className = "value badge success";
        else if (data.intent.confidence > 0.4) intentLabel.className = "value badge warning";
        else intentLabel.className = "value badge danger";
        
        intentConfidenceBar.style.width = `${conf}%`;
        
        // Change progress bar color based on value
        if(conf > 70) intentConfidenceBar.style.backgroundColor = 'var(--success)';
        else if (conf > 40) intentConfidenceBar.style.backgroundColor = 'var(--warning)';
        else intentConfidenceBar.style.backgroundColor = 'var(--error)';
        
        intentConfidenceText.textContent = `${conf}%`;

        // Escalation
        const decision = data.escalation.decision;
        if (decision === "auto_handle") {
            escalationDecision.textContent = "AUTO-HANDLE";
            escalationDecision.className = "value badge success";
        } else {
            escalationDecision.textContent = "ESCALATE";
            escalationDecision.className = "value badge warning"; // Escalate isn't bad, just needs human
        }
        escalationReason.textContent = data.escalation.reason;

        // Retrieval
        const sim = (data.retrieval.top_similarity * 100).toFixed(0);
        retrievalConfidenceBar.style.width = `${Math.min(100, Math.max(0, sim))}%`;
        retrievalConfidenceText.textContent = `${sim}%`;
        
        if (data.retrieval.examples && data.retrieval.examples.length > 0) {
            retrievedExample.innerHTML = `<strong>Best Match:</strong> "${data.retrieval.examples[0].customer}..."`;
        } else {
            retrievedExample.innerHTML = `<p class="context-hint">No close matches found.</p>`;
        }

        // Processing Time
        processingTime.textContent = `${data.processing_time_seconds.toFixed(2)}s`;
    }
});
