document.addEventListener('DOMContentLoaded', function() {
    // Constants
    const API_ENDPOINT = 'https://muchadoaboutrag.onrender.com/query';    
    // DOM Elements
    const questionInput = document.getElementById('question-input');
    const askButton = document.getElementById('ask-button');
    const loadingContainer = document.getElementById('loading-container');
    const answerContainer = document.getElementById('answer-container');
    const questionDisplay = document.getElementById('question-display');
    const answerDisplay = document.getElementById('answer-display');
    const answerTimestamp = document.getElementById('answer-timestamp');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    const conversationHistory = document.getElementById('conversation-history');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const copyAnswerBtn = document.getElementById('copy-answer-btn');
    const sampleQuestions = document.querySelectorAll('.sample-question');
    
    // Initialize conversation history from localStorage
    let history = JSON.parse(localStorage.getItem('conversationHistory')) || [];
    updateHistoryDisplay();
    
    // Event Listeners
    askButton.addEventListener('click', handleAskQuestion);
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleAskQuestion();
        }
    });
    
    clearHistoryBtn.addEventListener('click', clearHistory);
    copyAnswerBtn.addEventListener('click', copyAnswerToClipboard);
    
    // Sample questions click handler
    sampleQuestions.forEach(question => {
        question.addEventListener('click', function() {
            questionInput.value = this.textContent;
            handleAskQuestion();
        });
    });
    
    // Functions
    async function handleAskQuestion() {
        const question = questionInput.value.trim();
        if (!question) return;
        
        // Show loading, hide previous results
        loadingContainer.classList.remove('d-none');
        answerContainer.classList.add('d-none');
        errorContainer.classList.add('d-none');
        
        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: question })
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Display the answer
            questionDisplay.textContent = question;
            answerDisplay.innerHTML = formatAnswer(data.answer);
            answerTimestamp.textContent = getCurrentTimestamp();
            answerContainer.classList.remove('d-none');
            
            // Add to history
            addToHistory(question, data.answer);
            
        } catch (error) {
            console.error('Error:', error);
            errorMessage.textContent = `Error: ${error.message || 'Something went wrong. Please try again.'}`;
            errorContainer.classList.remove('d-none');
        } finally {
            loadingContainer.classList.add('d-none');
        }
    }
    
    function formatAnswer(text) {
        // Basic formatting to handle markdown-like syntax
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/Act (\d+), Scene (\d+)/gi, '<strong>Act $1, Scene $2</strong>');
    }
    
    function getCurrentTimestamp() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' + 
               now.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
    
    function addToHistory(question, answer) {
        const timestamp = new Date().toISOString();
        
        // Add to the beginning of the array
        history.unshift({ question, answer, timestamp });
        
        // Keep only the last 10 items
        if (history.length > 10) {
            history = history.slice(0, 10);
        }
        
        // Save to localStorage
        localStorage.setItem('conversationHistory', JSON.stringify(history));
        
        // Update the display
        updateHistoryDisplay();
    }
    
    function updateHistoryDisplay() {
        if (history.length === 0) {
            conversationHistory.innerHTML = '<li class="list-group-item text-center text-muted">No questions yet</li>';
            return;
        }
        
        conversationHistory.innerHTML = '';
        
        history.forEach((item, index) => {
            const li = document.createElement('li');
            li.className = 'list-group-item history-item';
            li.dataset.index = index;
            
            // Create a snippet of the answer (first 60 chars)
            const answerSnippet = item.answer.length > 60 
                ? item.answer.substring(0, 60) + '...' 
                : item.answer;
            
            // Format the date
            const date = new Date(item.timestamp);
            const formattedDate = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
            
            li.innerHTML = `
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <div class="history-question">${item.question}</div>
                        <div class="history-snippet">${answerSnippet}</div>
                    </div>
                    <small class="text-muted">${formattedDate}</small>
                </div>
            `;
            
            li.addEventListener('click', function() {
                loadHistoryItem(index);
            });
            
            conversationHistory.appendChild(li);
        });
    }
    
    function loadHistoryItem(index) {
        const item = history[index];
        
        // Display the selected Q&A pair
        questionDisplay.textContent = item.question;
        answerDisplay.innerHTML = formatAnswer(item.answer);
        
        // Format timestamp
        const date = new Date(item.timestamp);
        answerTimestamp.textContent = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' + 
            date.toLocaleDateString([], { month: 'short', day: 'numeric' });
        
        // Show the answer container
        answerContainer.classList.remove('d-none');
        errorContainer.classList.add('d-none');
        
        // Also populate the question input
        questionInput.value = item.question;
        
        // Scroll to the answer section
        answerContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    function clearHistory() {
        // Clear history array
        history = [];
        
        // Update localStorage
        localStorage.removeItem('conversationHistory');
        
        // Update the display
        updateHistoryDisplay();
    }
    
    function copyAnswerToClipboard() {
        if (!answerDisplay.textContent) return;
        
        // Create a temporary textarea element to copy from
        const textarea = document.createElement('textarea');
        textarea.value = `Q: ${questionDisplay.textContent}\n\nA: ${answerDisplay.textContent}`;
        document.body.appendChild(textarea);
        textarea.select();
        
        try {
            document.execCommand('copy');
            // Change button text temporarily
            const originalText = copyAnswerBtn.innerHTML;
            copyAnswerBtn.innerHTML = '<i class="bi bi-check"></i> Copied!';
            
            setTimeout(() => {
                copyAnswerBtn.innerHTML = originalText;
            }, 2000);
        } catch (err) {
            console.error('Failed to copy text: ', err);
        }
        
        document.body.removeChild(textarea);
    }
}); 