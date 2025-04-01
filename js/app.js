document.addEventListener('DOMContentLoaded', function() {
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
    const searchSuggestions = document.getElementById('search-suggestions');
    const suggestionItems = document.querySelectorAll('.suggestion-item');
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
  
    // API endpoint
    const API_ENDPOINT = 'https://muchadoaboutrag.onrender.com/query';
  
    // Initialize conversation history from localStorage
    let history = JSON.parse(localStorage.getItem('conversationHistory')) || [];
    updateHistoryDisplay();
  
    // Event Listeners
    askButton.addEventListener('click', handleAskQuestion);
    
    questionInput.addEventListener('focus', function() {
      if (questionInput.value.trim() === '') {
        searchSuggestions.classList.remove('d-none');
        updateSuggestionsPosition();
      }
    });
    questionInput.addEventListener('input', function() {
      if (questionInput.value.trim() !== '') {
        searchSuggestions.classList.add('d-none');
      }
    });
    questionInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        handleAskQuestion();
      }
    });
  
    clearHistoryBtn.addEventListener('click', clearHistory);
    copyAnswerBtn.addEventListener('click', copyAnswerToClipboard);
  
    // Sidebar toggle
    sidebarToggle.addEventListener('click', function() {
      sidebar.classList.toggle('sidebar-hidden');
      const icon = this.querySelector('i');
      if (sidebar.classList.contains('sidebar-hidden')) {
        icon.classList.replace('bi-chevron-left', 'bi-chevron-right');
      } else {
        icon.classList.replace('bi-chevron-right', 'bi-chevron-left');
      }
    });
  
    // Handle suggestion selection
    suggestionItems.forEach(item => {
      item.addEventListener('click', function() {
        questionInput.value = this.dataset.value;
        searchSuggestions.classList.add('d-none');
        handleAskQuestion();
      });
    });
  
    // Hide suggestions when clicking outside
    document.addEventListener('click', function(e) {
      if (
        !questionInput.contains(e.target) &&
        !searchSuggestions.contains(e.target)
      ) {
        searchSuggestions.classList.add('d-none');
      }
    });
  
    // Main question handler
    async function handleAskQuestion() {
      const question = questionInput.value.trim();
      if (!question) return;
  
      // Hide suggestions
      searchSuggestions.classList.add('d-none');
  
      // Show loading, hide previous results
      loadingContainer.classList.remove('d-none');
      answerContainer.classList.add('d-none');
      errorContainer.classList.add('d-none');
  
      // Disable button and show loading state
      askButton.disabled = true;
      askButton.innerHTML =
        '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Thinking...';
  
      try {
        const response = await fetch(API_ENDPOINT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ question: question })
        });
  
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(
            errorData.detail || 
            `Server error: ${response.status} ${response.statusText}`
          );
        }
  
        const data = await response.json();
  
        // Display the answer
        questionDisplay.textContent = question;
        answerDisplay.innerHTML = formatAnswer(data.answer);
        answerTimestamp.textContent = getCurrentTimestamp();
        answerContainer.classList.remove('d-none');
  
        // Add to history
        addToHistory(question, data.answer);
  
        // On mobile, hide sidebar
        if (window.innerWidth < 768) {
          sidebar.classList.add('sidebar-hidden');
          const icon = sidebarToggle.querySelector('i');
          icon.classList.replace('bi-chevron-left', 'bi-chevron-right');
        }
      } catch (error) {
        console.error('Error:', error);
        errorMessage.textContent = `Error: ${
          error.message || 'Something went wrong. Please try again.'
        }`;
        errorContainer.classList.remove('d-none');
      } finally {
        askButton.disabled = false;
        askButton.innerHTML = '<i class="bi bi-search me-1"></i> Ask';
        loadingContainer.classList.add('d-none');
      }
    }
  
    function formatAnswer(text) {
      return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n\n/g, '<br><br>')
        .replace(
          /Act (\d+|[IVX]+), Scene (\d+|[IVX]+)/gi,
          '<strong>Act $1, Scene $2</strong>'
        );
    }
  
    function getCurrentTimestamp() {
      const now = new Date();
      return (
        now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) +
        ' ' +
        now.toLocaleDateString([], { month: 'short', day: 'numeric' })
      );
    }
  
    function addToHistory(question, answer) {
      const timestamp = new Date().toISOString();
      history.unshift({ question, answer, timestamp });
      if (history.length > 10) {
        history = history.slice(0, 10);
      }
      localStorage.setItem('conversationHistory', JSON.stringify(history));
      updateHistoryDisplay();
    }
  
    function updateHistoryDisplay() {
      if (history.length === 0) {
        conversationHistory.innerHTML =
          '<li class="list-group-item text-center text-muted bg-dark text-light">No questions yet</li>';
        return;
      }
  
      conversationHistory.innerHTML = '';
      history.forEach((item, index) => {
        const li = document.createElement('li');
        li.className = 'list-group-item history-item';
        li.dataset.index = index;
  
        const snippet =
          item.answer.length > 50
            ? item.answer.substring(0, 50) + '...'
            : item.answer;
  
        const date = new Date(item.timestamp);
        const formattedDate = date.toLocaleDateString([], {
          month: 'short',
          day: 'numeric'
        });
  
        li.innerHTML = `
          <div class="d-flex flex-column">
            <div class="history-question">Q: ${item.question}</div>
            <div class="history-snippet"><strong>A:</strong> ${snippet}</div>
          </div>
          <small class="text-muted">${formattedDate}</small>
        `;
  
        li.addEventListener('click', () => loadHistoryItem(index));
        conversationHistory.appendChild(li);
      });
    }
  
    function loadHistoryItem(index) {
      const item = history[index];
  
      questionDisplay.textContent = item.question;
      answerDisplay.innerHTML = formatAnswer(item.answer);
  
      const date = new Date(item.timestamp);
      answerTimestamp.textContent =
        date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) +
        ' ' +
        date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  
      answerContainer.classList.remove('d-none');
      errorContainer.classList.add('d-none');
      questionInput.value = item.question;
  
      if (window.innerWidth < 768) {
        sidebar.classList.add('sidebar-hidden');
        const icon = sidebarToggle.querySelector('i');
        icon.classList.replace('bi-chevron-left', 'bi-chevron-right');
      }
    }
  
    function clearHistory() {
      history = [];
      localStorage.removeItem('conversationHistory');
      updateHistoryDisplay();
    }
  
    function copyAnswerToClipboard() {
      if (!answerDisplay.textContent) return;
      const textarea = document.createElement('textarea');
      textarea.value = `Q: ${questionDisplay.textContent}\n\nA: ${answerDisplay.textContent}`;
      document.body.appendChild(textarea);
      textarea.select();
  
      try {
        document.execCommand('copy');
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
  
    function updateSuggestionsPosition() {
      const questionInput = document.getElementById('question-input');
      const searchSuggestions = document.getElementById('search-suggestions');
      
      if (questionInput && searchSuggestions) {
        // Position the suggestions below the search bar
        const inputRect = questionInput.getBoundingClientRect();
        const parentRect = questionInput.closest('.search-container').getBoundingClientRect();
        
        searchSuggestions.style.width = inputRect.width + 'px';
        searchSuggestions.style.top = (inputRect.bottom - parentRect.top) + 'px';
      }
    }
  
    // Call this when the page loads and when window resizes
    document.addEventListener('DOMContentLoaded', updateSuggestionsPosition);
    window.addEventListener('resize', updateSuggestionsPosition);
  });