(function () {
  const tab = document.getElementById('tutor-tab');
  const panel = document.getElementById('tutor-panel');
  const closeBtn = document.getElementById('tutor-close');
  const form = document.getElementById('tutor-form');
  const input = document.getElementById('tutor-input');
  const messagesEl = document.getElementById('tutor-messages');
  if (!tab || !panel) return;

  const lessonId = window.NEURONICA_CURRENT_LESSON_ID || null;

  function getCookie(name) {
    const match = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
    return match ? match.pop() : '';
  }
  const csrftoken = getCookie('csrftoken');

  function appendMessage(role, content) {
    const div = document.createElement('div');
    div.className = 'tutor-msg ' + role;
    div.textContent = content;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function openPanel() {
    panel.classList.add('open');
    tab.classList.add('hidden');
    loadHistory();
  }
  function closePanel() {
    panel.classList.remove('open');
    tab.classList.remove('hidden');
  }

  tab.addEventListener('click', openPanel);
  closeBtn.addEventListener('click', closePanel);

  let historyLoaded = false;
  function loadHistory() {
    if (historyLoaded) return;
    historyLoaded = true;
    const url = lessonId ? `/tutor/history/?lesson_id=${lessonId}` : '/tutor/history/';
    fetch(url)
      .then((r) => r.json())
      .then((data) => {
        messagesEl.innerHTML = '';
        if (!data.messages || data.messages.length === 0) {
          appendMessage('assistant', "Hi! I'm your AI tutor. Ask me anything about this lesson or the curriculum.");
          return;
        }
        data.messages.forEach((m) => appendMessage(m.role, m.content));
      })
      .catch(() => appendMessage('error', 'Could not load conversation history.'));
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    appendMessage('user', question);
    input.value = '';

    const thinkingEl = document.createElement('div');
    thinkingEl.className = 'tutor-msg assistant';
    thinkingEl.textContent = '…';
    messagesEl.appendChild(thinkingEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    fetch('/tutor/ask/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken,
      },
      body: JSON.stringify({ message: question, lesson_id: lessonId }),
    })
      .then((r) => r.json().then((data) => ({ ok: r.ok, data })))
      .then(({ ok, data }) => {
        thinkingEl.remove();
        if (ok) {
          appendMessage('assistant', data.reply);
        } else {
          appendMessage('error', data.error || 'Something went wrong.');
        }
      })
      .catch(() => {
        thinkingEl.remove();
        appendMessage('error', 'Network error — could not reach the tutor.');
      });
  });
})();
