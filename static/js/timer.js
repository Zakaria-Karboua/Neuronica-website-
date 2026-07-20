(function () {
  const display = document.getElementById('timer-display');
  const ring = document.getElementById('progress-ring');
  const label = document.getElementById('phase-label');
  const startBtn = document.getElementById('start-btn');
  const pauseBtn = document.getElementById('pause-btn');
  const resetBtn = document.getElementById('reset-btn');
  const lessonSelect = document.getElementById('lesson-select');
  if (!display || !ring) return;

  const CIRCUMFERENCE = 2 * Math.PI * 100; // r=100
  ring.style.strokeDasharray = CIRCUMFERENCE;

  let focusMinutes = 25;
  let breakMinutes = 5;
  let remaining = focusMinutes * 60;
  let totalForPhase = remaining;
  let onBreak = false;
  let intervalId = null;
  let sessionStartedAt = null;

  document.querySelectorAll('input[name="duration"]').forEach((radio) => {
    radio.addEventListener('change', (e) => {
      focusMinutes = parseInt(e.target.value, 10);
      breakMinutes = focusMinutes === 25 ? 5 : 10;
      resetTimer();
    });
  });

  function formatTime(totalSeconds) {
    const m = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
    const s = Math.floor(totalSeconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  function updateDisplay() {
    display.textContent = formatTime(remaining);
    const fraction = remaining / totalForPhase;
    ring.style.strokeDashoffset = CIRCUMFERENCE * (1 - fraction);
  }

  function tick() {
    remaining -= 1;
    updateDisplay();
    if (remaining <= 0) {
      completePhase();
    }
  }

  function completePhase() {
    clearInterval(intervalId);
    intervalId = null;
    logSession();
    onBreak = !onBreak;
    totalForPhase = (onBreak ? breakMinutes : focusMinutes) * 60;
    remaining = totalForPhase;
    label.textContent = onBreak ? 'ORBIT (BREAK)' : 'FOCUS';
    updateDisplay();
    startTimer(); // auto-continue into the next phase
  }

  function startTimer() {
    if (intervalId) return;
    sessionStartedAt = new Date().toISOString();
    intervalId = setInterval(tick, 1000);
  }

  function pauseTimer() {
    clearInterval(intervalId);
    intervalId = null;
  }

  function resetTimer() {
    pauseTimer();
    onBreak = false;
    totalForPhase = focusMinutes * 60;
    remaining = totalForPhase;
    label.textContent = 'FOCUS';
    updateDisplay();
  }

  function logSession() {
    if (!window.NEURONICA_USER_AUTHENTICATED) return;
    // `onBreak` here still reflects the phase that JUST finished (flip happens after this call).
    const justFinishedWasFocus = !onBreak;
    const duration = justFinishedWasFocus ? focusMinutes * 60 : breakMinutes * 60;
    fetch('/focus/log-session/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': window.NEURONICA_CSRF_TOKEN,
      },
      body: JSON.stringify({
        session_type: justFinishedWasFocus ? 'focus' : 'short_break',
        duration_seconds: duration,
        started_at: sessionStartedAt,
        lesson_id: lessonSelect ? lessonSelect.value || null : null,
      }),
    }).catch(() => {/* fail silently, don't block the UI */});
  }

  startBtn.addEventListener('click', startTimer);
  pauseBtn.addEventListener('click', pauseTimer);
  resetBtn.addEventListener('click', resetTimer);

  updateDisplay();
})();
