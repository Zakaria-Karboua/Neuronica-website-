(function () {
  const root = document.documentElement;
  const stored = localStorage.getItem('neuronica-theme');
  const preferred = stored || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  root.setAttribute('data-theme', preferred);

  document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('theme-toggle');
    if (!toggle) return;
    updateLabel(toggle, preferred);

    toggle.addEventListener('click', () => {
      const current = root.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem('neuronica-theme', next);
      updateLabel(toggle, next);
    });
  });

  function updateLabel(el, mode) {
    el.textContent = mode === 'dark' ? '🌙 Dark' : '☀️ Light';
  }
})();
