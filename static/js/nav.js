(function () {
  function wireDropdown(toggleId, menuId) {
    const toggle = document.getElementById(toggleId);
    const menu = document.getElementById(menuId);
    if (!toggle || !menu) return;

    toggle.addEventListener('click', (e) => {
      e.stopPropagation();
      menu.classList.toggle('open');
    });

    document.addEventListener('click', (e) => {
      if (!menu.contains(e.target) && e.target !== toggle) {
        menu.classList.remove('open');
      }
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    wireDropdown('user-menu-toggle', 'user-menu');
    wireDropdown('login-menu-toggle', 'login-menu');
  });
})();
