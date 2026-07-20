// Lightweight parallax starfield — no external library needed.
(function () {
  const canvas = document.getElementById('starfield');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let stars = [];

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    stars = [];
    const layers = [
      { count: 60, speed: 0.02, size: 1 },
      { count: 35, speed: 0.05, size: 1.6 },
      { count: 15, speed: 0.09, size: 2.2 },
    ];
    layers.forEach((layer) => {
      for (let i = 0; i < layer.count; i++) {
        stars.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          speed: layer.speed,
          size: layer.size,
          twinkle: Math.random() * Math.PI * 2,
        });
      }
    });
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const color = getComputedStyle(document.documentElement)
      .getPropertyValue('--accent-cyan').trim() || '#00d9ff';
    stars.forEach((s) => {
      s.y += s.speed;
      s.twinkle += 0.02;
      if (s.y > canvas.height) s.y = 0;
      const alpha = 0.5 + 0.5 * Math.sin(s.twinkle);
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha * 0.8;
      ctx.fill();
    });
    ctx.globalAlpha = 1;
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  resize();
  draw();
})();
