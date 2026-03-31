// OpenEyes Website JavaScript

// Mobile menu toggle
function toggleMenu() {
  document.querySelector('.navbar-links').classList.toggle('active');
}

// Copy code functionality
function copyCode(btn) {
  const codeBlock = btn.closest('.code-block');
  const code = codeBlock.querySelector('code').textContent;
  
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => {
      btn.textContent = 'Copy';
    }, 2000);
  });
}

// Navbar scroll effect
window.addEventListener('scroll', () => {
  const navbar = document.querySelector('.navbar');
  if (window.scrollY > 50) {
    navbar.style.background = 'rgba(10, 10, 15, 0.95)';
  } else {
    navbar.style.background = 'rgba(10, 10, 15, 0.8)';
  }
});