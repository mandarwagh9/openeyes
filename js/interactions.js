/**
 * OpenEyes Framework v3.0.1 - Interaction Scripts
 * Premium Industrial Interactions: Film Grain, Spotlight, Scroll Reveals, Typing
 */

document.addEventListener('DOMContentLoaded', () => {

  /* 1. Film Grain & Spotlight Background */
  // Inject film grain overlay dynamically to avoid HTML bloat
  if (!document.querySelector('.film-grain')) {
    const grain = document.createElement('div');
    grain.classList.add('film-grain');
    document.body.appendChild(grain);
  }

  // Global mouse tracker for body spotlight
  window.addEventListener('mousemove', (e) => {
    document.body.style.setProperty('--mouse-x', `${e.clientX}px`);
    document.body.style.setProperty('--mouse-y', `${e.clientY}px`);
  });

  /* 2. Bento Spotlight Effect */
  const cards = document.querySelectorAll('.bento-card');
  cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      card.style.setProperty('--card-mouse-x', `${x}px`);
      card.style.setProperty('--card-mouse-y', `${y}px`);
    });
  });

  /* 3. Terminal Typing Effect (Only on index.html) */
  const terminal = document.querySelector('.terminal-body');
  if (terminal) {
    // Keep existing structure but animate lines
    const lines = terminal.querySelectorAll('div');
    let delay = 0;
    
    // Hide initially
    lines.forEach(line => {
      line.style.opacity = '0';
      line.style.display = 'none';
    });
    
    // Typing simulation
    const typeLine = (index) => {
      if (index >= lines.length) return;
      
      const line = lines[index];
      line.style.display = 'block';
      line.style.opacity = '1';
      
      // If the line is a command prompt, reveal it instantly
      if (line.textContent.includes('openeyes@v3')) {
         setTimeout(() => typeLine(index + 1), 300);
      } else {
         const originalHTML = line.innerHTML;
         const text = line.textContent; // pure text
         line.innerHTML = '';
         
         let charIndex = 0;
         const typeChar = () => {
             if (charIndex < text.length) {
                 line.innerHTML += text.charAt(charIndex);
                 charIndex++;
                 setTimeout(typeChar, 10 + Math.random() * 15); // Fast typing speed
             } else {
                 line.innerHTML = originalHTML; // Restore spans (like [OK])
                 setTimeout(() => typeLine(index + 1), 150 + Math.random() * 100);
             }
         };
         typeChar();
      }
    };
    
    // Start typing after a short delay
    setTimeout(() => typeLine(0), 500);
    
    // Simple blinking cursor logic for terminal
    const cursor = document.querySelector('.cursor');
    if (cursor) {
      setInterval(() => {
        cursor.style.opacity = cursor.style.opacity === '0' ? '1' : '0';
      }, 500);
    }
  }

  /* 4. Scroll Reveals (Intersection Observer) */
  const revealElements = document.querySelectorAll('.section, .bento-card, .spec-table tr, .docs-section');
  revealElements.forEach(el => el.classList.add('reveal')); // Add base reveal class

  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
        revealObserver.unobserve(entry.target); // Optional: Unobserve after reveal
      }
    });
  }, {
    threshold: 0.1, // Trigger when 10% visible
    rootMargin: "0px 0px -50px 0px"
  });

  document.querySelectorAll('.reveal').forEach(el => {
    revealObserver.observe(el);
  });

  /* 5. Docs ScrollSpy (Only on docs.html) */
  const docsSections = document.querySelectorAll('.docs-section');
  const sidebarLinks = document.querySelectorAll('.sidebar-nav a');
  
  if (docsSections.length > 0 && sidebarLinks.length > 0) {
    const scrollSpyObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute('id');
          // Remove active class from all links
          sidebarLinks.forEach(link => link.classList.remove('active'));
          // Add active class to corresponding link
          const activeLink = document.querySelector(`.sidebar-nav a[href="#${id}"]`);
          if (activeLink) {
            activeLink.classList.add('active');
          }
        }
      });
    }, {
      rootMargin: "-20% 0px -70% 0px", // Trigger slightly offset from top
      threshold: 0
    });
    
    docsSections.forEach(section => scrollSpyObserver.observe(section));
  }

});
