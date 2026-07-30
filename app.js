const citationStylesheet = document.createElement('link');
citationStylesheet.rel = 'stylesheet';
citationStylesheet.href = 'citation.css';
document.head.appendChild(citationStylesheet);

const toggle = document.querySelector('.nav-toggle');
const nav = document.querySelector('#site-nav');

if (toggle && nav) {
  toggle.addEventListener('click', () => {
    const isOpen = nav.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(isOpen));
  });

  nav.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      nav.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    });
  });
}

document.querySelectorAll('[data-copy-target]').forEach((button) => {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;

    const originalLabel = button.textContent;
    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      button.textContent = 'Copied';
    } catch (error) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Select and copy';
    }

    window.setTimeout(() => {
      button.textContent = originalLabel;
    }, 1800);
  });
});

const ANALYTICS_ENDPOINT = 'https://tgcm-analytics.mccsinicawebsite.workers.dev/collect';
const PAPER_URL_PREFIX = 'https://arxiv.org/abs/2606.18651';
const DATASET_FILE_ID = '1a9aYk9uXbi1I6miggYp2U1fT8HG6xNcZ';

function trackEvent(eventType) {
  const payload = JSON.stringify({
    event_type: eventType,
    page_path: window.location.pathname,
  });

  fetch(ANALYTICS_ENDPOINT, {
    method: 'POST',
    mode: 'cors',
    credentials: 'omit',
    cache: 'no-store',
    keepalive: true,
    headers: {
      'Content-Type': 'text/plain;charset=UTF-8',
    },
    body: payload,
  }).catch(() => {
    // Analytics must never interrupt navigation or page rendering.
  });
}

function recordPageView() {
  trackEvent('page_view');
}

if (document.visibilityState === 'prerender') {
  document.addEventListener(
    'visibilitychange',
    () => {
      if (document.visibilityState === 'visible') recordPageView();
    },
    { once: true },
  );
} else {
  recordPageView();
}

document.addEventListener('click', (event) => {
  const link = event.target.closest('a[href]');
  if (!link) return;

  const destination = link.href;
  if (destination.startsWith(PAPER_URL_PREFIX)) {
    trackEvent('paper_click');
  } else if (destination.includes(DATASET_FILE_ID)) {
    trackEvent('dataset_download_click');
  }
});
