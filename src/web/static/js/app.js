/**
 * DataWhale AI — Core JavaScript
 * Handles sidebar toggle, page transitions, toasts, and utility functions.
 */

// ─── Sidebar Toggle ────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('sidebar');

    if (toggle && sidebar) {
        toggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });

        // Close sidebar on outside click (mobile)
        document.addEventListener('click', (e) => {
            if (
                window.innerWidth <= 768 &&
                sidebar.classList.contains('open') &&
                !sidebar.contains(e.target) &&
                !toggle.contains(e.target)
            ) {
                sidebar.classList.remove('open');
            }
        });
    }

    // ─── Page Navigation Transitions ─────────────────────
    initPageTransitions();
});

function initPageTransitions() {
    // Respect prefers-reduced-motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return;
    }

    const contentArea = document.querySelector('.content-area');
    if (!contentArea) return;

    // Ensure content area is ready and smooth
    contentArea.style.willChange = 'opacity, transform';
}

// Restore clean state on browser back/forward (bfcache)
window.addEventListener('pageshow', () => {
    const contentArea = document.querySelector('.content-area');
    if (contentArea) {
        contentArea.style.willChange = 'auto';
    }
});

// ─── Toast Notifications ───────────────────────────────
function showToast(message, type = 'success', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const iconName = type === 'success' ? 'check-circle-2' : type === 'error' ? 'alert-circle' : 'info';
    toast.innerHTML = `<i data-lucide="${iconName}" class="toast-icon"></i><span class="toast-message">${message}</span>`;

    container.appendChild(toast);
    if (window.lucide) {
        lucide.createIcons({
            root: toast,
        });
    }

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(6px)';
        toast.style.transition = 'opacity 160ms cubic-bezier(0.7, 0, 0.84, 0), transform 160ms cubic-bezier(0.7, 0, 0.84, 0)';
        setTimeout(() => toast.remove(), 170);
    }, duration);
}

// ─── Fetch Helper ──────────────────────────────────────
async function apiFetch(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        showToast(error.message, 'error');
        throw error;
    }
}

