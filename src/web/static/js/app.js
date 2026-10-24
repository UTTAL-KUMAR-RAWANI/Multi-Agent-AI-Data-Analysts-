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

    // ─── Global Provider Badge ────────────────────────────
    initProviderBadge();

    // ─── AI Copilot Slide-Out Sidebar ──────────────────────
    initCopilotEvents();
});

function initProviderBadge() {
    const badge = document.getElementById('provider-badge');
    if (!badge) return;

    fetch('/health')
        .then(r => r.json())
        .then(data => {
            const displayProviders = (data.providers || [])
                .filter(p => p.toLowerCase() !== 'gemini')
                .join(', ');
            badge.innerHTML = `
                <div class="provider-dot provider-dot-active"></div>
                <span>${displayProviders || data.default_provider || 'llamafile'}</span>
            `;
        })
        .catch(() => {
            badge.innerHTML = `
                <div class="provider-dot provider-dot-error"></div>
                <span>Disconnected</span>
            `;
        });
}

function initPageTransitions() {
    // Respect prefers-reduced-motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return;
    }

    const contentArea = document.querySelector('.content-area');
    if (!contentArea) return;

    // Ensure content area is ready and smooth
    contentArea.style.willChange = 'opacity';
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

// ─── Global AI Copilot Slide-Out Sidebar ────────────────
let isCopilotProcessing = false;
let copilotHistory = [];

function initCopilotEvents() {
    const copilotInput = document.getElementById('copilot-input');
    if (copilotInput) {
        copilotInput.addEventListener('input', () => {
            copilotInput.style.height = 'auto';
            copilotInput.style.height = Math.min(copilotInput.scrollHeight, 140) + 'px';
        });

        copilotInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendCopilotMessage();
            }
        });
    }

    // Escape key closes Copilot sidebar
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const card = document.getElementById('dashboard-copilot-card');
            if (card && card.classList.contains('active')) {
                closeCopilot();
            }
        }
    });
}

function toggleCopilot() {
    const card = document.getElementById('dashboard-copilot-card');
    const backdrop = document.getElementById('copilot-sidebar-backdrop');
    const headerBtn = document.getElementById('btn-copilot-header');
    const copilotInput = document.getElementById('copilot-input');
    if (!card) return;

    const isOpen = card.classList.toggle('active');
    if (backdrop) backdrop.classList.toggle('active', isOpen);
    if (headerBtn) {
        headerBtn.classList.toggle('active', isOpen);
        headerBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    }

    if (isOpen) {
        if (copilotInput) setTimeout(() => copilotInput.focus(), 250);
    }
    if (window.lucide) lucide.createIcons();
}

function closeCopilot() {
    const card = document.getElementById('dashboard-copilot-card');
    const backdrop = document.getElementById('copilot-sidebar-backdrop');
    const headerBtn = document.getElementById('btn-copilot-header');
    if (!card) return;

    card.classList.remove('active');
    if (backdrop) backdrop.classList.remove('active');
    if (headerBtn) {
        headerBtn.classList.remove('active');
        headerBtn.setAttribute('aria-expanded', 'false');
        headerBtn.focus();
    }
    if (window.lucide) lucide.createIcons();
}

function openCopilot() {
    const card = document.getElementById('dashboard-copilot-card');
    if (!card) return;
    if (!card.classList.contains('active')) {
        toggleCopilot();
    } else {
        const copilotInput = document.getElementById('copilot-input');
        if (copilotInput) copilotInput.focus();
    }
}

function focusCopilotChat() {
    openCopilot();
}

function askCopilot(question) {
    openCopilot();
    const copilotInput = document.getElementById('copilot-input');
    if (copilotInput) {
        copilotInput.value = question;
        sendCopilotMessage();
    }
}

function clearCopilot() {
    copilotHistory = [];
    const copilotMessages = document.getElementById('copilot-messages');
    if (!copilotMessages) return;
    copilotMessages.innerHTML = `
        <div class="copilot-msg assistant">
            <div class="copilot-msg-avatar">
                <i data-lucide="brain-circuit"></i>
            </div>
            <div class="copilot-msg-content">
                <div class="copilot-msg-header">
                    <span class="copilot-msg-sender">DataWhale Copilot</span>
                </div>
                <div class="copilot-msg-bubble">
                    <p>Conversation reset. I'm ready for any question about your dashboard, datasets, or AI specialists.</p>
                    <div class="copilot-chips-row" id="copilot-quick-chips">
                        <button type="button" class="copilot-chip" onclick="askCopilot('Summarize all my loaded datasets')">
                            <i data-lucide="sparkles"></i> Summarize all datasets
                        </button>
                        <button type="button" class="copilot-chip" onclick="askCopilot('Which datasets have missing values or quality issues?')">
                            <i data-lucide="scan-search"></i> Check missing data
                        </button>
                        <button type="button" class="copilot-chip" onclick="askCopilot('What do the 9 AI specialist agents do?')">
                            <i data-lucide="network"></i> Explain AI agents
                        </button>
                        <button type="button" class="copilot-chip" onclick="askCopilot('How do I run an analysis in Studio?')">
                            <i data-lucide="zap"></i> How to run Studio analysis
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    if (window.lucide) lucide.createIcons();
    showToast('Copilot conversation reset', 'info');
}

function formatMarkdown(text) {
    if (!text) return '';
    let escaped = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Pre/code blocks
    escaped = escaped.replace(/```([a-z]*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // Inline code
    escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Headings
    escaped = escaped.replace(/^#### (.*$)/gim, '<h5 style="margin:6px 0; font-weight:700;">$1</h5>');
    escaped = escaped.replace(/^### (.*$)/gim, '<h4 style="margin:8px 0 4px; font-weight:700;">$1</h4>');
    escaped = escaped.replace(/^## (.*$)/gim, '<h3 style="margin:10px 0 6px; font-weight:700;">$1</h3>');
    // Bold & Italics
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    escaped = escaped.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // Markdown Links: [text](url)
    escaped = escaped.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
    // List items
    escaped = escaped.replace(/^\s*[-*]\s+(.*$)/gim, '<li>$1</li>');
    escaped = escaped.replace(/(<li>.*<\/li>)/gms, '<ul style="margin: 6px 0 8px 18px; padding-left: 0;">$1</ul>');
    escaped = escaped.replace(/<\/ul>\s*<ul[^>]*>/g, '');

    // Paragraphs
    const blocks = escaped.split(/\n{2,}/);
    return blocks.map(b => {
        const trimmed = b.trim();
        if (!trimmed) return '';
        if (trimmed.startsWith('<h') || trimmed.startsWith('<ul') || trimmed.startsWith('<pre')) {
            return trimmed;
        }
        return `<p style="margin-bottom: 0.5rem; line-height: 1.6;">${trimmed.replace(/\n/g, '<br>')}</p>`;
    }).join('');
}

function addCopilotLoadingIndicator() {
    const copilotMessages = document.getElementById('copilot-messages');
    if (!copilotMessages) return;
    const div = document.createElement('div');
    div.className = 'copilot-msg assistant';
    div.id = 'copilot-loading-msg';
    div.innerHTML = `
        <div class="copilot-msg-avatar">
            <i data-lucide="brain-circuit"></i>
        </div>
        <div class="copilot-msg-content">
            <div class="copilot-msg-header">
                <span class="copilot-msg-sender">DataWhale Copilot</span>
            </div>
            <div class="copilot-msg-bubble">
                <div class="typing-indicator" style="display: flex; gap: 4px; padding: 4px 0;">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    copilotMessages.appendChild(div);
    if (window.lucide) lucide.createIcons();
    copilotMessages.scrollTop = copilotMessages.scrollHeight;
}

function removeCopilotLoadingIndicator() {
    const el = document.getElementById('copilot-loading-msg');
    if (el) el.remove();
}

function appendCopilotMessage(role, rawContent, actions = [], quickReplies = []) {
    const copilotMessages = document.getElementById('copilot-messages');
    if (!copilotMessages) return;

    const div = document.createElement('div');
    div.className = `copilot-msg ${role}`;

    const avatarIcon = role === 'user' ? 'user' : 'brain-circuit';
    const sender = role === 'user' ? 'You' : 'DataWhale Copilot';
    const formattedBody = role === 'user'
        ? `<p>${rawContent.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</p>`
        : formatMarkdown(rawContent);

    let actionsHtml = '';
    if (actions && actions.length > 0) {
        actionsHtml = `
            <div class="copilot-actions-row">
                ${actions.map(act => `
                    <a href="${act.url}" class="copilot-action-pill" id="action-${act.label.toLowerCase().replace(/[^a-z0-9]/g, '-')}">
                        <i data-lucide="${act.icon || 'arrow-right'}"></i>
                        <span>${act.label}</span>
                    </a>
                `).join('')}
            </div>
        `;
    }

    let quickRepliesHtml = '';
    if (quickReplies && quickReplies.length > 0) {
        quickRepliesHtml = `
            <div class="copilot-chips-row">
                ${quickReplies.map(reply => `
                    <button type="button" class="copilot-chip" onclick="askCopilot('${reply.replace(/'/g, "\\'")}')">
                        <i data-lucide="sparkles"></i>
                        <span>${reply}</span>
                    </button>
                `).join('')}
            </div>
        `;
    }

    div.innerHTML = `
        <div class="copilot-msg-avatar">
            <i data-lucide="${avatarIcon}"></i>
        </div>
        <div class="copilot-msg-content">
            <div class="copilot-msg-header">
                <span class="copilot-msg-sender">${sender}</span>
            </div>
            <div class="copilot-msg-bubble">
                ${formattedBody}
                ${actionsHtml}
                ${quickRepliesHtml}
            </div>
        </div>
    `;

    copilotMessages.appendChild(div);
    if (window.lucide) lucide.createIcons();
    copilotMessages.scrollTop = copilotMessages.scrollHeight;
}

async function sendCopilotMessage() {
    const copilotInput = document.getElementById('copilot-input');
    const copilotSendBtn = document.getElementById('btn-copilot-send');
    if (!copilotInput || isCopilotProcessing) return;

    const message = copilotInput.value.trim();
    if (!message) return;

    isCopilotProcessing = true;
    if (copilotSendBtn) copilotSendBtn.disabled = true;
    copilotInput.value = '';
    copilotInput.style.height = 'auto';

    // Render user message
    appendCopilotMessage('user', message);
    copilotHistory.push({ role: 'user', content: message });

    // Show typing indicator
    addCopilotLoadingIndicator();

    try {
        const match = window.location.pathname.match(/\/(?:analysis|chat)\/([a-zA-Z0-9_\-]+)/);
        const activeDatasetId = match ? match[1] : null;

        const payload = {
            message: message,
            history: copilotHistory.slice(-6),
        };
        if (activeDatasetId) {
            payload.dataset_id = activeDatasetId;
        }

        const resp = await fetch('/api/dashboard/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        removeCopilotLoadingIndicator();

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            appendCopilotMessage(
                'assistant',
                `Error: ${errData.detail || 'Unable to process question at this moment.'}`
            );
        } else {
            const data = await resp.json();
            appendCopilotMessage(
                'assistant',
                data.reply,
                data.suggested_actions || [],
                data.quick_replies || []
            );
            copilotHistory.push({ role: 'assistant', content: data.reply });
        }
    } catch (error) {
        removeCopilotLoadingIndicator();
        appendCopilotMessage('assistant', `Connection error: ${error.message}. Please verify the local AI service is running.`);
    } finally {
        isCopilotProcessing = false;
        if (copilotSendBtn) copilotSendBtn.disabled = false;
        if (copilotInput) copilotInput.focus();
    }
}


