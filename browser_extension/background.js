/**
 * MangaAutoColor Pro - Background Script
 */

const DEFAULT_API_URL = 'http://localhost:8000';

// Menu de contexto
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'mangaautocolor-root',
    title: '🎨 MangaAutoColor',
    contexts: ['image', 'page']
  });

  chrome.contextMenus.create({
    id: 'colorize-image',
    parentId: 'mangaautocolor-root',
    title: '🎨 Colorize this Image',
    contexts: ['image']
  });

  chrome.contextMenus.create({
    id: 'colorize-page',
    parentId: 'mangaautocolor-root',
    title: '📄 Colorize All Images',
    contexts: ['page']
  });

  console.log('[MangaAutoColor] Extension installed');
});

// Handler de cliques
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['content_script.js']
    });
  } catch (e) {}

  await new Promise(r => setTimeout(r, 100));

  if (info.menuItemId === 'colorize-image' && info.srcUrl) {
    chrome.tabs.sendMessage(tab.id, {
      action: 'colorizeSingle',
      imageUrl: info.srcUrl
    });
  } else if (info.menuItemId === 'colorize-page') {
    chrome.tabs.sendMessage(tab.id, { action: 'colorizeAll' });
  }
});

// Health check
async function checkApiStatus() {
  try {
    const result = await chrome.storage.local.get(['apiUrl']);
    const apiUrl = result.apiUrl || DEFAULT_API_URL;
    const response = await fetch(`${apiUrl}/health`, { method: 'GET' });
    return response.ok;
  } catch (error) {
    return false;
  }
}

// Message handler
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  (async () => {
    switch (request.action) {
      case 'checkStatus':
        sendResponse({ online: await checkApiStatus() });
        break;
        
      case 'getApiUrl':
        const result = await chrome.storage.local.get(['apiUrl']);
        sendResponse({ apiUrl: result.apiUrl || DEFAULT_API_URL });
        break;
        
      case 'setApiUrl':
        await chrome.storage.local.set({ apiUrl: request.apiUrl });
        sendResponse({ success: true });
        break;
        
      case 'colorizeSingle':
        await chrome.storage.local.set({ 
          selectedImage: request.imageSrc,
          colorizeMode: 'single'
        });
        sendResponse({ success: true });
        break;
        
      case 'colorizeAll':
        await chrome.storage.local.set({ colorizeMode: 'chapter' });
        sendResponse({ success: true });
        break;
    }
  })();
  return true;
});

console.log('[MangaAutoColor] Background script loaded');
