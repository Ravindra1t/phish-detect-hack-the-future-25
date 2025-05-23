// Background script - handles URL scanning
const API_BASE = 'http://localhost:5000/api';

// Scan URL when tab is updated
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    // Only scan HTTP/HTTPS URLs
    if (tab.url.startsWith('http://') || tab.url.startsWith('https://')) {
      await scanCurrentUrl(tab.url, tabId);
    }
  }
});

// Main scanning function
async function scanCurrentUrl(url, tabId) {
  try {
    console.log('🔍 Scanning URL:', url);
    
    const response = await fetch(`${API_BASE}/scan-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    console.log('📊 Scan result:', result);
    
    // Store result for popup
    await chrome.storage.local.set({
      [`scan_${tabId}`]: {
        url: url,
        result: result,
        timestamp: Date.now()
      }
    });

    // Update badge and show warning if needed
    if (result.is_phishing) {
      // Red badge for phishing
      chrome.action.setBadgeText({
        text: '!',
        tabId: tabId
      });
      chrome.action.setBadgeBackgroundColor({
        color: '#FF0000',
        tabId: tabId
      });
      
      // Send warning to content script
      chrome.tabs.sendMessage(tabId, {
        action: 'showWarning',
        data: result
      }).catch(err => console.log('Could not send message to content script'));
      
    } else {
      // Green badge for safe
      chrome.action.setBadgeText({
        text: '✓',
        tabId: tabId
      });
      chrome.action.setBadgeBackgroundColor({
        color: '#4CAF50',
        tabId: tabId
      });
    }

  } catch (error) {
    console.error('❌ Error scanning URL:', error);
    
    // Gray badge for error
    chrome.action.setBadgeText({
      text: '?',
      tabId: tabId
    });
    chrome.action.setBadgeBackgroundColor({
      color: '#757575',
      tabId: tabId
    });
  }
}

// Clear badge when tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.local.remove(`scan_${tabId}`);
});
