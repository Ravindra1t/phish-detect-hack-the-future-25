// Popup script
const API_BASE = 'http://localhost:5000/api';

document.addEventListener('DOMContentLoaded', async () => {
  console.log('🎯 Popup loaded');
  
  await loadCurrentStatus();
  await loadStats();
  
  // Add event listeners
  document.getElementById('scan-now').addEventListener('click', scanCurrentSite);
  document.getElementById('report-issue').addEventListener('click', reportIssue);
});

async function loadCurrentStatus() {
  try {
    // Get current active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab || !tab.url) {
      updateStatus('No active tab', 'loading');
      return;
    }

    document.getElementById('current-url').textContent = tab.url;

    // Check if we have a recent scan for this tab
    const scanData = await chrome.storage.local.get(`scan_${tab.id}`);
    
    if (scanData[`scan_${tab.id}`]) {
      const data = scanData[`scan_${tab.id}`];
      
      if (data.result.is_phishing) {
        updateStatus(
          `⚠️ Phishing Detected (${(data.result.confidence * 100).toFixed(1)}% confidence)`,
          'danger'
        );
      } else {
        updateStatus('✅ Site appears safe', 'safe');
      }
    } else {
      updateStatus('🔍 Not yet scanned', 'loading');
    }

  } catch (error) {
    console.error('Error loading current status:', error);
    updateStatus('Error checking status', 'loading');
  }
}

async function loadStats() {
  try {
    const response = await fetch(`${API_BASE}/stats`);
    const stats = await response.json();
    
    document.getElementById('total-scans').textContent = stats.total_scans;
    document.getElementById('threats-blocked').textContent = stats.phishing_detected;
    document.getElementById('safe-sites').textContent = stats.safe_scans;
    
  } catch (error) {
    console.error('Error loading stats:', error);
    // Keep default values if API is not available
  }
}

function updateStatus(message, type) {
  const statusDiv = document.getElementById('current-status');
  statusDiv.innerHTML = `<strong>${message}</strong>`;
  
  // Update classes
  statusDiv.className = `status status-${type}`;
}

async function scanCurrentSite() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab || !tab.url) {
      alert('No active tab to scan');
      return;
    }

    updateStatus('🔍 Scanning...', 'loading');

    const response = await fetch(`${API_BASE}/scan-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: tab.url })
    });

    const result = await response.json();
    
    // Store result
    await chrome.storage.local.set({
      [`scan_${tab.id}`]: {
        url: tab.url,
        result: result,
        timestamp: Date.now()
      }
    });

    // Update display
    if (result.is_phishing) {
      updateStatus(
        `⚠️ Phishing Detected (${(result.confidence * 100).toFixed(1)}% confidence)`,
        'danger'
      );
    } else {
      updateStatus('✅ Site appears safe', 'safe');
    }

    // Refresh stats
    await loadStats();

  } catch (error) {
    console.error('Error scanning site:', error);
    updateStatus('❌ Scan failed', 'loading');
  }
}

function reportIssue() {
  alert('Report feature coming soon! For now, you can contact the developers.');
}
