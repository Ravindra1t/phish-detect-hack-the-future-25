// Content script - shows warnings on page
console.log('🛡️ Phishing Shield content script loaded');

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('📨 Message received:', message);
  
  if (message.action === 'showWarning') {
    showPhishingWarning(message.data);
  }
});



function showPhishingWarning(data) {
  // Remove existing warning if present
  const existingWarning = document.getElementById('phishing-shield-warning');
  if (existingWarning) {
    existingWarning.remove();
  }

  // Create warning overlay
  const warningDiv = document.createElement('div');
  warningDiv.id = 'phishing-shield-warning';
  warningDiv.innerHTML = `
    <div style="
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(244, 67, 54, 0.95);
      z-index: 999999;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: Arial, sans-serif;
    ">
      <div style="
        background: white;
        padding: 40px;
        border-radius: 12px;
        text-align: center;
        max-width: 500px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        animation: slideIn 0.3s ease-out;
      ">
        <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
        <h2 style="color: #d32f2f; margin-bottom: 20px; font-size: 24px;">
          Phishing Warning!
        </h2>
        <p style="margin-bottom: 20px; color: #333; font-size: 16px; line-height: 1.5;">
          This website may be attempting to steal your personal information.
          <br><br>
          <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%<br>
          <strong>Risk Level:</strong> ${data.risk_level}
        </p>
        <div style="margin-bottom: 20px; padding: 10px; background: #f5f5f5; border-radius: 6px;">
          <small style="color: #666; word-break: break-all;">${data.url}</small>
        </div>
        <button id="phishing-go-back" style="
          background: #4CAF50;
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 16px;
          margin-right: 10px;
          font-weight: bold;
        ">
          ← Go Back to Safety
        </button>
        <button id="phishing-continue" style="
          background: #757575;
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 16px;
        ">
          Continue Anyway
        </button>
      </div>
    </div>
  `;

  // Add CSS animation
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideIn {
      from { opacity: 0; transform: scale(0.8); }
      to { opacity: 1; transform: scale(1); }
    }
  `;
  document.head.appendChild(style);

  document.body.appendChild(warningDiv);

  // Add event listeners
  document.getElementById('phishing-go-back').addEventListener('click', () => {
    window.history.back();
  });

  document.getElementById('phishing-continue').addEventListener('click', () => {
    document.body.removeChild(warningDiv);
  });

  console.log('⚠️ Phishing warning displayed');
}
