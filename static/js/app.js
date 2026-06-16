/* =========================================================
   EffViT-Hybrid — Frontend logic
   ========================================================= */
(() => {
  const $form        = document.getElementById('upload-form');
  const $fileInput   = document.getElementById('file-input');
  const $dropzone    = document.getElementById('dropzone');
  const $preview     = document.getElementById('dropzone-preview');
  const $submitBtn   = document.getElementById('submit-btn');
  const $resetBtn    = document.getElementById('reset-btn');

  const $loading     = document.getElementById('loading-panel');
  const $results     = document.getElementById('results');
  const $errorPanel  = document.getElementById('error-panel');
  const $errorBody   = document.getElementById('error-body');

  const $resultClass      = document.getElementById('result-class');
  const $resultConfidence = document.getElementById('result-confidence');
  const $probList         = document.getElementById('prob-list');
  const $resultOriginal   = document.getElementById('result-original');
  const $resultGradcam    = document.getElementById('result-gradcam');
  const $gradcamWrap      = document.getElementById('gradcam-wrap');

  let currentFile = null;

  // -------------------------------------------------------
  // File selection
  // -------------------------------------------------------
  function setFile(file) {
    currentFile = file;
    if (!file) {
      $preview.src = '';
      $dropzone.classList.remove('has-file');
      $submitBtn.disabled = true;
      $resetBtn.disabled = true;
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      $preview.src = e.target.result;
      $dropzone.classList.add('has-file');
    };
    reader.readAsDataURL(file);
    $submitBtn.disabled = false;
    $resetBtn.disabled = false;
  }

  $fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) setFile(file);
  });

  // Drag-and-drop
  ['dragenter', 'dragover'].forEach(evt => {
    $dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      $dropzone.classList.add('is-dragging');
    });
  });
  ['dragleave', 'drop'].forEach(evt => {
    $dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      $dropzone.classList.remove('is-dragging');
    });
  });
  $dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      $fileInput.files = e.dataTransfer.files;
      setFile(file);
    }
  });

  // -------------------------------------------------------
  // Submit
  // -------------------------------------------------------
  $form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!currentFile) return;

    // Reset visible state
    $errorPanel.hidden = true;
    $results.hidden = true;
    $loading.hidden = false;
    $submitBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
      const resp = await fetch('/predict', { method: 'POST', body: formData });
      const data = await resp.json();

      if (!resp.ok) {
        const msg = data.detail || `Server returned ${resp.status}`;
        showError(msg);
        return;
      }

      renderResults(data);
    } catch (err) {
      showError(`Network error: ${err.message}`);
    } finally {
      $loading.hidden = true;
      $submitBtn.disabled = false;
    }
  });

  $resetBtn.addEventListener('click', () => {
    setFile(null);
    $fileInput.value = '';
    $results.hidden = true;
    $errorPanel.hidden = true;
  });

  // -------------------------------------------------------
  // Render results
  // -------------------------------------------------------
  function renderResults(data) {
    $resultClass.textContent = data.predicted_class;
    $resultConfidence.textContent = (data.confidence * 100).toFixed(2) + '%';

    // Probability list
    $probList.innerHTML = '';
    data.probabilities.forEach((p, idx) => {
      const row = document.createElement('div');
      row.className = 'prob-row' + (idx === 0 ? ' prob-row--top' : '');
      row.innerHTML = `
        <div class="prob-top">
          <span class="prob-name">${escapeHtml(p.name)}</span>
          <span class="prob-value">${(p.prob * 100).toFixed(2)}%</span>
        </div>
        <div class="prob-bar">
          <div class="prob-bar-fill" style="width: 0%"></div>
        </div>
      `;
      $probList.appendChild(row);
      // Animate fill on next frame
      requestAnimationFrame(() => {
        row.querySelector('.prob-bar-fill').style.width = (p.prob * 100) + '%';
      });
    });

    $resultOriginal.src = data.original_url + '?t=' + Date.now();
    if (data.gradcam_url) {
      $resultGradcam.src = data.gradcam_url + '?t=' + Date.now();
      $gradcamWrap.parentElement.style.display = '';
    } else {
      $gradcamWrap.parentElement.style.display = 'none';
    }

    $results.hidden = false;
    $results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function showError(message) {
    $errorBody.textContent = message;
    $errorPanel.hidden = false;
    $errorPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
})();
