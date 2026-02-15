document.addEventListener('DOMContentLoaded', async () => {
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const apiUrlInput = document.getElementById('apiUrl');
  const apiUrlChapter = document.getElementById('apiUrlChapter');
  const saveBtn = document.getElementById('saveBtn');
  const modeTabs = document.querySelectorAll('.mode-tab');
  const modeContents = document.querySelectorAll('.mode-content');
  const detectBtn = document.getElementById('detectBtn');
  const detectedInfo = document.getElementById('detectedInfo');
  const detectedCount = document.getElementById('detectedCount');
  const imageGrid = document.getElementById('imageGrid');
  const colorizeAllBtn = document.getElementById('colorizeAllBtn');
  const progressContainer = document.getElementById('progressContainer');
  const progressPhase = document.getElementById('progressPhase');
  const progressPercent = document.getElementById('progressPercent');
  const progressFill = document.getElementById('progressFill');
  const progressText = document.getElementById('progressText');
  const chapterIdBox = document.getElementById('chapterIdBox');
  const chapterIdSpan = document.getElementById('chapterId');
  const downloadBtn = document.getElementById('downloadBtn');
  const errorMessage = document.getElementById('errorMessage');

  let detectedImages = [];
  let currentChapterId = null;
  let statusCheckInterval = null;
  let currentTabId = null;

  // STATE MANAGEMENT
  async function saveState(state) {
    const data = {
      timestamp: Date.now(),
      chapterId: currentChapterId,
      ...state
    };
    await chrome.storage.local.set({ 'processingState': data });
  }

  async function clearState() {
    await chrome.storage.local.remove('processingState');
    currentChapterId = null;
    if (statusCheckInterval) clearInterval(statusCheckInterval);
    resetUI();
  }

  async function restoreState() {
    const result = await chrome.storage.local.get('processingState');
    const state = result.processingState;
    console.log('[Popup] Restoring state:', state);

    if (state && state.chapterId) {
      // Verifica se o estado é recente (< 24h)
      const ageHours = (Date.now() - state.timestamp) / (1000 * 60 * 60);
      if (ageHours > 24) {
        console.log('[Popup] Estado muito antigo, descartando');
        await clearState();
        return;
      }

      currentChapterId = state.chapterId;
      chapterIdSpan.textContent = currentChapterId;
      chapterIdBox.style.display = 'block';

      // Restaura UI básica
      if (state.phase === 'analyzing' || state.phase === 'generating') {
        progressContainer.style.display = 'block';
        colorizeAllBtn.disabled = true;
        colorizeAllBtn.innerHTML = state.phase === 'analyzing' ? 'Analisando...' : 'Gerando...';

        // Retoma polling
        startStatusCheck(currentChapterId, apiUrlChapter.value || 'http://localhost:8000');
      } else if (state.phase === 'completed') {
        progressContainer.style.display = 'block';
        progressFill.style.width = '100%';
        progressPercent.textContent = '100%';
        colorizeAllBtn.textContent = 'Concluido';
        downloadBtn.style.display = 'block';
      }
    }
  }

  function resetUI() {
    progressContainer.style.display = 'none';
    chapterIdBox.style.display = 'none';
    colorizeAllBtn.disabled = false;
    colorizeAllBtn.textContent = 'Colorir Todas';
    colorizeAllBtn.innerHTML = 'Colorir Todas';
    downloadBtn.style.display = 'none';
    detectedInfo.style.display = 'none';
    if (statusCheckInterval) clearInterval(statusCheckInterval);
  }

  modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const mode = tab.dataset.mode;
      modeTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      modeContents.forEach(c => c.classList.remove('active'));
      document.getElementById(mode + '-mode').classList.add('active');
    });
  });

  async function checkStatus() {
    try {
      const response = await chrome.runtime.sendMessage({ action: 'checkStatus' });
      if (response && response.online) {
        statusDot.classList.add('online');
        statusText.textContent = 'Online';
        return true;
      }
    } catch (e) {
      console.error('Status check failed:', e);
    }
    statusDot.classList.remove('online');
    statusText.textContent = 'Offline';
    return false;
  }

  checkStatus();
  setInterval(checkStatus, 5000);

  apiUrlInput.addEventListener('change', () => apiUrlChapter.value = apiUrlInput.value);
  apiUrlChapter.addEventListener('change', () => apiUrlInput.value = apiUrlChapter.value);

  saveBtn.addEventListener('click', async () => {
    await chrome.storage.local.set({ apiUrl: apiUrlInput.value });
    saveBtn.textContent = 'Salvo!';
    setTimeout(() => saveBtn.textContent = 'Salvar Config', 1500);
    checkStatus();
  });

  const saved = await chrome.storage.local.get(['apiUrl', 'textCompositing', 'maxQuality', 'stylePreset']);
  if (saved.apiUrl) {
    apiUrlInput.value = saved.apiUrl;
    apiUrlChapter.value = saved.apiUrl;
  }

  // Carrega preferência de text compositing (padrão: false)
  const textCompositingCheck = document.getElementById('textCompositingCheck');
  if (textCompositingCheck) {
    textCompositingCheck.checked = saved.textCompositing === true;

    // Salva alterações automaticamente
    textCompositingCheck.addEventListener('change', async () => {
      await chrome.storage.local.set({ textCompositing: textCompositingCheck.checked });
      console.log('[MangaAutoColor] Text compositing:', textCompositingCheck.checked);
    });
  }

  // Carrega preferência de máxima qualidade (padrão: true)
  const maxQualityCheck = document.getElementById('maxQualityCheck');
  if (maxQualityCheck) {
    maxQualityCheck.checked = saved.maxQuality !== false;  // padrão: true

    // Salva alterações automaticamente
    maxQualityCheck.addEventListener('change', async () => {
      await chrome.storage.local.set({ maxQuality: maxQualityCheck.checked });
      console.log('[MangaAutoColor] Max quality:', maxQualityCheck.checked);
    });
  }

  // Carrega preferência de estilo (padrão: default)
  const stylePresetSelect = document.getElementById('stylePresetSelect');
  if (stylePresetSelect) {
    stylePresetSelect.value = saved.stylePreset || 'default';

    // Salva alterações automaticamente
    stylePresetSelect.addEventListener('change', async () => {
      await chrome.storage.local.set({ stylePreset: stylePresetSelect.value });
      console.log('[MangaAutoColor] Style preset:', stylePresetSelect.value);
    });
  }

  // Gerenciamento de imagens de referência coloridas
  const colorReferencesInput = document.getElementById('colorReferencesInput');
  const uploadRefBtn = document.getElementById('uploadRefBtn');
  const referencePreview = document.getElementById('referencePreview');
  const refCount = document.getElementById('refCount');
  const refThumbs = document.getElementById('refThumbs');
  const clearRefBtn = document.getElementById('clearRefBtn');

  let colorReferenceFiles = [];

  if (uploadRefBtn && colorReferencesInput) {
    uploadRefBtn.addEventListener('click', () => {
      colorReferencesInput.click();
    });

    colorReferencesInput.addEventListener('change', (e) => {
      const files = Array.from(e.target.files);
      if (files.length > 0) {
        colorReferenceFiles = [...colorReferenceFiles, ...files];
        updateReferencePreview();
        console.log('[MangaAutoColor] Referências adicionadas:', files.length);
      }
    });
  }

  if (clearRefBtn) {
    clearRefBtn.addEventListener('click', () => {
      colorReferenceFiles = [];
      colorReferencesInput.value = '';
      updateReferencePreview();
      console.log('[MangaAutoColor] Referências limpas');
    });
  }

  // --- Botão de Reset Manual ---
  const headerTitle = document.querySelector('.header h1');
  if (headerTitle) {
    const resetBtn = document.createElement('span');
    resetBtn.innerHTML = '↺';
    resetBtn.title = 'Resetar Estado';
    resetBtn.style.cursor = 'pointer';
    resetBtn.style.fontSize = '12px';
    resetBtn.style.marginLeft = '10px';
    resetBtn.style.color = '#e74c3c';
    resetBtn.onclick = async (e) => {
      e.stopPropagation();
      if (confirm('Resetar estado e limpar progresso atual?')) {
        await clearState();
        location.reload();
      }
    };
    headerTitle.appendChild(resetBtn);
  }

  // Restaura estado ao iniciar
  restoreState();

  function updateReferencePreview() {
    if (colorReferenceFiles.length === 0) {
      referencePreview.style.display = 'none';
      return;
    }

    referencePreview.style.display = 'block';
    refCount.textContent = colorReferenceFiles.length;
    refThumbs.innerHTML = '';

    colorReferenceFiles.forEach((file, index) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const thumb = document.createElement('div');
        thumb.style.cssText = 'width: 40px; height: 40px; border-radius: 4px; overflow: hidden; position: relative;';
        thumb.innerHTML = `
          <img src="${e.target.result}" style="width: 100%; height: 100%; object-fit: cover;">
          <button data-index="${index}" class="remove-ref" style="position: absolute; top: -2px; right: -2px; 
                  width: 14px; height: 14px; background: #e74c3c; border: none; border-radius: 50%; 
                  color: white; font-size: 8px; cursor: pointer; line-height: 1;">×</button>
        `;
        refThumbs.appendChild(thumb);

        thumb.querySelector('.remove-ref').addEventListener('click', (evt) => {
          evt.stopPropagation();
          const idx = parseInt(evt.target.dataset.index);
          colorReferenceFiles.splice(idx, 1);
          updateReferencePreview();
        });
      };
      reader.readAsDataURL(file);
    });
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.add('active');
  }

  function hideError() {
    errorMessage.classList.remove('active');
  }

  // Progress listener from content script
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'downloadProgress') {
      progressText.textContent = 'Baixando imagem ' + request.current + '/' + request.total + '...';
      const pct = Math.round((request.current / request.total) * 20);
      progressFill.style.width = pct + '%';
    }
  });

  detectBtn.addEventListener('click', async () => {
    hideError();
    detectBtn.disabled = true;
    detectBtn.innerHTML = 'Detectando...';

    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) throw new Error('Nenhuma aba ativa');
      currentTabId = tab.id;

      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['content_script.js']
      });

      await new Promise(r => setTimeout(r, 200));

      const results = await chrome.tabs.sendMessage(tab.id, { action: 'detectImages' });

      if (!results || results.length === 0) {
        showError('Nenhuma imagem detectada');
        detectedInfo.style.display = 'none';
        colorizeAllBtn.disabled = true;
        return;
      }

      detectedImages = results;
      detectedCount.textContent = detectedImages.length;
      imageGrid.innerHTML = detectedImages.slice(0, 12).map(img =>
        '<div class="image-thumb"><img src="' + img.src + '" loading="lazy"></div>'
      ).join('');

      if (detectedImages.length > 12) {
        imageGrid.innerHTML += '<div class="image-thumb" style="display:flex;align-items:center;justify-content:center;font-size:10px;color:#888;">+' + (detectedImages.length - 12) + '</div>';
      }

      detectedInfo.style.display = 'block';
      colorizeAllBtn.disabled = false;

    } catch (error) {
      showError('Erro: ' + error.message);
    } finally {
      detectBtn.disabled = false;
      detectBtn.textContent = 'Detectar Imagens';
    }
  });

  colorizeAllBtn.addEventListener('click', async () => {
    if (detectedImages.length === 0 || !currentTabId) return;

    const apiUrl = apiUrlChapter.value || 'http://localhost:8000';

    hideError();
    colorizeAllBtn.disabled = true;
    colorizeAllBtn.innerHTML = 'Baixando...';
    progressContainer.style.display = 'block';
    progressPhase.textContent = 'Passo 1/2';
    progressText.textContent = 'Baixando imagens da pagina...';

    try {
      // Pede ao content script para baixar as imagens
      const downloadResult = await chrome.tabs.sendMessage(currentTabId, {
        action: 'downloadImages',
        images: detectedImages
      });

      if (!downloadResult.success || !downloadResult.images || downloadResult.images.length === 0) {
        throw new Error('Nao foi possivel baixar as imagens');
      }

      console.log('[MangaAutoColor] Downloaded', downloadResult.images.length, 'images');

      // Converte base64 para blobs
      progressText.textContent = 'Preparando envio...';
      const formData = new FormData();

      for (let i = 0; i < downloadResult.images.length; i++) {
        const img = downloadResult.images[i];
        // Converte base64 data URL para blob
        const response = await fetch(img.data);
        const blob = await response.blob();
        formData.append('files', blob, img.name);
      }

      // Adiciona imagens de referência coloridas (se houver)
      if (colorReferenceFiles.length > 0) {
        console.log('[MangaAutoColor] Enviando', colorReferenceFiles.length, 'imagens de referência');
        for (let i = 0; i < colorReferenceFiles.length; i++) {
          formData.append('color_references', colorReferenceFiles[i], colorReferenceFiles[i].name);
        }
        progressText.textContent = `Preparando envio (${colorReferenceFiles.length} referências)...`;
      }

      // Envia para o backend
      progressText.textContent = 'Enviando para analise...';
      const xhr = new XMLHttpRequest();
      const analyzeData = await new Promise((resolve, reject) => {
        xhr.open('POST', apiUrl + '/chapter/analyze', true);
        xhr.onload = () => {
          if (xhr.status === 200) {
            try { resolve(JSON.parse(xhr.responseText)); }
            catch (e) { reject(new Error('Resposta invalida')); }
          } else {
            reject(new Error('Erro ' + xhr.status + ': ' + xhr.responseText));
          }
        };
        xhr.onerror = () => reject(new Error('Erro de conexao'));
        xhr.send(formData);
      });

      currentChapterId = analyzeData.chapter_id;
      chapterIdSpan.textContent = currentChapterId;
      chapterIdBox.style.display = 'block';

      startStatusCheck(currentChapterId, apiUrl);

    } catch (error) {
      console.error('[MangaAutoColor] Error:', error);
      showError('Erro: ' + error.message);
      colorizeAllBtn.disabled = false;
      colorizeAllBtn.textContent = 'Colorir Todas';
      await clearState(); // Limpa estado em caso de erro fatal no setup
    }
  });


  async function updateState(statusData, phase) {
    await saveState({
      phase: phase, // 'analyzing', 'generating', 'completed', 'error'
      progress: statusData.progress || 0,
      total_pages: statusData.total_pages || 0,
      message: statusData.message
    });
  }

  function startStatusCheck(chapterId, apiUrl) {
    if (statusCheckInterval) clearInterval(statusCheckInterval);
    let pass1Complete = false;

    statusCheckInterval = setInterval(async () => {
      try {
        const resp = await fetch(apiUrl + '/chapter/' + chapterId + '/status');
        const data = await resp.json();

        // SAVE STATE ON TICK
        let phase = 'analyzing';
        if (pass1Complete) phase = 'generating';
        if (data.status === 'completed') phase = 'completed';
        if (data.status === 'error') phase = 'error';

        await updateState(data, phase);

        if (data.total_pages > 0) {
          const base = pass1Complete ? 50 : 0;
          const pct = Math.round(base + (data.progress / data.total_pages) * 50);
          progressFill.style.width = pct + '%';
          progressPercent.textContent = pct + '%';
        }
        progressText.textContent = data.message;

        if (data.status === 'analyzed' && !pass1Complete) {
          pass1Complete = true;
          progressPhase.textContent = 'Passo 2/2';

          await updateState(data, 'generating');

          // Obtém preferências
          const textCompositing = document.getElementById('textCompositingCheck')?.checked || false;
          const maxQuality = document.getElementById('maxQualityCheck')?.checked !== false;  // padrão: true
          const stylePreset = document.getElementById('stylePresetSelect')?.value || 'default';

          fetch(apiUrl + '/chapter/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              chapter_id: chapterId,
              page_numbers: null,
              options: {},
              text_compositing: textCompositing,
              max_quality: maxQuality,
              style_preset: stylePreset
            })
          });
        } else if (data.status === 'completed') {
          clearInterval(statusCheckInterval);
          await updateState(data, 'completed');

          progressFill.style.width = '100%';
          progressPercent.textContent = '100%';
          colorizeAllBtn.textContent = 'Concluido';
          downloadBtn.style.display = 'block';

          // Substituir imagens no navegador
          await replaceImagesInBrowser(data);

          // Clear state after success? No, user might want to see 'Completed' until they reset.
          // But maybe we should allow them to download first.
        } else if (data.status === 'error') {
          clearInterval(statusCheckInterval);
          await updateState(data, 'error');
          showError('Erro: ' + data.message);
          colorizeAllBtn.disabled = false;
          colorizeAllBtn.textContent = 'Tentar Novamente';
        }
      } catch (e) {
        console.error('Status check error:', e);
      }
    }, 2000);
  }

  downloadBtn.addEventListener('click', async () => {
    if (!currentChapterId) return;
    const apiUrl = apiUrlChapter.value || 'http://localhost:8000';
    downloadBtn.textContent = 'Baixando...';

    try {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', apiUrl + '/chapter/' + currentChapterId + '/download', true);
      xhr.responseType = 'blob';

      const blob = await new Promise((resolve, reject) => {
        xhr.onload = () => xhr.status === 200 ? resolve(xhr.response) : reject(new Error('Download falhou'));
        xhr.onerror = () => reject(new Error('Erro de rede'));
        xhr.send();
      });

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'manga_' + currentChapterId + '.zip';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      downloadBtn.textContent = 'Baixado!';
    } catch (error) {
      showError('Erro no download: ' + error.message);
      downloadBtn.textContent = 'Download ZIP';
    }
  });

  // Funcao para substituir imagens colorizadas no navegador
  async function replaceImagesInBrowser(statusData) {
    try {
      colorizeAllBtn.textContent = 'Atualizando pagina...';

      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) return;

      // Buscar URLs das imagens colorizadas do backend
      const apiUrl = apiUrlChapter.value || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/chapter/${currentChapterId}/colored-images`);

      if (!response.ok) {
        console.warn('[Popup] Nao foi possivel buscar imagens colorizadas');
        return;
      }

      const coloredData = await response.json();

      if (!coloredData.images || coloredData.images.length === 0) {
        console.warn('[Popup] Nenhuma imagem colorizada encontrada');
        return;
      }

      console.log('[Popup] =======================================');
      console.log('[Popup] IMAGENS DETECTADAS:', detectedImages.length);
      console.log('[Popup] IMAGENS COLORIZADAS:', coloredData.images.length);
      console.log('[Popup] =======================================');

      // Criar mapa: src_original -> data_url_colorizada
      // Assume que a ordem é preservada (primeira detectada -> primeira colorizada)
      const imageMap = {};
      const count = Math.min(detectedImages.length, coloredData.images.length);

      console.log('[Popup] Criando mapeamento para', count, 'imagens...');

      for (let i = 0; i < count; i++) {
        const originalSrc = detectedImages[i].src;
        const colorizedDataUrl = `data:image/png;base64,${coloredData.images[i].data}`;
        imageMap[originalSrc] = colorizedDataUrl;

        // Log todas as entradas para debug
        console.log(`[Popup] Mapeamento [${i}]:`);
        console.log(`  Original: ${originalSrc.substring(0, 70)}...`);
        console.log(`  Colorizada: ${colorizedDataUrl.substring(0, 50)}...`);
      }

      console.log('[Popup] Total no mapa:', Object.keys(imageMap).length);

      // Primeiro, injeta o content script se necessário
      try {
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ['content_script.js']
        });
        await new Promise(r => setTimeout(r, 200));
      } catch (e) {
        console.log('[Popup] Content script já injetado ou erro:', e.message);
      }

      // Enviar para content script substituir
      await chrome.tabs.sendMessage(tab.id, {
        action: 'replaceImages',
        imageMap: imageMap,
        mappings: coloredData.mappings || []
      });

      colorizeAllBtn.textContent = 'Pagina Colorida!';
      console.log('[Popup] Imagens substituidas com sucesso');

    } catch (error) {
      console.error('[Popup] Erro ao substituir imagens:', error);
      colorizeAllBtn.textContent = 'Concluido (sem preview)';
    }
  }

  window.addEventListener('unload', () => {
    if (statusCheckInterval) clearInterval(statusCheckInterval);
  });
});
