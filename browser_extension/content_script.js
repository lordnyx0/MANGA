/**
 * MangaAutoColor Pro - Content Script
 */

(function() {
  'use strict';
  
  if (window.mangaAutoColorInjected) return;
  window.mangaAutoColorInjected = true;
  
  console.log('[MangaAutoColor] Content script active');
  
  // Função auxiliar para encontrar imagem correspondente no mapa
  function findMatchingImage(originalSrc, imageMap) {
    // Tenta match exato primeiro
    if (imageMap[originalSrc]) {
      return imageMap[originalSrc];
    }
    
    // Tenta match por filename
    const originalFilename = originalSrc.split('/').pop().split('?')[0];
    for (const [key, value] of Object.entries(imageMap)) {
      const keyFilename = key.split('/').pop().split('?')[0];
      if (keyFilename === originalFilename) {
        return value;
      }
    }
    
    return null;
  }
  
  function isMangaImage(img) {
    const width = img.naturalWidth || img.width || 0;
    const height = img.naturalHeight || img.height || 0;
    if (width < 200 || height < 200) return false;
    const ratio = height / width;
    if (ratio < 0.8 || ratio > 2.5) return false;
    return true;
  }
  
  function findMangaImages() {
    const results = [];
    const seen = new Set();
    const images = document.querySelectorAll('img');
    
    images.forEach(img => {
      const src = img.src || img.dataset?.src;
      if (!src || seen.has(src)) return;
      if (!isMangaImage(img)) return;
      
      seen.add(src);
      results.push({
        src: src,
        width: img.naturalWidth || img.width,
        height: img.naturalHeight || img.height,
        alt: img.alt || ''
      });
    });
    
    results.sort((a, b) => (b.width * b.height) - (a.width * a.height));
    return results.slice(0, 50);
  }
  
  // Converte imagem para blob
  async function imageToBlob(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) return null;
      return await response.blob();
    } catch (e) {
      console.warn('[MangaAutoColor] Failed to fetch:', url, e.message);
      return null;
    }
  }
  
  // Message listener
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'detectImages') {
      const images = findMangaImages();
      console.log('[MangaAutoColor] Found', images.length, 'images');
      sendResponse(images);
      return true;
    }
    
    if (request.action === 'downloadImages') {
      // Baixa todas as imagens e converte para base64
      (async () => {
        console.log('[MangaAutoColor] Downloading', request.images.length, 'images...');
        const downloaded = [];
        
        for (let i = 0; i < request.images.length; i++) {
          const img = request.images[i];
          try {
            const blob = await imageToBlob(img.src);
            if (blob) {
              // Converte blob para base64
              const reader = new FileReader();
              const base64 = await new Promise((resolve) => {
                reader.onloadend = () => resolve(reader.result);
                reader.readAsDataURL(blob);
              });
              
              downloaded.push({
                name: 'page_' + String(i).padStart(3, '0') + '.jpg',
                data: base64
              });
            }
          } catch (e) {
            console.warn('[MangaAutoColor] Failed:', img.src);
          }
          
          // Reporta progresso
          chrome.runtime.sendMessage({
            action: 'downloadProgress',
            current: i + 1,
            total: request.images.length
          });
        }
        
        console.log('[MangaAutoColor] Downloaded', downloaded.length, 'images');
        sendResponse({ success: true, images: downloaded });
      })();
      
      return true; // Async response
    }
    
    // Substitui imagens na pagina pelas versoes colorizadas
    if (request.action === 'replaceImages') {
      (async () => {
        try {
          const imageMap = request.imageMap || {};  // Mapa: src_original -> src_colorida
          const mappings = request.mappings || [];
          
          console.log('[MangaAutoColor] =======================================');
          console.log('[MangaAutoColor] INICIANDO SUBSTITUICAO DE IMAGENS');
          console.log('[MangaAutoColor] Mapa recebido:', Object.keys(imageMap).length, 'entradas');
          console.log('[MangaAutoColor] Mappings:', mappings);
          
          // Lista as primeiras 3 entradas do mapa para debug
          const mapEntries = Object.entries(imageMap);
          for (let i = 0; i < Math.min(3, mapEntries.length); i++) {
            console.log(`[MangaAutoColor] Mapa[${i}]: ${mapEntries[i][0].substring(0, 60)}...`);
          }
          
          // Encontra todas as imagens na pagina que correspondem ao mapa
          const allImages = Array.from(document.querySelectorAll('img'));
          console.log('[MangaAutoColor] Total de imagens na pagina:', allImages.length);
          
          let replaced = 0;
          let checked = 0;
          
          for (const img of allImages) {
            const originalSrc = img.src || img.dataset?.src;
            if (!originalSrc) continue;
            
            checked++;
            
            // Procura no mapa (compara apenas a parte relevante da URL)
            const matchedColorizedSrc = findMatchingImage(originalSrc, imageMap);
            
            if (matchedColorizedSrc) {
              console.log('[MangaAutoColor] ✓ MATCH! Substituindo:', originalSrc.substring(0, 50) + '...');
              
              // Guarda src original para possível restauração
              if (!img.dataset.originalSrc) {
                img.dataset.originalSrc = originalSrc;
              }
              
              // Cria um efeito de fade
              img.style.transition = 'opacity 0.5s ease';
              img.style.opacity = '0.5';
              
              // Aguarda um pouco e substitui
              await new Promise(r => setTimeout(r, 300));
              
              img.src = matchedColorizedSrc;
              img.style.opacity = '1';
              
              // Marca como colorida
              img.dataset.mangaColored = 'true';
              img.style.border = '3px solid #00ff00'; // Borda verde indicando colorido
              
              replaced++;
            }
          }
          
          console.log('[MangaAutoColor] =======================================');
          console.log('[MangaAutoColor] RESULTADO:', replaced, 'de', checked, 'imagens substituidas');
          console.log('[MangaAutoColor] =======================================');
          
          sendResponse({ success: true, replaced: replaced, checked: checked });
        } catch (e) {
          console.error('[MangaAutoColor] ERRO ao substituir:', e);
          sendResponse({ success: false, error: e.message });
        }
      })();
      
      return true; // Async response
    }
    
    // Guarda o mapeamento de imagens detectadas para uso posterior
    if (request.action === 'storeImageMap') {
      window.mangaAutoColorImageMap = request.imageMap || {};
      console.log('[MangaAutoColor] Mapa de imagens armazenado:', Object.keys(window.mangaAutoColorImageMap).length);
      sendResponse({ success: true });
      return true;
    }
  });
})();
