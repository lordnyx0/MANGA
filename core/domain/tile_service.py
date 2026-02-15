from typing import List, Dict, TYPE_CHECKING
import json
import numpy as np
from PIL import Image
from core.database.chapter_db import TileJob
from core.generation.tiling import TilingManager
from core.utils.image_ops import extract_canny_edges
from config.settings import TILE_SIZE, TILE_OVERLAP

if TYPE_CHECKING:
    from core.database.chapter_db import ChapterDatabase

class TileService:
    """
    Serviço de domínio para gerenciamento de tiles.
    Centraliza a criação de jobs de tile e pré-processamento de geometria.
    """
    
    def __init__(self, db: "ChapterDatabase"):
        self.db = db
        self.tiling_manager = TilingManager(TILE_SIZE, TILE_OVERLAP)

    def generate_jobs_for_chapter(self):
        """
        Gera TileJobs para todas as páginas do capítulo.
        Deve ser chamado após a análise e consolidação (Passo 1).
        """
        print(f"[TileService] Gerando TileJobs para capítulo {self.db.chapter_id}...")
        
        # Garante que temos as páginas
        if self.db._pages_df is None:
             self.db._load_parquets()
        
        if self.db._pages_df is None or len(self.db._pages_df) == 0:
            print("[TileService] Nenhuma página encontrada para processar.")
            return

        pages = self.db._pages_df.to_dict('records')
        
        count = 0
        for page in pages:
            self._process_page(page)
            count += 1
            
        # Salva jobs no banco
        self.db.save_all()
        print(f"[TileService] Jobs gerados para {count} páginas.")

    def _process_page(self, page_data: Dict):
        page_num = page_data['page_num']
        image_path = page_data['image_path']
        # Detections pode vir como string JSON se carregado do DF
        detections = page_data.get('detections', [])
        
        if isinstance(detections, str):
            try:
                detections = json.loads(detections)
            except Exception as e:
                print(f"[TileService] Erro ao decodificar detecções: {e}")
                detections = []
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"[TileService] Erro ao abrir imagem {image_path}: {e}")
            return
            
        w, h = image.size
        
        # Calculate grid
        _, _, bboxes = self.tiling_manager.calculate_tile_grid((w, h))
        
        # Create jobs
        for i, bbox in enumerate(bboxes):
            # Identify characters
            chars_in_tile = self.tiling_manager.get_characters_in_tile(bbox, detections)
            
            # Extract IDs - detections use 'char_id' key
            active_ids = []
            for c in chars_in_tile:
                if isinstance(c, dict) and 'char_id' in c:
                     active_ids.append(c['char_id'])
            
            # Generate Canny
            tile_img = image.crop(bbox)
            tile_np = np.array(tile_img)
            
            if tile_np.size > 0:
                canny = extract_canny_edges(tile_np)
            else:
                 canny = np.zeros((bbox[3]-bbox[1], bbox[2]-bbox[0]), dtype=np.uint8)

            # Save Canny
            canny_filename = f"canny_p{page_num}_t{i}.npy"
            canny_path = self.db.canny_dir / canny_filename
            np.save(canny_path, canny)
            
            # Create Job
            job = TileJob(
                page_num=page_num,
                tile_bbox=bbox,
                active_char_ids=active_ids,
                mask_paths={}, # Máscaras podem ser geradas aqui ou on-demand no Pass 2
                canny_path=str(canny_path)
            )
            
            self.db.save_tile_job(job)
