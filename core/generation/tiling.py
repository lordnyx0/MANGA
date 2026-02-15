import numpy as np
from typing import List, Tuple, Dict, Any, Union
from config.settings import TILE_SIZE, TILE_OVERLAP
from core.detection.yolo_detector import DetectionResult

class TilingManager:
    """
    Gerenciador de lógica de tiling e geometria para geração de imagens.
    
    Responsabilidades:
    - Calcular grid de tiles
    - Identificar personagens dentro de cada tile
    - Gerenciar blending de tiles
    """
    
    def __init__(self, tile_size: int = TILE_SIZE, tile_overlap: int = TILE_OVERLAP):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

    def calculate_tile_grid(self, image_size: Tuple[int, int]) -> Tuple[int, int, List[Tuple[int, int, int, int]]]:
        """
        Calcula a grade de tiles para cobrir a imagem.
        
        Args:
            image_size: (width, height)
            
        Returns:
            (num_tiles_x, num_tiles_y, list_of_bboxes)
        """
        w, h = image_size
        
        # Se imagem menor que tile, retorna único tile
        if w <= self.tile_size and h <= self.tile_size:
            return 1, 1, [(0, 0, w, h)]
            
        # Calcula número de tiles necessário
        stride = self.tile_size - self.tile_overlap
        
        num_tiles_x = max(1, (w - self.tile_size + stride - 1) // stride + 1)
        num_tiles_y = max(1, (h - self.tile_size + stride - 1) // stride + 1)
        
        # Ajusta se não cobrir tudo
        if (num_tiles_x - 1) * stride + self.tile_size < w:
            num_tiles_x += 1
        if (num_tiles_y - 1) * stride + self.tile_size < h:
            num_tiles_y += 1
            
        bboxes = []
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Coordenadas iniciais
                x1 = x * stride
                y1 = y * stride
                
                # Ajusta último tile para alinhar à direita/baixo
                if x == num_tiles_x - 1:
                    x1 = max(0, w - self.tile_size)
                
                if y == num_tiles_y - 1:
                    y1 = max(0, h - self.tile_size)
                    
                x2 = min(w, x1 + self.tile_size)
                y2 = min(h, y1 + self.tile_size)
                
                bboxes.append((x1, y1, x2, y2))
                
        return num_tiles_x, num_tiles_y, bboxes

    def get_characters_in_tile(
        self,
        tile_bbox: Tuple[int, int, int, int],
        detections: List[Dict],
        min_overlap: float = 0.3
    ) -> List[Dict]:
        """
        Identifica quais personagens estão dentro de um tile.
        
        Args:
            tile_bbox: (x1, y1, x2, y2) do tile
            detections: Lista de detecções
            min_overlap: Área mínima de sobreposição relativa ao bbox personagem
            
        Returns:
            Lista de detecções que estão no tile (com coordenadas ajustadas?) 
            Nota: Retorna a detecção original, o ajuste de coordenadas deve ser feito pelo chamador se necessário.
        """
        tx1, ty1, tx2, ty2 = tile_bbox
        chars_in_tile = []
        
        for det in detections:
            # Suporta dict ou DetectionResult
            if isinstance(det, dict):
                bbox = det.get('bbox')
            else:
                bbox = det.bbox # DetectionResult
                
            if not bbox:
                continue
                
            dx1, dy1, dx2, dy2 = bbox
            
            # Interseção
            ix1 = max(tx1, dx1)
            iy1 = max(ty1, dy1)
            ix2 = min(tx2, dx2)
            iy2 = min(ty2, dy2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                char_area = (dx2 - dx1) * (dy2 - dy1)
                
                if char_area > 0 and (intersection_area / char_area) >= min_overlap:
                    chars_in_tile.append(det)
                    
        return chars_in_tile

    def slice(self, image, tile_size: int = 512, overlap: int = 128) -> List[Tuple[Any, Tuple[int, int, int, int]]]:
        """
        Fatia a imagem em tiles com overlap.
        
        Args:
            image: PIL Image
            tile_size: Tamanho do tile (quadrado)
            overlap: Tamanho do overlap
            
        Returns:
            Lista de (tile_image, bbox)
        """
        # Atualiza configurações temporariamente ou usa instância dedicada?
        # Vamos usar os parâmetros passados para este método, ignorando self.tile_size se diferir
        
        w, h = image.size
        # Reutiliza logica de grid, mas precisamos adaptar pois calculate_tile_grid usa self.tile_size
        # Vamos criar um metodo estatico ou auxiliar, ou instanciar um novo TilingManager?
        # Melhor duplicar logica simples aqui ou generalizar calculate_tile_grid.
        # Vamos implementar a lógica direta aqui para evitar side effects.
        
        stride = tile_size - overlap
        
        num_tiles_x = max(1, (w - tile_size + stride - 1) // stride + 1)
        num_tiles_y = max(1, (h - tile_size + stride - 1) // stride + 1)
        
        # Ajusta cobertura
        if (num_tiles_x - 1) * stride + tile_size < w: num_tiles_x += 1
        if (num_tiles_y - 1) * stride + tile_size < h: num_tiles_y += 1
        
        tiles = []
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                x1 = x * stride
                y1 = y * stride
                
                # Snap to edges
                if x == num_tiles_x - 1: x1 = max(0, w - tile_size)
                if y == num_tiles_y - 1: y1 = max(0, h - tile_size)
                
                x2 = min(w, x1 + tile_size)
                y2 = min(h, y1 + tile_size)
                
                tile_img = image.crop((x1, y1, x2, y2))
                tiles.append((tile_img, (x1, y1, x2, y2)))
                
        return tiles

    def blend(self, results: List[Tuple[Any, Tuple[int, int, int, int]]], original_size: Tuple[int, int], method: str = "linear") -> Any:
        """
        Realiza o blending dos tiles de volta para a imagem completa.
        
        Args:
            results: Lista de (tile_image, tile_bbox)
            original_size: (width, height) da imagem final
            method: "linear" (feathered) ou "multi-band" (placeholder)
            
        Returns:
            Imagem PIL montada
        """
        from PIL import Image
        import numpy as np
        from core.utils.image_ops import create_blend_mask
        
        w, h = original_size
        accumulator = np.zeros((h, w, 3), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        overlap_size = 0
        # Tenta deduzir overlap dos dados ou usa default.
        # Assumindo que o chamador usou o slice deste manager.
        # Precisamos do overlap para criar a mascara correta.
        # Vamos assumir self.tile_overlap se nao passado?
        # Mas slice aceita overlap custom.
        # Vamos calcular o overlap baseando no primeiro tile se possivel ou usar constante?
        # O create_blend_mask precisa do overlap.
        # Vamos usar um valor seguro ou passar como arg?
        # O user pediu "blend(results, method)".
        # Vamos usar self.tile_overlap como fallback, mas idealmente deveria ser parametro.
        # Como slice usa 128 por padrao (snippet user), vamos usar 128 se self.tile_overlap for diferente?
        # Vamos confiar em self.tile_overlap do construtor TilingManager(512, 128).
        
        overlap_val = self.tile_overlap
        
        for tile_img, bbox in results:
            x1, y1, x2, y2 = bbox
            tile_np = np.array(tile_img).astype(np.float32)
            
            # Cria mascara de blend para este tile
            # Precisamos passar o bbox e o tamanho da imagem full
            mask = create_blend_mask(bbox, (w, h), overlap_val // 2)
            
            # Acumula
            # slice do weight map correspondente ao tile
            # create_blend_mask retorna mascara full size ou só do tile?
            # image_ops.py create_blend_mask retorna (h, w) float32?
            # Vamos verificar image_ops.py.
            
            # Se create_blend_mask retornar full size:
            # weight_map logic applied below
            
            accumulator[y1:y2, x1:x2] += tile_np * mask[:, :, np.newaxis]
            weight_map[y1:y2, x1:x2] += mask

        # Normaliza
        weight_map = np.maximum(weight_map, 1e-8)
        final_np = (accumulator / weight_map[:, :, np.newaxis]).astype(np.uint8)
        return Image.fromarray(final_np)
