from typing import List, Dict, Tuple
from PIL import Image

class TextCompositor:
    """
    Responsável por compor texto original sobre a imagem gerada.
    Extraído de TileAwareGenerator para cumprir SRP.
    """
    
    def apply_compositing(
        self,
        generated_image: Image.Image,
        original_image: Image.Image,
        detections: List[Dict]
    ) -> Image.Image:
        """
        Aplica text compositing - recorta texto da original e cola na gerada.
        
        Args:
            generated_image: Imagem colorizada pela IA
            original_image: Imagem original em P&B
            detections: Lista de detecções (com text bboxes)
            
        Returns:
            Imagem com texto restaurado
        """
        result = generated_image.copy()
        text_count = 0
        
        for det in detections:
            # Verifica se é detecção de texto (class_id=3 no Manga109)
            class_id = det.get('class_id', -1)
            class_name = det.get('class_name', '')
            
            if class_id == 3 or class_name == 'text':
                bbox = det.get('bbox')
                if bbox is None:
                    continue
                    
                x1, y1, x2, y2 = bbox
                
                # Adiciona padding de segurança (6px) 
                # YOLO dá caixas justas - precisamos de margem para a borda do balão
                x1 = max(0, x1 - 6)
                y1 = max(0, y1 - 6)
                x2 = min(original_image.width, x2 + 6)
                y2 = min(original_image.height, y2 + 6)
                
                # Verifica se bbox é válido
                if x2 <= x1 or y2 <= y1:
                    continue
                
                try:
                    # Recorta do ORIGINAL (P&B Nítido)
                    text_crop = original_image.crop((x1, y1, x2, y2))
                    
                    # Cola na imagem GERADA (Colorida)
                    result.paste(text_crop, (x1, y1))
                    text_count += 1
                except Exception as e:
                    print(f"[TextCompositing] Erro ao processar bbox {bbox}: {e}")
        
        if text_count > 0:
            print(f"[TextCompositing] {text_count} regiões de texto restauradas")
        
        return result
    
    def apply_compositing_with_scaling(
        self,
        generated_image: Image.Image,
        original_image: Image.Image,
        detections: List[Dict],
        target_size: Tuple[int, int],
        scale_x: float = 1.0,
        scale_y: float = 1.0
    ) -> Image.Image:
        """
        Aplica text compositing com suporte a upscale e downscale.
        
        Suporta dois modos:
        - scale_x/y < 1.0: Modo downscale (imagem gerada menor que original)
        - scale_x/y > 1.0: Modo upscale (imagem gerada maior que original - SKIP_FINAL_DOWNSCALE)
        
        Args:
            generated_image: Imagem colorizada pela IA
            original_image: Imagem original em P&B (para recortar texto) - SEM REDIMENSIONAR
            detections: Lista de detecções (com text bboxes) - coordenadas no espaço ORIGINAL
            target_size: Tamanho alvo (w, h) da imagem gerada
            scale_x: Fator de escala X (original -> gerado)
            scale_y: Fator de escala Y (original -> gerado)
            
        Returns:
            Imagem com texto restaurado
        """
        result = generated_image.copy()
        text_count = 0
        target_w, target_h = target_size
        orig_w, orig_h = original_image.size
        
        # Determina modo: upscale (>1.0) ou downscale (<1.0)
        is_upscale = scale_x > 1.0 or scale_y > 1.0
        
        for det in detections:
            # Verifica se é detecção de texto (class_id=3 no Manga109)
            class_id = det.get('class_id', -1)
            class_name = det.get('class_name', '')
            
            if class_id == 3 or class_name == 'text':
                bbox = det.get('bbox')
                if bbox is None:
                    continue
                    
                # Coordenadas da detecção estão no ESPAÇO ORIGINAL
                x1_orig, y1_orig, x2_orig, y2_orig = [int(c) for c in bbox]
                
                # Adiciona padding de 4px no espaço original
                x1_orig_pad = max(0, x1_orig - 4)
                y1_orig_pad = max(0, y1_orig - 4)
                x2_orig_pad = min(orig_w, x2_orig + 4)
                y2_orig_pad = min(orig_h, y2_orig + 4)
                
                # Calcula coordenadas de DESTINO na imagem gerada
                # Multiplica pelo fator de escala para converter original -> gerado
                x1_dest = int(x1_orig * scale_x)
                y1_dest = int(y1_orig * scale_y)
                x2_dest = int(x2_orig * scale_x)
                y2_dest = int(y2_orig * scale_y)
                
                # Garante que as coordenadas de destino estão dentro dos limites
                x1_dest = max(0, min(x1_dest, target_w))
                y1_dest = max(0, min(y1_dest, target_h))
                x2_dest = max(0, min(x2_dest, target_w))
                y2_dest = max(0, min(y2_dest, target_h))
                
                # Verifica se bbox é válido
                if x2_dest <= x1_dest or y2_dest <= y1_dest:
                    continue
                
                try:
                    # Recorta da ORIGINAL (Preto e Branco Nítido)
                    text_crop = original_image.crop((x1_orig_pad, y1_orig_pad, x2_orig_pad, y2_orig_pad))
                    
                    # Calcula tamanho do crop no espaço gerado
                    dest_w = x2_dest - x1_dest
                    dest_h = y2_dest - y1_dest
                    
                    # Redimensiona o crop para o tamanho de destino
                    # No modo upscale: faz upscale do crop original
                    # No modo downscale: faz downscale do crop original
                    if text_crop.size != (dest_w, dest_h):
                        text_crop = text_crop.resize((dest_w, dest_h), Image.LANCZOS)
                    
                    # Cola na FINAL (Colorida) nas coordenadas de destino
                    result.paste(text_crop, (x1_dest, y1_dest))
                    text_count += 1
                    
                    if text_count <= 3:  # Log apenas os primeiros para não poluir
                        mode_str = "upscale" if is_upscale else "downscale"
                        print(f"[TextCompositing][{mode_str}] Restaurado em ({x1_dest},{y1_dest},{x2_dest},{y2_dest}) <- crop from original ({x1_orig},{y1_orig},{x2_orig},{y2_orig})")
                except Exception as e:
                    print(f"[TextCompositing] Erro ao processar bbox {bbox}: {e}")
        
        if text_count > 0:
            print(f"[TextCompositing] {text_count} regiões de texto restauradas (scale={scale_x:.3f},{scale_y:.3f})")
        
        return result
