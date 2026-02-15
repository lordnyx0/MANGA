"""
MangaAutoColor Pro - Pass 2: Geração Tile-Aware (VRAM Bound)

Processa páginas em qualquer ordem usando cache do Pass 1.
- Carrega embeddings necessários (Top-K)
- Paletas CIELAB para consistência de cores
- Differential Diffusion com Change Maps
- Multi-band Blending

Baseado na arquitetura Two-Pass descrita em ARCHITECTURE.md
"""

from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import gc

from config.settings import (
    DEVICE, DTYPE, TILE_SIZE, TILE_OVERLAP,
    V3_STEPS, V3_GUIDANCE_SCALE, V3_CONTROL_SCALE,
    IP_ADAPTER_END_STEP, VERBOSE
)
from core.database.chapter_db import ChapterDatabase
from core.generation.engines.sd15_lineart_engine import SD15LineartEngine
from core.generation.interfaces import ColorizationEngine
from core.identity.palette_manager import CharacterPalette
from core.generation.scene_palette_service import ScenePaletteService
from core.logging import GenerationLogger
from core.utils.image_ops import create_gaussian_mask, create_blend_mask
from core.constants import SceneType, DetectionClass
from core.generation.tiling import TilingManager
from core.generation.prompt_builder import MangaPromptBuilder
from core.generation.text_compositor import TextCompositor
from core.logging.setup import get_logger

logger = get_logger("Pass2Generator")


class Pass2Generator:
    """
    Gerador de capítulo - Passo 2 do Two-Pass System.
    
    Este é o passo VRAM-bound que gera as páginas colorizadas
    usando o cache imutável criado no Pass 1.
    """
    
    def __init__(
        self,
        chapter_id: str,
        device: str = DEVICE,
        dtype=torch.float16,
        output_dir: Optional[str] = None,
        enable_logging: bool = True,
        engine: Optional["ColorizationEngine"] = None
    ):
        self.chapter_id = chapter_id
        self.device = device
        self.dtype = dtype
        self.enable_logging = enable_logging
        
        # Carrega database do Pass 1
        self.db = ChapterDatabase(chapter_id)
        self.db.load_all()
        
        # Inicializa logger
        if enable_logging and output_dir:
            self.logger = GenerationLogger(chapter_id, output_dir)
            # Registra configurações globais
            from config import settings
            self.logger.set_global_config({
                "device": device,
                "dtype": str(dtype),
                "tile_size": settings.TILE_SIZE,
                "tile_overlap": settings.TILE_OVERLAP,
                "v3_steps": settings.V3_STEPS,
                "v3_guidance_scale": settings.V3_GUIDANCE_SCALE,
                "v3_control_scale": settings.V3_CONTROL_SCALE,
                "ip_adapter_end_step": settings.IP_ADAPTER_END_STEP
            })
        else:
            from core.logging import NullLogger
            self.logger = NullLogger()
        
        # Inicializa serviços auxiliares
        self.scene_palette_service = ScenePaletteService(self.db, output_dir=output_dir)
        
        # Inicializa gerador base (Engine V3)
        self.prompt_builder = MangaPromptBuilder(self.logger)
        self.tiling_manager = TilingManager(TILE_SIZE, TILE_OVERLAP)
        self.text_compositor = TextCompositor()

        # ENGINE V3 (Dependency Injection)
        if engine:
            self.engine = engine
        else:
            self.engine = SD15LineartEngine(device=device, dtype=dtype)

        
        logger.info(f"Inicializado para capítulo: {chapter_id}")
    
    def generate_page(
        self,
        page_num: int,
        output_path: Optional[str] = None,
        options: Optional[Dict] = None
    ) -> Image.Image:
        """
        Gera uma página colorizada.
        
        Args:
            page_num: Número da página
            output_path: Path opcional para salvar resultado
            options: Opções de geração (guidance_scale, steps, etc.)
            
        Returns:
            Imagem colorizada PIL
        """
        options = options or {}
        
        # Inicia logging da página
        self.logger.start_step(f"generate_page_{page_num}", {
            "page_number": page_num,
            "output_path": output_path,
            "options_keys": list(options.keys())
        })
        
        # Carrega análise da página
        page_analysis = self.db.get_page_analysis(page_num)
        if page_analysis is None:
            error_msg = f"Página {page_num} não encontrada no cache. Execute Pass 1 primeiro."
            self.logger.log_error(error_msg)
            self.logger.finish_step()
            raise ValueError(error_msg)
        
        # INJECT SCENE TYPE INTO OPTIONS
        # This ensures TileAwareGenerator receives the detected scene
        if 'scene_type' not in options:
            options['scene_type'] = getattr(page_analysis, 'scene_type', SceneType.PRESENT.value)
            logger.info(f"Scene Context: {options['scene_type']}")

        # 1. DETERMINISTIC GENERATOR (Phase 3 Fix)
        # Garante que todos os tiles usem a mesma semente base de forma controlada
        base_seed = options.get("seed", 42)
        # No V3, passamos o gerador para garantir consistência entre tiles (se houver tiling)
        options['generator'] = torch.Generator(device=self.device).manual_seed(base_seed)

        # Carrega imagem original
        image = Image.open(page_analysis.image_path).convert('RGB')
        img_w, img_h = image.size
        
        # Carrega TileJobs
        tile_jobs = self.db.get_tile_jobs(page_num)
        
        if not tile_jobs:
            error_msg = f"Nenhum TileJob encontrado para página {page_num}. Pass 1 incompleto ou falhou."
            self.logger.log_error(error_msg)
            raise ValueError(error_msg)
        
        # Decide estratégia
        # V3 Deep Fix: Desativa Mosaic Tiling (512px) que gera "patches" coloridos.
        # Usa Single Pass Global (Engine v3 lida com downscale interno e ControlNet)
        logger.info("Modo: Global Single Pass (Phase 2 Fix - consistency focus)")
        
        # Passa as deteções explicitamente para o engine fazer o "Pre-Cleaning"
        options['detections'] = self._get_detections_for_page(page_num)
        
        # Pega o primeiro tile job apenas para metadados de personagens se necessário,
        # mas gera a página inteira.
        result = self._generate_single_tile_page(page_num, tile_jobs[0], image, options)

        # Nota: A lógica legada de tile único foi removida na v3
        
        # Salva se path fornecido
        if output_path:
            result.save(output_path, quality=95)
            logger.info(f"Página salva: {output_path}")
        
        # Finaliza logging da página
        self.logger.finish_step({
            "output_size": result.size,
            "output_path": output_path,
            "tile_count": len(tile_jobs),
            "mode": "single" if len(tile_jobs) == 1 else "multi"
        })
        
        return result
    
    def generate_chapter(
        self,
        output_dir: str,
        page_numbers: Optional[List[int]] = None,
        options: Optional[Dict] = None,
        progress_callback=None
    ) -> List[str]:
        """
        Gera todas as páginas do capítulo.
        
        Args:
            output_dir: Diretório para salvar imagens
            page_numbers: Lista opcional de páginas específicas (None = todas)
            options: Opções de geração (text_compositing, guidance_scale, etc.)
            progress_callback: Callback de progresso
            
        Returns:
            Lista de paths das imagens geradas
        """
        options = options or {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicia logging do capítulo
        self.logger.start_step("generate_chapter", {
            "output_dir": str(output_dir),
            "page_numbers": page_numbers,
            "options": {k: str(v)[:100] for k, v in options.items()}  # Limita tamanho
        })
        
        # Determina quais páginas gerar
        if page_numbers is None:
            # Pega todas as páginas do database
            summary = self.db.get_summary()
            page_numbers = list(range(summary['total_pages']))
        
        logger.info(f"Gerando {len(page_numbers)} páginas...")
        logger.debug(f"Opções: {options}")
        
        # Log das opções principais
        self.logger.log_config({
            "total_pages": len(page_numbers),
            "page_numbers": page_numbers,
            "text_compositing": options.get('text_compositing', False),
            "max_quality": options.get('max_quality', True),
            "style_preset": options.get('style_preset', 'default'),
            "guidance_scale": options.get('guidance_scale', None),
            "steps": options.get('steps', None)
        })
        
        generated_paths = []
        
        for i, page_num in enumerate(page_numbers):
            try:
                output_path = output_dir / f"page_{page_num:03d}_colored.png"
                
                self.generate_page(
                    page_num=page_num,
                    output_path=str(output_path),
                    options=options
                )
                
                generated_paths.append(str(output_path))
                
                # Callback de progresso
                if progress_callback:
                    progress_callback(i + 1, len(page_numbers))
                gc.collect()
                
            except Exception as e:
                error_msg = f"ERRO na página {page_num}: {e}"
                logger.error(f"{error_msg}")
                self.logger.log_error(error_msg)
                import traceback
                traceback_str = traceback.format_exc()
                self.logger.log_error(traceback_str)
                traceback.print_exc()
        
        # Finaliza logging do capítulo
        self.logger.finish_step({
            "total_pages": len(page_numbers),
            "generated_pages": len(generated_paths),
            "failed_pages": len(page_numbers) - len(generated_paths)
        })
        
        # Finaliza sessão de logging
        self.logger.finalize({
            "total_pages": len(page_numbers),
            "generated_pages": len(generated_paths),
            "failed_pages": len(page_numbers) - len(generated_paths),
            "output_dir": str(output_dir)
        })
        
        logger.info(f"Capítulo gerado: {len(generated_paths)} páginas")
        logger.info(f"Logs salvos em: {Path(output_dir) / 'logs'}")
        return generated_paths
    
    
    def _generate_single_tile_page(
        self,
        page_num: int,
        tile_job,
        original_image: Image.Image,
        options: Dict
    ) -> Image.Image:
        """Gera página com tile único (Engine v3)"""
        logger.debug(f"TileJob: page={page_num}, active_char_ids={tile_job.active_char_ids}")
        
        # 1. Determina referência visual
        reference_image = None
        # V3 simplificado: pega o primeiro personagem ativo como referência principal
        # Futuro: Regional IP para múltiplos personagens
        if tile_job.active_char_ids:
            char_id = tile_job.active_char_ids[0]
            reference_image = self._get_reference_image(char_id)
            if reference_image:
                logger.info(f"Usando referência visual para: {char_id}")
            else:
                logger.info(f"Sem referência visual encontrada para: {char_id}")
        
        # 2. Configura opções
        engine_options = options.copy()
        engine_options['reference_image'] = reference_image
        
        # 3. Gera Color Layer (SD 1.5 + ControlNet Lineart)
        # O engine já lida com o prompt e condicionamento
        color_layer = self.engine.generate_page(original_image, engine_options)
        
        # 4. Composição Multiply (Crucial para preservar traço original)
        result = self.engine.compose_final(original_image, color_layer)
        
        return result
    
    def _get_reference_image(self, char_id: str) -> Optional[Image.Image]:
        """
        Recupera imagem de referência visual para o personagem.
        Procura em: Cache de referência global (reference_gallery)
        """
        if not char_id:
            return None
            
        # Tenta carregar da galeria de referências
        from config.settings import REFERENCE_GALLERY_DIR
        ref_path = REFERENCE_GALLERY_DIR / f"{char_id}.png"
        
        if ref_path.exists():
            try:
                return Image.open(ref_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Falha ao carregar referência para {char_id}: {e}")
                return None
        
        return None

    def _load_character_palettes(self, char_ids: List[str]) -> Dict[str, CharacterPalette]:
        """
        Carrega paletas de cores dos personagens.
        
        Inclui paletas de referência colorida se disponíveis.
        
        Args:
            char_ids: Lista de IDs de personagens
            
        Returns:
            Dict mapeando char_id -> CharacterPalette
        """
        palettes = {}
        
        # Carrega paletas dos personagens detectados
        for char_id in char_ids:
            palette = self.db.load_character_palette(char_id)
            if palette:
                palettes[char_id] = palette
        
        # Carrega paletas de referência colorida (se houver)
        ref_palettes = self.db.load_reference_palettes()
        if ref_palettes:
            logger.info(f"Carregadas {len(ref_palettes)} paletas de referência colorida: {list(ref_palettes.keys())}")
            # Adiciona as referências ao dict de paletas
            palettes.update(ref_palettes)
        else:
            # DEBUG: Verifica se há paletas no banco
            all_palettes = self.db.load_all_palettes()
            logger.debug(f"Total de paletas no banco: {len(all_palettes)}")
            for pid, p in all_palettes.items():
                is_ref = getattr(p, 'is_color_reference', False)
                src_page = getattr(p, 'source_page', 0)
                logger.debug(f"{pid}: is_color_reference={is_ref}, source_page={src_page}")
        
        return palettes
    
    def _prepare_regional_ip_adapter(
        self,
        chars_in_tile: List[Dict],
        tile_bbox: Tuple[int, int, int, int]
    ) -> Tuple[List[Image.Image], List[Image.Image], List[str]]:
        """
        Prepara dados para Regional IP-Adapter.
        
        Returns:
            Tupla de (ref_images, ip_masks, zero_shot_prompts)
        """
        ref_images = []
        ip_masks = []
        zero_shot_prompts = []
        tx1, ty1, tx2, ty2 = tile_bbox
        tile_w, tile_h = tx2 - tx1, ty2 - ty1
        
        for char_det in chars_in_tile:
            char_id = char_det['char_id']
            ref_img = self._get_reference_image(char_id)
            
            if ref_img:
                ref_images.append(ref_img)
                char_mask = self._create_character_mask(
                    char_det['bbox'], tile_bbox, (tile_w, tile_h)
                )
                if char_mask:
                    ip_masks.append(char_mask)
            else:
                # ZERO-SHOT: usa ScenePalette no prompt
                profile = self.scene_palette_service.get_profile(char_id)
                prompt = self.prompt_builder.build_prompt_for_character(
                    character_desc=f"{profile.archetype} character",
                    color_profile=profile,
                    scene_palette=self.scene_palette_service.scene_palette
                )
                zero_shot_prompts.append(prompt)
        
        return ref_images, ip_masks, zero_shot_prompts
    
    def _create_character_mask(
        self,
        char_bbox: Tuple[int, int, int, int],
        tile_bbox: Tuple[int, int, int, int],
        tile_size: Tuple[int, int]
    ) -> Optional[Image.Image]:
        """Cria máscara de personagem para IP-Adapter regional."""
        from PIL import ImageDraw
        
        tx1, ty1, _, _ = tile_bbox
        tile_w, tile_h = tile_size
        dx1, dy1, dx2, dy2 = char_bbox
        
        # Coordenadas locais ao tile
        ix1 = max(0, dx1 - tx1)
        iy1 = max(0, dy1 - ty1)
        ix2 = min(tile_w, dx2 - tx1)
        iy2 = min(tile_h, dy2 - ty1)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        
        char_mask = Image.new("L", (tile_w, tile_h), 0)
        draw = ImageDraw.Draw(char_mask)
        draw.rectangle((ix1, iy1, ix2, iy2), fill=255)
        
        return char_mask
    
    def _build_tile_options(
        self,
        base_options: Dict,
        ref_images: List[Image.Image],
        ip_masks: List[Image.Image],
        zero_shot_prompts: List[str]
    ) -> Dict:
        """Constrói opções de geração para um tile."""
        from config import settings
        
        tile_options = base_options.copy()
        
        # Adiciona prompts de zero-shot
        if zero_shot_prompts:
            existing = tile_options.get('prompt', '')
            unique_prompts = list(set(zero_shot_prompts))
            tile_options['prompt'] = existing + ", " + ", ".join(unique_prompts)
        
        # Configura IP-Adapter
        tile_options['ip_adapter_scale'] = settings.V3_IP_SCALE
        tile_options['ip_adapter_end_step'] = settings.IP_ADAPTER_END_STEP
        tile_options['reference_image'] = ref_images if ref_images else None
        tile_options['ip_adapter_masks'] = ip_masks if ip_masks else None
        
        return tile_options
    
    def _generate_native_tiled(
        self,
        full_image: Image.Image,
        page_num: int,
        options: Dict
    ) -> Image.Image:
        """
        Implementa estratégia V3 Real: Tiling 512px com overlap.
        Substitui downscale/upscale por geração nativa mosaica.
        """
        w, h = full_image.size
        TILE_SIZE_NATIVE = 512
        OVERLAP_NATIVE = 128
        
        tiles = self.tiling_manager.slice(full_image, TILE_SIZE_NATIVE, OVERLAP_NATIVE)
        detections = self._get_detections_for_page(page_num)
        results = []
        
        logger.info(f"Gerando {len(tiles)} tiles nativos (512px) para página {page_num} ({w}x{h})")
        
        for tile_img, bbox in tqdm(tiles, desc="Native Tiles"):
            # Identifica personagens no tile
            chars_in_tile = self.tiling_manager.get_characters_in_tile(
                bbox, detections, min_overlap=0.1
            )
            chars_in_tile.sort(
                key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]),
                reverse=True
            )
            
            # Prepara IP-Adapter
            ref_images, ip_masks, zero_shot_prompts = self._prepare_regional_ip_adapter(
                chars_in_tile, bbox
            )
            
            # Gera o tile
            tile_options = self._build_tile_options(options, ref_images, ip_masks, zero_shot_prompts)
            colored_tile = self.engine.generate_page(tile_img, tile_options)
            results.append((colored_tile, bbox))
        
        # Blending e composição final
        color_layer = self.tiling_manager.blend(results, (w, h), method="linear")
        return self.engine.compose_final(full_image, color_layer)

    def _generate_multi_tile_page(
        self,
        page_num: int,
        tile_jobs: List,
        original_image: Image.Image,
        options: Dict
    ) -> Image.Image:
        """Gera página com múltiplos tiles (Engine v3)"""
        img_w, img_h = original_image.size
        
        # Acumuladores para blending
        accumulator = np.zeros((img_h, img_w, 3), dtype=np.float32)
        weight_map = np.zeros((img_h, img_w), dtype=np.float32)
        
        for tile_job in tqdm(tile_jobs, desc=f"Tiles página {page_num}"):
            # 1. Determina referência visual para este tile
            reference_image = None
            if tile_job.active_char_ids:
                # Pega o primeiro (ou mais importante)
                char_id = tile_job.active_char_ids[0]
                reference_image = self._get_reference_image(char_id)
            
            # 2. Extrai imagem do tile
            tx1, ty1, tx2, ty2 = tile_job.tile_bbox
            tile_image = original_image.crop((tx1, ty1, tx2, ty2))
            
            # 3. Configura opções
            tile_options = options.copy()
            tile_options['reference_image'] = reference_image
            
            # 4. Gera Color Layer do Tile
            # Nota: Usamos generate_page (que chama generate_region) para o tile
            tile_color_layer = self.engine.generate_page(tile_image, tile_options)
            
            # 5. Redimensiona se necessário (segurança)
            if tile_color_layer.size != tile_image.size:
                tile_color_layer = tile_color_layer.resize(tile_image.size, Image.LANCZOS)
            
            # 6. Accumulate (Blending)
            tile_np = np.array(tile_color_layer).astype(np.float32)
            th, tw = tile_np.shape[:2]
            blend_mask = create_blend_mask(
                (th, tw), TILE_OVERLAP, image_size=(img_w, img_h), tile_bbox=tile_job.tile_bbox
            )
            
            accumulator[ty1:ty2, tx1:tx2] += tile_np * blend_mask[:, :, np.newaxis]
            weight_map[ty1:ty2, tx1:tx2] += blend_mask
            
            # Limpeza
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # 7. Normaliza Blending (Full Color Layer)
        weight_map = np.maximum(weight_map, 1e-8)
        full_color_layer_np = (accumulator / weight_map[:, :, np.newaxis]).astype(np.uint8)
        full_color_layer = Image.fromarray(full_color_layer_np)
        
        # 8. Composição Multiply Final
        result = self.engine.compose_final(original_image, full_color_layer)
        
        return result
    

    
    def _get_detections_for_page(self, page_num: int) -> List[Dict]:
        """Recupera detecções da página do database"""
        page_analysis = self.db.get_page_analysis(page_num)
        if page_analysis is None:
            return []
        return page_analysis.detections
    
    def _generate_z_ordered_masks(
        self,
        tile_bbox: Tuple[int, int, int, int],
        detections: List[Dict],
        active_char_ids: List[str],
        page_num: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Gera máscaras com Z-Ordering para resolver sobreposições.
        
        ADR 004 §3.3: Fórmula de Isolamento completa:
        M_i_final = M_i_SAM ∩ (¬⋃ Front(i)) ∩ (¬⋃ ε(SamePlane(i)))
        
        Pipeline:
        1. Carrega máscaras SAM do database (RLE decode)
        2. Aplica resolução de oclusões via MaskProcessor (Z-Ordering + erosão)
        3. Recorta para o tile atual
        4. Suaviza bordas (Gaussian blur sigma=0.5)
        """
        x1, y1, x2, y2 = tile_bbox
        w, h = x2 - x1, y2 - y1
        
        # Obtém depth_order do database (ADR 004)
        depth_order = []
        if page_num is not None:
            depth_order = self.db.get_depth_order(page_num)
        
        # Fallback: ordena por área se não houver depth_order
        if not depth_order:
            areas = {}
            for det in detections:
                if det.get('char_id') in active_char_ids:
                    dx1, dy1, dx2, dy2 = det['bbox']
                    areas[det['char_id']] = (dx2 - dx1) * (dy2 - dy1)
            depth_order = sorted(areas.keys(), key=lambda k: areas[k])
        
        # Filtra apenas char_ids ativos neste tile
        depth_order = [cid for cid in depth_order if cid in active_char_ids]
        
        # Carrega máscaras SAM do database
        sam_masks = {}
        for char_id in depth_order:
            mask = self.db.get_character_mask(char_id)
            if mask is not None:
                sam_masks[char_id] = mask
        
        # GAP #5: Fallback BBox com validação de limites
        for det in detections:
            char_id = det.get('char_id')
            if char_id in active_char_ids and char_id not in sam_masks:
                dx1, dy1, dx2, dy2 = det['bbox']
                shape = det.get('mask_shape', (y2, x2))
                try:
                    img_h, img_w = int(shape[0]), int(shape[1])
                except (TypeError, IndexError, ValueError):
                    img_h, img_w = y2, x2
                
                # Valida coordenadas dentro dos limites
                dx1 = max(0, min(int(dx1), img_w))
                dy1 = max(0, min(int(dy1), img_h))
                dx2 = max(dx1, min(int(dx2), img_w))
                dy2 = max(dy1, min(int(dy2), img_h))
                
                if dx2 > dx1 and dy2 > dy1:
                    mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    mask[dy1:dy2, dx1:dx2] = 255
                    sam_masks[char_id] = mask
        
        if not sam_masks:
            return {}
        
        # ADR 004 §3.3: Instancia MaskProcessor com erosão de 2px
        # para resolver contatos diretos (SamePlane erosion)
        from core.analysis.mask_processor import MaskProcessor
        from core.analysis.segmentation import SegmentationResult
        
        processor = MaskProcessor(
            close_kernel_size=3,
            erosion_pixels=2,       # ADR 004: ε = 2px em contatos diretos
            blur_sigma=0.5,         # Anti-hard-edge
            overlap_dilation=1      # Garantir overlap mínimo em fundo
        )
        
        # Converte para SegmentationResult via from_mask (lida com RLE internamente)
        seg_results = {}
        for char_id, mask in sam_masks.items():
            # Encontra bbox correspondente
            bbox = (0, 0, mask.shape[1], mask.shape[0])  # fallback full-image
            for det in detections:
                if det.get('char_id') == char_id and det.get('bbox'):
                    bbox = tuple(int(c) for c in det['bbox'])
                    break
            seg_results[char_id] = SegmentationResult.from_mask(
                char_id=char_id,
                mask=mask,
                bbox=bbox,
                confidence=1.0
            )
        
        # Processa máscaras com pipeline completo:
        # morphological close → occlusion resolution → SamePlane erosion → overlap dilation → blur
        # IMPORTANT: depth_order pode conter char_ids sem máscara (sem SAM nem BBox).
        # Filtra para evitar KeyError em process_masks.
        effective_depth_order = [cid for cid in depth_order if cid in seg_results]
        processed = processor.process_masks(seg_results, effective_depth_order)
        
        # Recorta máscaras processadas para este tile
        masks = {}
        for char_id in depth_order:
            if char_id not in processed:
                continue
            
            pm = processed[char_id]
            # mask_float já é float32 [0.0-1.0] com blur aplicado pelo MaskProcessor
            full_mask = pm.mask_float
            mh, mw = full_mask.shape[:2]
            
            # Recorta região do tile
            tx1 = max(0, x1)
            ty1 = max(0, y1)
            tx2 = min(mw, x2)
            ty2 = min(mh, y2)
            
            if tx2 <= tx1 or ty2 <= ty1:
                continue
            
            mask_region = full_mask[ty1:ty2, tx1:tx2]
            
            # Posiciona no tile (float32 — mask_float já está em [0.0-1.0])
            tile_mask = np.zeros((h, w), dtype=np.float32)
            off_x = max(0, x1 - tx1) if x1 < 0 else 0
            off_y = max(0, y1 - ty1) if y1 < 0 else 0
            
            rh, rw = mask_region.shape[:2]
            try:
                tile_mask[off_y:off_y+rh, off_x:off_x+rw] = mask_region
            except ValueError as e:
                logger.debug(f"Erro ao posicionar máscara {char_id}: {e}")
                continue
            
            masks[char_id] = tile_mask
        
        return masks

    def unload(self):
        """Descarrega modelos da VRAM (Phase 3 Fix)"""
        if hasattr(self, "engine") and self.engine is not None:
            if hasattr(self.engine, "offload_models"):
                try:
                    self.engine.offload_models()
                except Exception as e:
                    logger.warning(f"Falha ao unload engine: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Modelos descarregados")
