"""
MangaAutoColor Pro - Pipeline Principal
Orquestrador do sistema Two-Pass
"""

import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import json

from config.settings import (
    DEVICE, DTYPE, MAX_RESOLUTION,
    CONTEXT_INFLATION_FACTOR
)
from core.detection.yolo_detector import YOLODetector, DetectionResult
from core.identity.hybrid_encoder import HybridIdentityEncoder
from core.identity.palette_manager import PaletteExtractor, CharacterPalette
from core.utils.image_ops import calculate_context_bbox
from core.exceptions import AnalysisError, GenerationError, ModelLoadError, ResourceError
from core.constants import SceneType, DetectionClass
from core.domain.character_service import CharacterService
from core.domain.narrative_service import NarrativeService
from core.logging.setup import get_logger

logger = get_logger("Pipeline")


@dataclass
class ChapterAnalysis:
    """Resultado do Passo 1: Análise completa do capítulo"""
    chapter_id: str
    num_pages: int
    num_characters: int
    characters: List[Dict]
    scene_breakdown: Dict[str, List[int]]
    estimated_generation_time: float
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'num_pages': self.num_pages,
            'num_characters': self.num_characters,
            'characters': self.characters,
            'scene_breakdown': self.scene_breakdown,
            'estimated_generation_time': self.estimated_generation_time,
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }


@dataclass
class GenerationOptions:
    """Opções de geração para o Passo 2"""
    style_preset: str = "default"
    quality_mode: str = "balanced"
    preserve_original_text: bool = True
    apply_narrative_transforms: bool = True
    lighting_consistency: bool = True
    seed: Optional[int] = None
    # Parâmetros específicos
    num_inference_steps: int = 20
    guidance_scale: float = 7.5


class MangaColorizationPipeline:
    """
    Pipeline principal do MangaAutoColor Pro.
    Implementa arquitetura Two-Pass para colorização consistente.
    """
    
    def __init__(
        self,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
        cache_dir: str = "./cache",
        enable_xformers: bool = True,
        enable_cpu_offload: bool = True
    ):
        """
        Inicializa o pipeline de colorização.
        
        Args:
            device: Dispositivo para execução ("cuda" ou "cpu")
            dtype: Tipo de dados (torch.float16 ou torch.float32)
            cache_dir: Diretório para cache de modelos
            enable_xformers: Habilitar otimização xformers
            enable_cpu_offload: Habilitar offload para CPU
        """
        self.device = device
        self.dtype = dtype
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Componentes (lazy loading)
        self._pass1_analyzer = None
        self._pass2_generator = None
        self._database = None
        
        # Callbacks
        self._progress_callback: Optional[Callable] = None
        self._character_callback: Optional[Callable] = None
        
        # Estado
        self._chapter_loaded = False
        self._current_chapter_id: Optional[str] = None
        self._page_paths: List[str] = []
        
        logger.info(f"Inicializado em {device} com {dtype}")
    
    def _get_analyzer(self):
        """Lazy loading do Pass1Analyzer"""
        if self._pass1_analyzer is None:
            from .pass1_analyzer import Pass1Analyzer
            self._pass1_analyzer = Pass1Analyzer(
                device=self.device,
                dtype=self.dtype
            )
        return self._pass1_analyzer
    
    def _get_generator(self):
        """Lazy loading do Pass2Generator"""
        if self._current_chapter_id is None:
            raise RuntimeError("Chapter ID not set. Run process_chapter first.")
            
        if self._pass2_generator is None or self._pass2_generator.chapter_id != self._current_chapter_id:
            from .pass2_generator import Pass2Generator
            from pathlib import Path
            
            # Habilita logging detalhado passando output_dir
            output_dir = Path("output/debug")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self._pass2_generator = Pass2Generator(
                chapter_id=self._current_chapter_id,
                device=self.device,
                dtype=self.dtype,
                output_dir=str(output_dir),
                enable_logging=True
            )
        return self._pass2_generator
    
    def _get_database(self, chapter_id: str):
        """Lazy loading do Database"""
        if self._database is None or self._current_chapter_id != chapter_id:
            from .database.chapter_db import ChapterDatabase
            self._database = ChapterDatabase(chapter_id)
            self._current_chapter_id = chapter_id
        return self._database
    
    def set_progress_callback(self, callback: Callable[[int, str, float], None]):
        """
        Define callback de progresso.
        
        Args:
            callback: Função(page_num, stage, progress_percent)
        """
        self._progress_callback = callback
    
    def set_character_callback(self, callback: Callable[[Dict], None]):
        """
        Define callback para detecção de personagens.
        
        Args:
            callback: Função(char_info)
        """
        self._character_callback = callback
    
    def _notify_progress(self, page_num: int, stage: str, progress: float):
        """Notifica progresso se callback definido"""
        if self._progress_callback:
            self._progress_callback(page_num, stage, progress)
    
    def _notify_character(self, char_info: Dict):
        """Notifica detecção de personagem se callback definido"""
        if self._character_callback:
            self._character_callback(char_info)
    
    def process_chapter(
        self, 
        page_paths: List[str],
        color_reference_paths: Optional[List[str]] = None,
        chapter_id: Optional[str] = None
    ) -> ChapterAnalysis:
        """
        PASSO 1: Análise completa do capítulo.
        
        Processa todas as páginas para extrair:
        - Detecção de personagens
        - Identidades e embeddings
        - Paletas de cores
        - Contexto narrativo
        
        Args:
            page_paths: Lista de caminhos para as páginas do capítulo
            color_reference_paths: Lista de paths para imagens de referência coloridas (opcional)
            chapter_id: ID externo do capítulo (opcional, gera automaticamente se não fornecido)
            
        Returns:
            ChapterAnalysis com informações consolidadas
            
        Raises:
            AnalysisError: Se houver erro na análise
            ModelLoadError: Se houver erro ao carregar modelos
        """
        if not page_paths:
            raise ValueError("Lista de páginas vazia")
        
        self._page_paths = page_paths
        
        # Usa ID externo se fornecido, senão gera um
        if chapter_id is None:
            chapter_id = self._generate_chapter_id(page_paths)
        
        logger.info(f"Iniciando Passo 1: Análise de {len(page_paths)} páginas")
        logger.info(f"Chapter ID: {chapter_id}")
        
        if color_reference_paths:
            logger.info(f"Imagens de referência coloridas: {len(color_reference_paths)}")
        
        analyzer = self._get_analyzer()
        db = self._get_database(chapter_id)
        
        # Processa imagens de referência coloridas primeiro (se houver)
        if color_reference_paths:
            self._process_color_references(color_reference_paths, db)
        
        # Análise página por página
        for page_num, path in enumerate(page_paths):
            self._notify_progress(page_num, "analyzing", 
                                 (page_num / len(page_paths)) * 100)
            
            logger.info(f"Analisando página {page_num + 1}/{len(page_paths)}")
            
            try:
                page_data = analyzer.analyze_page(path, page_num)
                
                # Salva personagens no database (com embeddings body/face)
                characters = page_data.get('characters', [])
                if characters:
                    db.save_characters_from_analysis(characters, page_num)
                
                # Extrai IDs dos personagens
                character_ids = [c['character_id'] for c in characters if 'character_id' in c]
                
                # IMPORTANTE: Associa char_ids às detecções para TileService
                # Isso permite que TileService encontre quais personagens estão em cada tile
                detections = page_data.get('detections', [])
                for det in detections:
                    det_bbox = det.get('bbox')
                    if det_bbox:
                        # Encontra character com mesmo bbox
                        for char in characters:
                            if char.get('bbox') == det_bbox and 'character_id' in char:
                                det['char_id'] = char['character_id']
                                break
                
                # Salva análise da página
                db.save_page_analysis(
                    page_num=page_num,
                    image_path=path,
                    detections=detections,  # Agora com char_id linkado
                    character_ids=character_ids,
                    scene_type=page_data.get('scene_type', SceneType.PRESENT.value),
                    processed=True,
                    # ADR 005: PCTC removido em ADR 006
                )
                
                # Notifica personagens detectados
                for char in characters:
                    self._notify_character(char)
                    
            except Exception as e:
                logger.error(f"Erro na página {page_num}: {e}")
                raise AnalysisError(f"Falha na análise da página {page_num}: {e}")
        
        # Consolidação cross-page
        self._notify_progress(len(page_paths), "consolidating", 95)
        
        # Usa CharacterService para consolidar personagens
        try:
            char_service = CharacterService(db)
            char_service.consolidate_characters()
        except Exception as e:
            logger.error(f"Erro na consolidação de personagens: {e}")
            
        # Usa NarrativeService para detectar arcos
        try:
            narrative_service = NarrativeService(db)
            narrative_service.detect_narrative_arcs()
        except Exception as e:
            logger.error(f"Erro na detecção de arcos narrativos: {e}")
            
        # Gera TileJobs para o Pass 2
        try:
            from core.domain.tile_service import TileService
            tile_service = TileService(db)
            tile_service.generate_jobs_for_chapter()
        except Exception as e:
            logger.error(f"Erro na geração de TileJobs: {e}")
        
        # Gera resumo
        analysis = self._create_chapter_analysis(db, len(page_paths))
        
        # Persiste dados em disco
        db.save_all()
        
        self._chapter_loaded = True
        self._notify_progress(len(page_paths), "complete", 100)
        
        logger.info(f"Passo 1 completo: {analysis.num_characters} personagens detectados")
        
        return analysis

    def _process_color_references(self, color_ref_paths: List[str], db):
        """
        Processa imagens de referência coloridas para extrair paletas.
        """
        logger.info(f"Processando {len(color_ref_paths)} imagens de referência coloridas...")
        
        analyzer = self._get_analyzer()
        yolo_detector = analyzer.yolo_detector
        identity_encoder = analyzer.identity_encoder
        palette_extractor = analyzer.palette_extractor
        
        for i, ref_path in enumerate(color_ref_paths):
            try:
                # Carrega imagem de referência
                ref_image = Image.open(ref_path).convert('RGB')
                logger.info(f"Referência {i+1}: {ref_path} ({ref_image.size})")
                
                # Detecta personagens na imagem de referência
                ref_np = np.array(ref_image)
                detections = yolo_detector.detect(ref_np)
                char_detections = [d for d in detections if d.class_id in [0, 1]]
                
                if not char_detections:
                    logger.warning(f"AVISO: Nenhum personagem detectado na referência {i+1}")
                    # Fallback pode ser implementado aqui
                    continue
                
                # Processa cada personagem detectado na referência
                for j, det in enumerate(char_detections):
                    ref_char_id = f"ref_char_{i:03d}_{j:03d}"
                    
                    # Extrai crop com contexto
                    x1, y1, x2, y2 = det.bbox
                    ctx = calculate_context_bbox(
                        (x1, y1, x2, y2), 
                        (ref_image.width, ref_image.height),
                        CONTEXT_INFLATION_FACTOR
                    )
                    crop = ref_image.crop(ctx)
                    
                    # Extrai embeddings da imagem de referência
                    try:
                        clip_emb, face_emb = identity_encoder.extract_identity(crop)
                        
                        # Salva embedding
                        db.save_character_embedding(
                            char_id=ref_char_id,
                            clip_embedding=clip_emb,
                            face_embedding=face_emb,
                            prominence_score=1.0,  # Alta prioridade
                            first_seen_page=-1  # -1 indica referência
                        )
                        
                        # Extrai paleta de cores da IMAGEM COLORIDA
                        palette = palette_extractor.extract(crop, character_hint=ref_char_id)
                        palette.character_id = ref_char_id
                        palette.source_page = -1  # -1 indica referência colorida
                        palette.is_color_reference = True  # Marca como referência
                        palette.extracted_at = datetime.now().isoformat()
                        
                        # SALVA a paleta
                        db.save_character_palette(char_id=ref_char_id, palette=palette)
                        
                        logger.info(f" → {ref_char_id}: paleta extraída ✓")
                        
                    except Exception as e:
                        logger.error(f"Erro ao processar referência {ref_char_id}: {e}")
                        
            except Exception as e:
                logger.error(f"Erro ao processar imagem de referência {ref_path}: {e}")
        
        logger.info(f"Processamento de referências concluído\n")
    
    def generate_page(
        self, 
        page_num: int, 
        options: Optional[GenerationOptions] = None
    ) -> Image:
        """
        PASSO 2: Geração de uma página específica.
        
        Usa dados do Passo 1 para gerar a página colorizada.
        Pode ser chamado em qualquer ordem (navegação não-linear).
        
        Args:
            page_num: Índice da página (0-based)
            options: Opções de geração (opcional)
            
        Returns:
            PIL.Image: Página colorizada
            
        Raises:
            GenerationError: Se houver erro na geração
            ValueError: Se página não existir ou Passo 1 não executado
        """
        if not self._chapter_loaded:
            raise ValueError("Execute process_chapter() antes de generate_page()")
        
        if page_num < 0 or page_num >= len(self._page_paths):
            raise ValueError(f"Página {page_num} fora do range [0, {len(self._page_paths)})")
        
        options = options or GenerationOptions()
        
        logger.info(f"Gerando página {page_num + 1}")
        
        db = self._get_database(self._current_chapter_id)
        generator = self._get_generator()
        
        # Carrega dados do Passo 1
        page_data = db.get_page_analysis(page_num)
        
        if page_data is None:
            raise ValueError(f"Página {page_num} não encontrada no cache. Execute Pass 1 novamente.")
        
        chapter_context = db.get_chapter_context(page_num)
        
        # Geração
        self._notify_progress(page_num, "generating_background", 10)
        
        try:
            # Prepara opções como dict para Pass2Generator
            preserve_text = getattr(options, 'preserve_original_text', False)
            gen_options = {
                'chapter_context': chapter_context,
                'scene_type': getattr(page_data, 'scene_type', 'PRESENT'),
                'style_preset': getattr(options, 'style_preset', 'default'),
                'quality_mode': getattr(options, 'quality_mode', 'balanced'),
                'preserve_original_text': preserve_text,
                'text_compositing': preserve_text,  # Alias for TileAwareGenerator
                'seed': getattr(options, 'seed', None),
                'guidance_scale': getattr(options, 'guidance_scale', 7.5),
                'steps': getattr(options, 'num_inference_steps', 20),
            }
            
            result = generator.generate_page(
                page_num=page_num,
                options=gen_options
            )
            
            # Atualiza database com refinamentos
            db.update_generation(page_num, result)
            
            self._notify_progress(page_num, "complete", 100)
            
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na geração da página {page_num}: {e}")
            import traceback
            traceback.print_exc()
            raise GenerationError(f"Falha na geração da página {page_num}: {e}")
    
    def generate_all(
        self, 
        output_dir: str, 
        options: Optional[GenerationOptions] = None
    ) -> List[Image.Image]:
        """
        Gera todas as páginas em sequência.
        
        Args:
            output_dir: Diretório para salvar resultados
            options: Opções de geração (opcional)
            
        Returns:
            Lista de PIL.Image com todas as páginas
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i in range(len(self._page_paths)):
            try:
                result = self.generate_page(i, options)
                results.append(result)
                
                # Salva
                output_file = output_path / f"page_{i+1:03d}.png"
                result.save(output_file)
                logger.info(f"Salvo: {output_file}")
                
            except GenerationError as e:
                logger.error(f"ERRO CRÍTICO na página {i+1}: {e}")
                logger.warning(f"Pulando página {i+1} para continuar o lote...")
                # Opcional: Criar imagem de erro?
                continue
            except Exception as e:
                 logger.error(f"Erro inesperado na página {i+1}: {e}")
                 continue
        
        return results
    
    def set_scene_context(
        self, 
        page_range: Tuple[int, int], 
        context_type: str = SceneType.PRESENT.value
    ):
        """
        Define contexto narrativo manualmente.
        
        Args:
            page_range: (início, fim) - índices das páginas
            context_type: Tipo de cena ("present", "flashback", "dream", etc)
        """
        if not self._chapter_loaded:
            raise ValueError("Execute process_chapter() primeiro")
        
        start, end = page_range
        db = self._get_database(self._current_chapter_id)
        
        for page_num in range(start, end + 1):
            db.set_scene_context(page_num, context_type)
        
        logger.info(f"Contexto '{context_type}' definido para páginas {start}-{end}")
    
    def _generate_chapter_id(self, page_paths: List[str]) -> str:
        """Gera ID único para o capítulo baseado nos paths"""
        import hashlib
        content = "".join(page_paths).encode()
        return hashlib.md5(content).hexdigest()[:12]
    
    def _create_chapter_analysis(
        self, 
        db, 
        num_pages: int
    ) -> ChapterAnalysis:
        """Cria objeto ChapterAnalysis a partir do database"""
        summary = db.get_summary()
        
        # Estima tempo de geração
        est_time_per_page = 8.0 if self.device == "cuda" else 60.0
        
        return ChapterAnalysis(
            chapter_id=db.chapter_id,
            num_pages=num_pages,
            num_characters=summary.get('num_characters', 0),
            characters=summary.get('characters', []),
            scene_breakdown=summary.get('scene_breakdown', {}),
            estimated_generation_time=num_pages * est_time_per_page
        )


