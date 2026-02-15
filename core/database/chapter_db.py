"""
MangaAutoColor Pro - Chapter Database (Two-Pass)

Persistência híbrida para Two-Pass:
- FAISS: Indexação vetorial rápida para busca por similaridade
- Parquet: Metadados estruturados (detecções, tiles, paletas)
- .pt files: Tensores de embeddings (cache imutável)

Baseado na arquitetura descrita em ARCHITECTURE.md
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
from core.database.vector_index import VectorIndex
from core.constants import SceneType
from core.logging.setup import get_logger

logger_db = get_logger("ChapterDatabase")

# Tenta importar FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger_db.warning("FAISS não disponível. Usando fallback numpy.")


@dataclass
class CharacterRecord:
    """Registro de personagem no banco de dados (ADR 004: com SAM e Z-Buffer)"""
    char_id: str
    clip_embedding_path: str
    face_embedding_path: Optional[str]
    prominence_score: float
    first_seen_page: int
    bbox_count: int
    # Paletas de cores extraídas
    palette_hair: Optional[List[Tuple[int, int, int]]] = None
    palette_skin: Optional[List[Tuple[int, int, int]]] = None
    palette_eyes: Optional[List[Tuple[int, int, int]]] = None
    palette_clothes: Optional[List[Tuple[int, int, int]]] = None
    created_at: str = None
    # ADR 004: Dados de segmentação e profundidade
    sam_mask_rle: Optional[str] = None  # Máscara SAM em RLE
    mask_shape: Optional[Tuple[int, int]] = None  # (height, width)
    depth_score: float = 0.0  # Score de profundidade (menor = mais à frente)
    depth_rank: int = 0  # Ordem de profundidade (1 = mais à frente)
    bbox: Optional[Tuple[int, int, int, int]] = None  # Bounding box (x1, y1, x2, y2)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class TileJob:
    """Job de processamento de tile"""
    page_num: int
    tile_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    active_char_ids: List[str]  # Top-K personagens neste tile
    mask_paths: Dict[str, str]  # char_id -> path da máscara gaussiana
    canny_path: str  # Path para edges Canny pré-computados


@dataclass
class PageAnalysis:
    """Resultado da análise de uma página (Pass 1)
    
    ADR 004: Inclui depth_order para ordenação de profundidade.
    ADR 005: Inclui attention_masks e temporal_data para PCTC.
    """
    page_num: int
    image_path: str
    detections: List[Dict]  # Detecções YOLO com bbox, class_id, etc.
    character_ids: List[str]  # IDs de personagens encontrados
    processed: bool = False
    scene_type: str = SceneType.PRESENT.value  # Tipo de cena: present, flashback, dream, nightmare
    # ADR 004: Ordenação de profundidade para Pass 2
    depth_order: List[str] = None  # char_ids da frente para o fundo
    def __post_init__(self):
        if self.depth_order is None:
            self.depth_order = []


class ChapterDatabase:
    """
    Database híbrido para Two-Pass System.
    
    Estrutura de diretórios:
    chapter_cache/
    └── {chapter_id}/
        ├── embeddings/          # .pt files (tensores imutáveis)
        │   ├── char_001.pt
        │   └── char_002.pt
        ├── masks/               # Máscaras Gaussianas (.npy)
        │   ├── page_0_tile_0_char_001.npy
        │   └── page_0_tile_0_char_002.npy
        ├── canny/               # Edges Canny pré-computados
        │   └── page_0_canny.npy
        ├── characters.parquet   # Metadados dos personagens
        ├── tiles.parquet        # TileJobs pré-computados
        ├── pages.parquet        # Análise das páginas
        └── index.faiss          # Índice FAISS para embeddings
    """
    
    def __init__(self, chapter_id: str, cache_root: Optional[str] = None):
        from config.settings import CHAPTER_CACHE_DIR
        self.chapter_id = chapter_id
        self.cache_dir = Path(cache_root or CHAPTER_CACHE_DIR) / chapter_id
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.masks_dir = self.cache_dir / "masks"
        self.canny_dir = self.cache_dir / "canny"
        
        # Cria diretórios
        for dir_path in [self.cache_dir, self.embeddings_dir, self.masks_dir, self.canny_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # DataFrames em memória (lazy load)
        self._characters_df: Optional[pd.DataFrame] = None
        self._tiles_df: Optional[pd.DataFrame] = None
        self._pages_df: Optional[pd.DataFrame] = None
        
        # FAISS index wrapper
        self.embedding_dim = 768
        self._vector_index = VectorIndex(self.embedding_dim)
        
        logger_db.info(f"Inicializado: {self.cache_dir}")
    
    def exists(self) -> bool:
        """
        Verifica se o capítulo já foi analisado (tem dados persistidos).
        
        Returns:
            True se existem embeddings ou páginas analisadas
        """
        # Verifica se há arquivos de embedding
        if self.embeddings_dir.exists():
            embedding_files = list(self.embeddings_dir.glob("*.pt"))
            if embedding_files:
                return True
        
        # Verifica se há parquet de páginas
        pages_parquet = self.cache_dir / "pages.parquet"
        if pages_parquet.exists():
            return True
        
        return False
    
    # =========================================================================
    # EMBEDDINGS (FAISS + .pt files)
    # =========================================================================
    
    def save_character_embedding(
        self,
        char_id: str,
        clip_embedding: torch.Tensor,
        face_embedding: Optional[torch.Tensor] = None,
        prominence_score: float = 0.0,
        first_seen_page: int = 0
    ) -> str:
        """
        Salva embedding de personagem.
        
        Args:
            char_id: ID único do personagem
            clip_embedding: Tensor CLIP (768-dim)
            face_embedding: Tensor ArcFace opcional (512-dim)
            prominence_score: Score de prominência
            first_seen_page: Primeira página onde aparece
            
        Returns:
            Path do arquivo salvo
        """
        # Salva em .pt (imutável)
        clip_path = self.embeddings_dir / f"{char_id}_clip.pt"
        torch.save({
            'embedding': clip_embedding.cpu(),
            'char_id': char_id,
            'created_at': datetime.now().isoformat()
        }, clip_path)
        
        face_path = None
        if face_embedding is not None and hasattr(face_embedding, 'cpu'):
            face_path = self.embeddings_dir / f"{char_id}_face.pt"
            torch.save({
                'embedding': face_embedding.cpu(),
                'char_id': char_id,
                'created_at': datetime.now().isoformat()
            }, face_path)
        
        # Adiciona ao Vector Index
        self._vector_index.add(char_id, clip_embedding)
        
        # Atualiza DataFrame de personagens
        record = CharacterRecord(
            char_id=char_id,
            clip_embedding_path=str(clip_path),
            face_embedding_path=str(face_path) if face_path else None,
            prominence_score=prominence_score,
            first_seen_page=first_seen_page,
            bbox_count=1
        )
        self._append_character_record(record)
        
        return str(clip_path)
    
    # ADR 004: Métodos de segmentação SAM ======================================
    
    def _update_character_segmentation(
        self,
        char_id: str,
        sam_mask_rle: Optional[str] = None,
        mask_shape: Optional[Tuple[int, int]] = None,
        depth_score: float = 0.0,
        depth_rank: int = 0,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Atualiza registro do personagem com dados de segmentação (ADR 004).
        
        Args:
            char_id: ID do personagem
            sam_mask_rle: Máscara SAM em formato RLE
            mask_shape: Dimensões da máscara (h, w)
            depth_score: Score de profundidade calculado
            depth_rank: Ordem de profundidade (1 = mais à frente)
            bbox: Bounding box (x1, y1, x2, y2)
        """
        if self._characters_df is None:
            return
        
        mask = self._characters_df['char_id'] == char_id
        if not mask.any():
            return
        
        idx = self._characters_df[mask].index[0]
        
        # Atualiza campos SAM e profundidade
        if sam_mask_rle is not None:
            self._characters_df.at[idx, 'sam_mask_rle'] = sam_mask_rle
        if mask_shape is not None:
            self._characters_df.at[idx, 'mask_shape'] = mask_shape
        if bbox is not None:
            self._characters_df.at[idx, 'bbox'] = bbox
        
        self._characters_df.at[idx, 'depth_score'] = depth_score
        self._characters_df.at[idx, 'depth_rank'] = depth_rank
    
    def get_character_mask(self, char_id: str) -> Optional[np.ndarray]:
        """
        Carrega máscara SAM de um personagem.
        
        ADR 004: Decodifica máscara RLE do registro do personagem.
        
        Args:
            char_id: ID do personagem
            
        Returns:
            Máscara numpy ou None se não encontrada
        """
        if self._characters_df is None:
            self._load_parquets()
        
        if self._characters_df is None:
            return None
        
        mask = self._characters_df['char_id'] == char_id
        if not mask.any():
            return None
        
        row = self._characters_df[mask].iloc[0]
        rle = row.get('sam_mask_rle')
        shape = row.get('mask_shape')
        
        # Verifica se rle é válido (pode ser None, NaN, ou string)
        try:
            if rle is None or (isinstance(rle, float) and pd.isna(rle)):
                return None
        except (ValueError, TypeError):
            return None
        
        if not isinstance(rle, str) or not rle:
            return None
        
        try:
            from core.analysis.segmentation import RLECodec
            # mask_shape pode voltar do Parquet como numpy array, list ou tuple
            if shape is not None:
                try:
                    h, w = int(shape[0]), int(shape[1])
                except (TypeError, IndexError, ValueError):
                    h, w = 0, 0
            else:
                h, w = 0, 0
            return RLECodec.decode(rle, h, w)
        except Exception as e:
            logger_db.warning(f"Erro ao decodificar máscara RLE: {e}")
            return None
    
    def get_depth_order(self, page_num: int) -> List[str]:
        """
        Retorna ordenação de profundidade para uma página.
        
        ADR 004: Lista char_ids ordenados da frente para o fundo.
        
        Prioridade:
        1. depth_order salvo em PageAnalysis (fonte primária)
        2. Fallback: personagens da página ordenados por depth_rank
        
        Args:
            page_num: Número da página
            
        Returns:
            Lista de char_ids ordenados por profundidade
        """
        # 1. Tenta obter depth_order de PageAnalysis (fonte primária)
        if self._pages_df is None:
            self._load_parquets()
        
        if self._pages_df is not None and not self._pages_df.empty:
            page = self._pages_df[self._pages_df['page_num'] == page_num]
            if not page.empty:
                row = page.iloc[0]
                if 'depth_order' in row:
                    try:
                        depth_order = json.loads(row['depth_order'])
                        if depth_order:
                            return depth_order
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        # 2. Fallback: ordena personagens da página por depth_rank
        if self._characters_df is None:
            self._load_parquets()
        
        if self._characters_df is None:
            return []
        
        # Filtra personagens da página
        page_chars = self._characters_df[
            self._characters_df['first_seen_page'] == page_num
        ]
        
        if page_chars.empty:
            return []
        
        # Ordena por depth_rank (menor = mais à frente)
        page_chars = page_chars.sort_values('depth_rank')
        
        return page_chars['char_id'].tolist()
    
    def find_similar_characters(self, query_embedding, top_k=5, threshold=0.8):
        """
        Busca personagens similares por embedding.
        
        Args:
            query_embedding: Embedding de consulta
            top_k: Número de resultados
            threshold: Limiar de similaridade (0-1)
            
        Returns:
            Lista de (char_id, score)
        """
        return self._vector_index.search(query_embedding, top_k, threshold)
    
    def load_embedding(self, char_id: str, embedding_type: str = "clip") -> Optional[torch.Tensor]:
        """Carrega embedding do disco"""
        path = self.embeddings_dir / f"{char_id}_{embedding_type}.pt"
        if not path.exists():
            return None
        
        # Cache files may contain numpy arrays, so we need weights_only=False
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            data = torch.load(path, map_location='cpu', weights_only=False)
        return data['embedding']
    
    def save_character_palette(
        self,
        char_id: str,
        palette: 'CharacterPalette'
    ):
        """
        Salva paleta de cores do personagem.
        
        Args:
            char_id: ID do personagem
            palette: CharacterPalette com as cores extraídas
        """
        # SALVAR SEMPRE o JSON da paleta (incluindo referências coloridas)
        # Isso garante que referências (ref_char_XXX) também sejam persistidas
        palette_path = self.embeddings_dir / f"{char_id}_palette.json"
        with open(palette_path, 'w') as f:
            json.dump(palette.to_dict(), f, indent=2)
        
        # Atualiza o registro do personagem com as paletas (se existir no DataFrame)
        if self._characters_df is not None:
            mask = self._characters_df['char_id'] == char_id
            if mask.any():
                # Extrai cores das regiões
                hair_color = palette.get_color('hair', None)
                skin_color = palette.get_color('skin', None)
                eyes_color = palette.get_color('eyes', None)
                clothes_color = palette.get_color('clothes_primary', None)
                
                # Atualiza DataFrame
                idx = self._characters_df[mask].index[0]
                if hair_color:
                    self._characters_df.at[idx, 'palette_hair'] = [hair_color]
                if skin_color:
                    self._characters_df.at[idx, 'palette_skin'] = [skin_color]
                if eyes_color:
                    self._characters_df.at[idx, 'palette_eyes'] = [eyes_color]
                if clothes_color:
                    self._characters_df.at[idx, 'palette_clothes'] = [clothes_color]
    
    def load_character_palette(self, char_id: str) -> Optional['CharacterPalette']:
        """Carrega paleta do personagem do disco"""
        from core.identity.palette_manager import CharacterPalette
        
        palette_path = self.embeddings_dir / f"{char_id}_palette.json"
        if not palette_path.exists():
            return None
        
        with open(palette_path, 'r') as f:
            data = json.load(f)
        
        return CharacterPalette.from_dict(data)
    
    def load_all_palettes(self) -> Dict[str, 'CharacterPalette']:
        """Carrega todas as paletas do capítulo (incluindo referências)"""
        from core.identity.palette_manager import CharacterPalette
        
        palettes = {}
        if not self.embeddings_dir.exists():
            return palettes
        
        for palette_file in self.embeddings_dir.glob("*_palette.json"):
            char_id = palette_file.stem.replace("_palette", "")
            try:
                with open(palette_file, 'r') as f:
                    data = json.load(f)
                palettes[char_id] = CharacterPalette.from_dict(data)
            except Exception as e:
                logger_db.warning(f"Erro ao carregar paleta {char_id}: {e}")
        
        return palettes
    
    def load_reference_palettes(self) -> Dict[str, 'CharacterPalette']:
        """Carrega apenas paletas de imagens de referência coloridas"""
        all_palettes = self.load_all_palettes()
        return {
            char_id: palette
            for char_id, palette in all_palettes.items()
            if char_id.startswith('ref_char_') or 
               getattr(palette, 'is_color_reference', False) or
               getattr(palette, 'source_page', 0) == -1
        }
    
    # =========================================================================
    # METADADOS (Parquet)
    # =========================================================================
    
    def _append_character_record(self, record: CharacterRecord):
        """Adiciona registro de personagem ao DataFrame"""
        new_row = pd.DataFrame([asdict(record)])
        
        if self._characters_df is None:
            self._characters_df = new_row
        else:
            self._characters_df = pd.concat([self._characters_df, new_row], ignore_index=True)
    
    def save_characters_from_analysis(self, characters: List[Dict], page_num: int):
        """
        Salva personagens detectados durante análise do Pass 1.
        
        ADR 004: Agora inclui máscaras SAM (RLE) e dados de profundidade.
        
        Args:
            characters: Lista de dicts com dados do personagem (de Pass1Analyzer)
            page_num: Número da página onde foram detectados
            
        Returns:
            Lista de char_ids gerados
        """
        import uuid
        
        char_ids = []
        
        for i, char_data in enumerate(characters):
            # Usa char_id existente (ADR 004) ou gera novo
            char_id = char_data.get('char_id')
            if char_id is None:
                char_id = f"char_{page_num:03d}_{i:03d}_{uuid.uuid4().hex[:6]}"
            
            # Extrai embeddings (podem ser listas ou tensores)
            clip_emb = char_data.get('embedding') or char_data.get('body_embedding')
            face_emb = char_data.get('face_embedding')
            
            # Converte listas para tensores se necessário
            if clip_emb is not None:
                if isinstance(clip_emb, list):
                    clip_emb = torch.tensor(clip_emb)
                elif not isinstance(clip_emb, torch.Tensor):
                    clip_emb = torch.tensor(clip_emb)
            
            if face_emb is not None:
                if isinstance(face_emb, list):
                    face_emb = torch.tensor(face_emb)
                elif not isinstance(face_emb, torch.Tensor):
                    face_emb = torch.tensor(face_emb)
            
            # Calcula prominence do bbox
            bbox = char_data.get('bbox', (0, 0, 100, 100))
            prominence = char_data.get('confidence', 0.5)
            
            # ADR 004: Extrai dados de segmentação e profundidade
            sam_mask_rle = char_data.get('sam_mask_rle')
            mask_shape = char_data.get('mask_shape')
            depth_score = char_data.get('depth_score', 0.0)
            depth_rank = char_data.get('depth_rank', 0)
            
            # Salva embedding
            if clip_emb is not None:
                try:
                    self.save_character_embedding(
                        char_id=char_id,
                        clip_embedding=clip_emb,
                        face_embedding=face_emb,
                        prominence_score=prominence,
                        first_seen_page=page_num
                    )
                except Exception as e:
                    logger_db.error(f"Erro ao salvar embedding de {char_id}: {e}")
                    continue
            
            # Salva paleta se existir
            palette = char_data.get('palette')
            if palette is not None:
                try:
                    self.save_character_palette(char_id, palette)
                except Exception as e:
                    logger_db.warning(f"Erro ao salvar paleta de {char_id}: {e}")
            
            # ADR 004: Atualiza registro com dados SAM e profundidade
            self._update_character_segmentation(
                char_id=char_id,
                sam_mask_rle=sam_mask_rle,
                mask_shape=mask_shape,
                depth_score=depth_score,
                depth_rank=depth_rank,
                bbox=bbox
            )
            
            # Adiciona char_id ao dict original para referência
            char_data['character_id'] = char_id
            char_ids.append(char_id)
            
            logger_db.debug(f"Personagem {char_id} salvo (página {page_num}, depth_rank={depth_rank})")
        
        return char_ids
    
    def save_tile_job(self, tile_job: TileJob):
        """Salva TileJob para processamento no Pass 2"""
        if self._tiles_df is None:
            self._tiles_df = pd.DataFrame()
        
        new_row = pd.DataFrame([{
            'page_num': tile_job.page_num,
            'tile_bbox': json.dumps(tile_job.tile_bbox),
            'active_char_ids': json.dumps(tile_job.active_char_ids),
            'mask_paths': json.dumps(tile_job.mask_paths),
            'canny_path': tile_job.canny_path
        }])
        
        self._tiles_df = pd.concat([self._tiles_df, new_row], ignore_index=True)
    
    def get_tile_jobs(self, page_num: int) -> List[TileJob]:
        """Retorna todos os TileJobs de uma página"""
        if self._tiles_df is None:
            self._load_parquets()
        
        if self._tiles_df is None or len(self._tiles_df) == 0:
            return []
        
        page_tiles = self._tiles_df[self._tiles_df['page_num'] == page_num]
        
        jobs = []
        for _, row in page_tiles.iterrows():
            jobs.append(TileJob(
                page_num=row['page_num'],
                tile_bbox=tuple(json.loads(row['tile_bbox'])),
                active_char_ids=json.loads(row['active_char_ids']),
                mask_paths=json.loads(row['mask_paths']),
                canny_path=row['canny_path']
            ))
        
        return jobs
    
    def save_page_analysis(self, analysis: PageAnalysis = None, **kwargs):
        """
        Salva análise da página.
        
        ADR 004: Inclui depth_order para ordenação de profundidade.
        ADR 005: Inclui attention_masks e temporal_data para PCTC.
        
        Args:
            analysis: Objeto PageAnalysis (opcional)
            **kwargs: Argumentos nomeados alternativos:
                - page_num: int
                - image_path: str
                - detections: List[Dict]
                - character_ids: List[str]
                - scene_type: str
                - processed: bool
                - depth_order: List[str]  # ADR 004
                - attention_masks: Dict[str, np.ndarray]  # ADR 005
                - temporal_data: Dict  # ADR 005
        """
        if self._pages_df is None:
            self._pages_df = pd.DataFrame()
        
        # Se recebeu objeto PageAnalysis
        if analysis is not None:
            page_num = analysis.page_num
            image_path = analysis.image_path
            detections = analysis.detections
            character_ids = analysis.character_ids
            scene_type = getattr(analysis, 'scene_type', SceneType.PRESENT.value)
            processed = analysis.processed
            depth_order = getattr(analysis, 'depth_order', [])  # ADR 004
        else:
            # Extrai de kwargs
            page_num = kwargs.get('page_num')
            image_path = kwargs.get('image_path')
            detections = kwargs.get('detections', [])
            character_ids = kwargs.get('character_ids', [])
            scene_type = kwargs.get('scene_type', SceneType.PRESENT.value)
            processed = kwargs.get('processed', True)
            depth_order = kwargs.get('depth_order', [])  # ADR 004
        
        new_row = pd.DataFrame([{
            'page_num': page_num,
            'image_path': image_path,
            'detections': json.dumps(detections),
            'character_ids': json.dumps(character_ids),
            'scene_type': scene_type,
            'processed': processed,
            'depth_order': json.dumps(depth_order),  # ADR 004
        }])
        
        self._pages_df = pd.concat([self._pages_df, new_row], ignore_index=True)
    
    def get_page_analysis(self, page_num: int) -> Optional[PageAnalysis]:
        """
        Retorna análise de uma página.
        
        ADR 004: Inclui depth_order se disponível.
        ADR 005: Inclui attention_masks e temporal_data se disponíveis.
        """
        if self._pages_df is None:
            self._load_parquets()
        
        if self._pages_df is None:
            return None
        
        page = self._pages_df[self._pages_df['page_num'] == page_num]
        if len(page) == 0:
            return None
        
        row = page.iloc[0]
        
        # ADR 004: Carrega depth_order
        depth_order = []
        if 'depth_order' in row:
            try:
                depth_order = json.loads(row['depth_order'])
            except:
                depth_order = []
        
        # ADR 005: Carrega attention_masks
        attention_masks = {}
        if 'attention_mask_paths' in row:
            try:
                mask_paths = json.loads(row['attention_mask_paths'])
                for char_id, mask_path in mask_paths.items():
                    if Path(mask_path).exists():
                        attention_masks[char_id] = np.load(mask_path)
            except Exception as e:
                logger_db.debug(f"Erro ao carregar attention masks: {e}")
        
        # ADR 005: Carrega temporal_data
        temporal_data = {}
        if 'temporal_data' in row:
            try:
                temporal_data = json.loads(row['temporal_data'])
                # Carrega arrays numpy se houver paths
                for key in list(temporal_data.keys()):
                    if key.endswith('_path'):
                        arr_path = temporal_data[key]
                        if Path(arr_path).exists():
                            arr_key = key.replace('_path', '')
                            temporal_data[arr_key] = np.load(arr_path)
            except Exception as e:
                logger_db.debug(f"Erro ao carregar temporal data: {e}")
        
        return PageAnalysis(
            page_num=row['page_num'],
            image_path=row['image_path'],
            detections=json.loads(row['detections']),
            character_ids=json.loads(row['character_ids']),
            processed=row['processed'],
            scene_type=row.get('scene_type', SceneType.PRESENT.value),
            depth_order=depth_order,  # ADR 004
        )
    
    def get_chapter_context(self, page_num: int) -> Dict:
        """
        Retorna contexto narrativo ao redor da página.
        Útil para manter consistência entre páginas.
        """
        if self._pages_df is None:
            self._load_parquets()
            
        context = {
            'prev_scene': None,
            'next_scene': None,
            'chapter_progress': 0.0
        }
        
        if self._pages_df is not None and not self._pages_df.empty:
            # Garante ordenação
            df = self._pages_df.sort_values('page_num').reset_index(drop=True)
            
            # Encontra índice atual
            matches = df.index[df['page_num'] == page_num].tolist()
            if matches:
                idx = matches[0]
                total = len(df)
                context['chapter_progress'] = (idx + 1) / total
                
                # Cena anterior
                if idx > 0:
                    context['prev_scene'] = df.iloc[idx-1].get('scene_type', SceneType.PRESENT.value)
                
                # Próxima cena
                if idx < total - 1:
                    context['next_scene'] = df.iloc[idx+1].get('scene_type', SceneType.PRESENT.value)
                    
        return context

    def update_generation(self, page_num: int, result: Any):
        """
        Atualiza o status da página após geração.
        """
        if self._pages_df is None:
            self._load_parquets()
            
        if self._pages_df is not None and not self._pages_df.empty:
            # Encontra índice e atualiza (se houvesse coluna de status de geração)
            # Por enquanto apenas loga ou passa
            pass

    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================
    
    def save_all(self):
        """Persiste todos os DataFrames em Parquet"""
        # Salva Parquets
        if self._characters_df is not None:
            self._characters_df.to_parquet(self.cache_dir / "characters.parquet")
            logger_db.info(f"{len(self._characters_df)} personagens salvos")
        
        if self._tiles_df is not None:
            self._tiles_df.to_parquet(self.cache_dir / "tiles.parquet")
            logger_db.info(f"{len(self._tiles_df)} tiles salvos")
        
        if self._pages_df is not None:
            self._pages_df.to_parquet(self.cache_dir / "pages.parquet")
            logger_db.info(f"{len(self._pages_df)} páginas salvas")
        
        # Salva FAISS index e ID map
        self._vector_index.save(self.cache_dir)
        
        logger_db.info(f"Cache persistido em: {self.cache_dir}")
    
    def _load_parquets(self):
        """Carrega DataFrames do disco"""
        # Carrega Parquets
        chars_path = self.cache_dir / "characters.parquet"
        if chars_path.exists():
            self._characters_df = pd.read_parquet(chars_path)
        
        tiles_path = self.cache_dir / "tiles.parquet"
        if tiles_path.exists():
            self._tiles_df = pd.read_parquet(tiles_path)
        
        pages_path = self.cache_dir / "pages.parquet"
        if pages_path.exists():
            self._pages_df = pd.read_parquet(pages_path)
        
        # Carrega FAISS index e ID map
        self._vector_index.load(self.cache_dir)
    
    def load_all(self):
        """Carrega todo o cache do disco"""
        self._load_parquets()
        logger_db.info(f"Cache carregado de: {self.cache_dir}")
    
    # =========================================================================
    # UTILIDADES
    # =========================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo do capítulo"""
        if self._characters_df is None:
            self._load_parquets()
        
        return {
            'chapter_id': self.chapter_id,
            'total_characters': len(self._characters_df) if self._characters_df is not None else 0,
            'total_tiles': len(self._tiles_df) if self._tiles_df is not None else 0,
            'total_pages': len(self._pages_df) if self._pages_df is not None else 0,
            'cache_dir': str(self.cache_dir)
        }
    
    def list_characters(self) -> List[str]:
        """Lista todos os IDs de personagens"""
        if self._characters_df is None:
            self._load_parquets()
        
        if self._characters_df is None:
            return []
        
        return self._characters_df['char_id'].tolist()
    
    def exists(self) -> bool:
        """Verifica se o capítulo já foi processado (Pass 1)"""
        return (self.cache_dir / "characters.parquet").exists()
    
    def clear(self):
        """Remove todo o cache do capítulo"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger_db.info(f"Cache removido: {self.cache_dir}")
