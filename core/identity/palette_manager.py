"""
MangaAutoColor Pro - Gerenciamento de Paletas

Implementa extração e gerenciamento de paletas de cores por personagem.
Usa espaço de cor CIELAB para cálculo perceptualmente uniforme de Delta E.

Funcionalidades:
- Extração de paleta por região (cabelo, pele, olhos, roupa)
- Cálculo de Delta E para consistência temporal
- Suavização temporal entre páginas
- Cache de paletas por personagem

Referências:
- CIELAB: https://en.wikipedia.org/wiki/CIELAB_color_space
- Delta E: https://en.wikipedia.org/wiki/Color_difference
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
from dataclasses import dataclass, field
from collections import defaultdict
import json
from sklearn.cluster import KMeans

try:
    from skimage.color import rgb2lab, lab2rgb
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from config.settings import (
    PALETTE_REGIONS, PALETTE_COLORS_PER_REGION, PALETTE_DRIFT_THRESHOLD,
    TEMPORAL_SMOOTHING, COLOR_SPACE, VERBOSE, DATA_DIR
)


@dataclass
class ColorRegion:
    """
    Representa uma região de cor extraída.
    
    Attributes:
        region_name: Nome da região (hair, skin, eyes, etc)
        dominant_color: Cor dominante em RGB (0-255)
        colors: Lista de cores principais em RGB
        percentages: Porcentagem de cada cor
        confidence: Confiança na extração
    """
    region_name: str
    dominant_color: Tuple[int, int, int]
    colors: List[Tuple[int, int, int]] = field(default_factory=list)
    percentages: List[float] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'region_name': self.region_name,
            'dominant_color': self.dominant_color,
            'colors': self.colors,
            'percentages': self.percentages,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ColorRegion':
        return cls(
            region_name=data['region_name'],
            dominant_color=tuple(data['dominant_color']),
            colors=[tuple(c) for c in data.get('colors', [])],
            percentages=data.get('percentages', []),
            confidence=data.get('confidence', 0.0)
        )


@dataclass
class CharacterPalette:
    """
    Paleta completa de um personagem.
    
    Attributes:
        character_id: ID do personagem
        regions: Dicionário de região -> ColorRegion
        extracted_at: Timestamp da extração
        source_page: Página de origem (-1 = imagem de referência colorida)
        is_color_reference: True se paleta veio de imagem de referência colorida
    """
    character_id: str
    regions: Dict[str, ColorRegion] = field(default_factory=dict)
    extracted_at: Optional[str] = None
    source_page: int = 0
    is_color_reference: bool = False
    
    def get_color(self, region: str, default: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[int, int, int]:
        """Retorna cor de uma região ou default se não existir"""
        if region in self.regions:
            return self.regions[region].dominant_color
        return default
    
    def to_dict(self) -> Dict:
        return {
            'character_id': self.character_id,
            'regions': {k: v.to_dict() for k, v in self.regions.items()},
            'extracted_at': self.extracted_at,
            'source_page': self.source_page,
            'is_color_reference': self.is_color_reference
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CharacterPalette':
        regions = {
            k: ColorRegion.from_dict(v) 
            for k, v in data.get('regions', {}).items()
        }
        return cls(
            character_id=data['character_id'],
            regions=regions,
            extracted_at=data.get('extracted_at'),
            source_page=data.get('source_page', 0),
            is_color_reference=data.get('is_color_reference', False)
        )


class PaletteExtractor:
    """
    Extrator de paletas de cores de personagens.
    
    Usa clustering K-means para identificar cores dominantes em
    diferentes regiões da imagem do personagem.
    
    Args:
        n_colors: Número de cores por região
        color_space: Espaço de cor para processamento (CIELAB ou RGB)
    """
    
    def __init__(
        self,
        n_colors: int = PALETTE_COLORS_PER_REGION,
        color_space: str = COLOR_SPACE
    ):
        self.n_colors = n_colors
        self.color_space = color_space
        
        if VERBOSE:
            print(f"[PaletteExtractor] Inicializado (n_colors={n_colors}, "
                  f"color_space={color_space})")
    
    def extract(
        self,
        image: Union[Image.Image, np.ndarray],
        character_hint: Optional[str] = None
    ) -> CharacterPalette:
        """
        Extrai paleta completa de uma imagem de personagem.
        
        Args:
            image: Imagem do personagem (PIL ou numpy)
            character_hint: Dica sobre o personagem (opcional)
            
        Returns:
            CharacterPalette com todas as regiões
        """
        # Converte para numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Garante RGB
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Inicializa paleta
        palette = CharacterPalette(
            character_id="temp",
            source_page=0
        )
        
        # Extrai regiões usando heurísticas de posição
        region_masks = self._segment_regions(image_np)
        
        for region_name, mask in region_masks.items():
            color_region = self._extract_region_colors(image_np, mask, region_name)
            if color_region:
                palette.regions[region_name] = color_region
        
        return palette
    
    def _segment_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segmenta imagem em regiões (cabelo, pele, olhos, roupa).
        
        Usa heurísticas de posição e cor para segmentação básica.
        
        Args:
            image: Imagem RGB
            
        Returns:
            Dicionário de máscaras por região
        """
        h, w = image.shape[:2]
        
        # Converte para HSV para análise de cor
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        regions = {}
        
        # 1. Pele: tons de bege/rosado, saturação média-baixa
        # Região central-superior (rosto)
        skin_lower = np.array([0, 20, 70])
        skin_upper = np.array([30, 170, 255])
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        
        # Foco na região central superior (onde geralmente está o rosto)
        face_region = np.zeros_like(skin_mask)
        fy1, fy2 = int(h * 0.1), int(h * 0.5)
        fx1, fx2 = int(w * 0.2), int(w * 0.8)
        face_region[fy1:fy2, fx1:fx2] = 255
        skin_mask = cv2.bitwise_and(skin_mask, face_region)
        
        regions['skin'] = skin_mask
        
        # 2. Cabelo: topo da imagem, cores diversas
        hair_region = np.zeros((h, w), dtype=np.uint8)
        hair_region[:int(h * 0.4), :] = 255  # Topo 40%
        
        # Exclui pele do cabelo
        hair_mask = cv2.bitwise_and(hair_region, cv2.bitwise_not(skin_mask))
        regions['hair'] = hair_mask
        
        # 3. Olhos: pequenas regiões na área do rosto
        eye_region = np.zeros((h, w), dtype=np.uint8)
        # Centro do rosto
        eye_y = int(h * 0.25)
        eye_region[eye_y-20:eye_y+20, int(w*0.3):int(w*0.7)] = 255
        eye_mask = cv2.bitwise_and(eye_region, skin_mask)
        regions['eyes'] = eye_mask
        
        # 4. Roupa: parte inferior
        clothes_region = np.zeros((h, w), dtype=np.uint8)
        clothes_region[int(h * 0.5):, :] = 255  # Metade inferior
        clothes_mask = cv2.bitwise_and(clothes_region, cv2.bitwise_not(skin_mask))
        regions['clothes_primary'] = clothes_mask
        
        # 5. Acessórios: pequenas regiões destacadas
        # Simplificação: usa edge detection para encontrar detalhes
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        regions['accessories'] = edges
        
        return regions
    
    def _extract_region_colors(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        region_name: str
    ) -> Optional[ColorRegion]:
        """
        Extrai cores dominantes de uma região usando K-means.
        
        Args:
            image: Imagem RGB
            mask: Máscara da região
            region_name: Nome da região
            
        Returns:
            ColorRegion ou None se região vazia
        """
        # Extrai pixels da região
        pixels = image[mask > 0]
        
        if len(pixels) < 10:
            return None
        
        # Amostragem para performance
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # Converte para espaço de cor apropriado
        if self.color_space == "CIELAB" and SKIMAGE_AVAILABLE:
            pixels_color = rgb2lab(pixels.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
        else:
            pixels_color = pixels.reshape(-1, 3)
        
        # K-means clustering
        n_clusters = min(self.n_colors, len(pixels_color))
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(pixels_color)
            
            # Cores dos centroids
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Calcula porcentagens
            percentages = []
            for i in range(n_clusters):
                pct = np.sum(labels == i) / len(labels)
                percentages.append(pct)
            
            # Converte de volta para RGB
            if self.color_space == "CIELAB" and SKIMAGE_AVAILABLE:
                colors_rgb = lab2rgb(colors.reshape(1, -1, 3)).reshape(-1, 3)
                colors_rgb = (colors_rgb * 255).astype(np.uint8)
            else:
                colors_rgb = colors.astype(np.uint8)
            
            # Cria ColorRegion
            color_list = [tuple(map(int, c)) for c in colors_rgb]
            dominant = color_list[np.argmax(percentages)]
            
            return ColorRegion(
                region_name=region_name,
                dominant_color=dominant,
                colors=color_list,
                percentages=percentages,
                confidence=min(1.0, len(pixels) / 1000)  # Confiança baseada em amostra
            )
            
        except Exception as e:
            if VERBOSE:
                print(f"[PaletteExtractor] Erro em K-means para {region_name}: {e}")
            return None
    
    def calculate_delta_e(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> float:
        """
        Calcula diferença de cor Delta E (CIE76) entre duas cores.
        
        Delta E < 1.0: imperceptível
        Delta E 1-2: perceptível por olhos treinados
        Delta E 2-10: perceptível
        Delta E > 10: cores diferentes
        
        Args:
            color1: Primeira cor em RGB
            color2: Segunda cor em RGB
            
        Returns:
            Delta E
        """
        if not SKIMAGE_AVAILABLE:
            # Fallback: distância Euclidiana em RGB
            return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5
        
        # Converte para CIELAB
        rgb1 = np.array([[color1]], dtype=np.float32) / 255.0
        rgb2 = np.array([[color2]], dtype=np.float32) / 255.0
        
        lab1 = rgb2lab(rgb1)[0, 0]
        lab2 = rgb2lab(rgb2)[0, 0]
        
        # Delta E CIE76
        delta_e = np.sum((lab1 - lab2) ** 2) ** 0.5
        
        return float(delta_e)


class PaletteManager:
    """
    Gerenciador de paletas para consistência temporal.
    
    Mantém histórico de paletas por personagem e aplica suavização
temporal para evitar flickering de cores entre páginas.
    
    Args:
        chapter_id: ID do capítulo
        cache_dir: Diretório para cache
        smoothing_factor: Fator de suavização temporal
    """
    
    def __init__(
        self,
        chapter_id: str,
        cache_dir: Path = DATA_DIR / "palettes",
        smoothing_factor: float = TEMPORAL_SMOOTHING
    ):
        self.chapter_id = chapter_id
        self.cache_dir = Path(cache_dir) / chapter_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.smoothing_factor = smoothing_factor
        self._palette_history: Dict[str, List[CharacterPalette]] = defaultdict(list)
        self._final_palettes: Dict[str, CharacterPalette] = {}
        
        self._load_cache()
    
    def register_palette(
        self,
        character_id: str,
        palette: CharacterPalette,
        page_num: int
    ):
        """
        Registra paleta de uma detecção.
        
        Args:
            character_id: ID do personagem
            palette: Paleta extraída
            page_num: Número da página
        """
        palette.character_id = character_id
        palette.source_page = page_num
        
        self._palette_history[character_id].append(palette)
    
    def consolidate_palettes(self) -> Dict[str, CharacterPalette]:
        """
        Consolida paletas de todas as detecções para cada personagem.
        
        Calcula média ponderada das cores considerando confiança e
        aplica suavização temporal.
        
        Returns:
            Dicionário de paletas finais por personagem
        """
        for char_id, palettes in self._palette_history.items():
            if not palettes:
                continue
            
            # Paleta consolidada
            consolidated = CharacterPalette(character_id=char_id)
            
            # Para cada região, calcula média ponderada
            all_regions = set()
            for p in palettes:
                all_regions.update(p.regions.keys())
            
            for region_name in all_regions:
                region_palettes = [
                    p.regions[region_name] for p in palettes 
                    if region_name in p.regions
                ]
                
                if not region_palettes:
                    continue
                
                # Pondera por confiança
                total_weight = sum(r.confidence for r in region_palettes)
                
                # Média ponderada das cores dominantes
                avg_color = [0, 0, 0]
                for r in region_palettes:
                    weight = r.confidence / total_weight
                    for i in range(3):
                        avg_color[i] += r.dominant_color[i] * weight
                
                dominant = tuple(int(c) for c in avg_color)
                
                consolidated.regions[region_name] = ColorRegion(
                    region_name=region_name,
                    dominant_color=dominant,
                    confidence=min(1.0, total_weight / len(region_palettes))
                )
            
            self._final_palettes[char_id] = consolidated
        
        self._save_cache()
        
        return self._final_palettes.copy()
    
    def get_palette(self, character_id: str) -> Optional[CharacterPalette]:
        """
        Retorna paleta consolidada de um personagem.
        
        Args:
            character_id: ID do personagem
            
        Returns:
            CharacterPalette ou None
        """
        return self._final_palettes.get(character_id)
    
    def check_drift(
        self,
        character_id: str,
        new_palette: CharacterPalette
    ) -> Dict[str, float]:
        """
        Verifica drift de cor em relação à paleta consolidada.
        
        Args:
            character_id: ID do personagem
            new_palette: Nova paleta para comparar
            
        Returns:
            Dicionário de região -> Delta E
        """
        if character_id not in self._final_palettes:
            return {}
        
        reference = self._final_palettes[character_id]
        drift = {}
        
        extractor = PaletteExtractor()
        
        for region_name, new_region in new_palette.regions.items():
            if region_name not in reference.regions:
                continue
            
            ref_color = reference.regions[region_name].dominant_color
            new_color = new_region.dominant_color
            
            delta_e = extractor.calculate_delta_e(ref_color, new_color)
            drift[region_name] = delta_e
        
        return drift
    
    def _save_cache(self):
        """Salva cache de paletas no disco"""
        cache_file = self.cache_dir / "palettes.json"
        
        data = {
            'chapter_id': self.chapter_id,
            'palettes': {
                k: v.to_dict() for k, v in self._final_palettes.items()
            },
            'history': {
                k: [p.to_dict() for p in v] 
                for k, v in self._palette_history.items()
            }
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_cache(self):
        """Carrega cache de paletas do disco"""
        cache_file = self.cache_dir / "palettes.json"
        
        if not cache_file.exists():
            return
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Carrega paletas finais
            for char_id, p_data in data.get('palettes', {}).items():
                self._final_palettes[char_id] = CharacterPalette.from_dict(p_data)
            
        except Exception as e:
            if VERBOSE:
                print(f"[PaletteManager] Erro ao carregar cache: {e}")


def generate_prompt_from_palette(palette: CharacterPalette) -> str:
    """
    Gera texto descritivo a partir da paleta.
    
    Args:
        palette: Paleta do personagem
        
    Returns:
        String descritiva das cores
    """
    descriptions = []
    
    color_map = {
        'hair': 'hair',
        'skin': 'skin',
        'eyes': 'eyes',
        'clothes_primary': 'clothes',
        'clothes_secondary': 'clothes details'
    }
    
    for region, name in color_map.items():
        if region in palette.regions:
            color = palette.regions[region].dominant_color
            # Converte RGB para nome aproximado
            color_name = _rgb_to_color_name(color)
            descriptions.append(f"{color_name} {name}")
    
    return ", ".join(descriptions)


def _rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Converte RGB para nome de cor aproximado.
    
    Args:
        rgb: Tupla (R, G, B)
        
    Returns:
        Nome da cor em inglês
    """
    r, g, b = rgb
    
    # Cores básicas
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r < 50 and g < 50 and b < 50:
        return "black"
    if abs(r - g) < 30 and abs(g - b) < 30:
        if r > 150:
            return "light gray"
        return "gray"
    
    # Tons quentes
    if r > g and r > b:
        if g > 150:
            return "yellow"
        if g > 100:
            return "orange"
        if b > 100:
            return "pink"
        return "red"
    
    # Tons frios
    if g > r and g > b:
        if r > 100:
            return "lime"
        return "green"
    
    if b > r and b > g:
        if r > 100:
            return "purple"
        return "blue"
    
    return "colored"
