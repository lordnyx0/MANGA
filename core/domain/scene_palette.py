from dataclasses import dataclass
from typing import List, Tuple, Optional
import hashlib
from PIL import Image

def get_dominant_hsl(image: Image.Image) -> Tuple[int, float, float]:
    """
    Extrai HSL dominante de uma imagem PIL.
    """
    # Resize para 1x1 para pegar a média
    small = image.resize((1, 1), Image.BICUBIC)
    if small.mode != "RGB":
        small = small.convert("RGB")
        
    color = small.getpixel((0, 0)) # RGB tuple
    
    # RGB to HSL manual conversion
    r, g, b = color[0]/255.0, color[1]/255.0, color[2]/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    
    h = 0
    s = 0
    l = (mx + mn) / 2
    
    if df != 0:
        s = df / (2 - mx - mn) if l < 0.5 else df / (mx + mn)
        if mx == r:
            h = (g - b) / df + (6 if g < b else 0)
        elif mx == g:
            h = (b - r) / df + 2
        else:
            h = (r - g) / df + 4
        h /= 6
        
    return (int(h * 360), s, l)

@dataclass
class ScenePalette:
    """Paleta dominante extraída dos protagonistas da cena."""
    primary_hues: List[int]      # Hues dos protagonistas
    base_saturation: float       # Média de saturação da cena
    base_lightness: float        # Média de luminosidade
    temperature: str             # "warm", "cool", "neutral"
    
    @classmethod
    def from_protagonists(cls, protagonist_refs: List[Image.Image]):
        """
        Extrai estatísticas cromáticas das referências dos protagonistas.
        """
        hues = []
        sats = []
        lights = []
        
        for img in protagonist_refs:
            if img:
                h, s, l = get_dominant_hsl(img)
                hues.append(h)
                sats.append(s)
                lights.append(l)
        
        # Temperatura baseada na média de hue
        avg_hue = sum(hues) / len(hues) if hues else 0
        # Warm: Reds (0-60), Yellows (45-75), Magentas (300-360) roughly
        # Cool: Blues, cyans, greens
        is_warm = (0 <= avg_hue <= 60) or (300 <= avg_hue <= 360) or (45 <= avg_hue <= 75)
        temp = "warm" if is_warm else "cool"
        
        # Defaults if empty
        base_sat = sum(sats) / len(sats) if sats else 0.5
        base_light = sum(lights) / len(lights) if lights else 0.5
        
        return cls(
            primary_hues=hues,
            base_saturation=base_sat,
            base_lightness=base_light,
            temperature=temp
        )

@dataclass
class CharacterColorProfile:
    """Perfil cromático determinístico para personagem sem referência."""
    char_id: str
    primary_hue: int      # 0-360 (HSL)
    secondary_hue: int    # 0-360
    saturation: float     # 0.0-1.0 (harmonizado com cena)
    lightness: float      # 0.0-1.0
    archetype: str        # "soldier", "civilian", "student", etc.
    
    @classmethod
    def generate_from_seed(cls, char_id: str, scene_palette: 'ScenePalette', archetype: str = "civilian"):
        """
        Gera cores determinísticas baseadas no hash do char_id,
        mas harmonizadas com a temperatura da cena.
        """
        # Hash determinístico
        hash_val = int(hashlib.md5(char_id.encode()).hexdigest(), 16)
        
        # Hue primário: distribuído uniformemente via hash
        primary_hue = hash_val % 360
        
        # Hue secundário: análogo/complementar baseado no primário
        # 30% chance de ser complementar (+180), 70% ser análogo (+30)
        # Vamos usar a temperatura da cena para influenciar?
        # Se cena é warm, favorecer hues quentes?
        # Por enquanto, determinístico puro + harmonização de SAT/LIG.
        
        offset_choice = (hash_val // 360) % 10
        offset = 180 if offset_choice > 6 else 30
        secondary_hue = (primary_hue + offset) % 360
        
        # Saturação e Lightness: harmonizados com a cena (não aleatórios)
        # Variação determinística em torno da base da cena
        sat_variation = ((hash_val % 30) - 15) / 100.0 # -0.15 a +0.15
        light_variation = ((hash_val % 20) - 10) / 100.0 # -0.10 a +0.10
        
        if scene_palette:
            sat = scene_palette.base_saturation + sat_variation
            light = scene_palette.base_lightness + light_variation
        else:
            sat = 0.5 + sat_variation 
            light = 0.5 + light_variation
        
        return cls(
            char_id=char_id,
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            saturation=min(1.0, max(0.2, sat)), # Clamp safe range
            lightness=min(0.9, max(0.2, light)), # Clamp safe range
            archetype=archetype
        )
