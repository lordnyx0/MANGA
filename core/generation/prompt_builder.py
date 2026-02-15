from typing import Dict, Optional, Any, Tuple
from config.settings import STYLE_PRESETS
from core.constants import SCENE_DESCRIPTIONS
# from core.domain.scene_palette import CharacterColorProfile, ScenePalette # Avoid circular imports if possible, used only for typing


class MangaPromptBuilder:
    """
    Constrói prompts para geração de imagens de mangá coloridas.
    Extraído de TileAwareGenerator para reduzir complexidade e acoplamento.
    """
    
    def __init__(self, logger=None):
        self.logger = logger

    def build_prompt(
        self,
        page_data: Dict,
        options: Any,
        num_characters: int
    ) -> str:
        """
        Constrói prompt para geração.
        
        PRIORIDADE DE ESTILO:
        1. Se houver paletas de referência colorida -> usa as cores das referências (ignora STYLE_PRESETS)
        2. Se não houver referências -> aplica STYLE_PRESETS
        
        Args:
            page_data: Dados da página
            options: Opções de geração (dict ou objeto)
            num_characters: Número de personagens no tile
            
        Returns:
            Prompt final
        """
        # Prompt base - suporta dict ou dataclass
        if isinstance(options, dict):
            base = options.get('prompt', 'manga page, anime style, detailed coloring')
            negative_prompt = options.get('negative_prompt', '')
            style_preset = options.get('style_preset', 'default')
        else:
            # GenerationOptions dataclass
            base = getattr(options, 'prompt', 'manga page, anime style, detailed coloring')
            negative_prompt = getattr(options, 'negative_prompt', '')
            style_preset = getattr(options, 'style_preset', 'default')
        
        # Verifica se há paletas de referência colorida
        palettes = options.get('character_palettes', {}) if isinstance(options, dict) else {}
        has_color_reference = False
        if palettes:
            # Verifica se alguma paleta vem de imagem de referência colorida
            has_color_reference = any(
                getattr(p, 'is_color_reference', False) or 
                getattr(p, 'source_page', 0) == -1
                for p in palettes.values()
            )
        
        # Aplica STYLE_PRESETS apenas se NÃO houver referências coloridas
        style_config = STYLE_PRESETS.get(style_preset, STYLE_PRESETS['default'])
        style_applied = False
        
        if not has_color_reference and style_config.get('prompt_addition'):
            base += ", " + style_config['prompt_addition']
            style_applied = True
            if self.logger:
                print(f"[MangaPromptBuilder] Aplicando style_preset '{style_preset}' (sem referências)")
        elif has_color_reference:
            if self.logger:
                print(f"[MangaPromptBuilder] Usando paletas de referência colorida (ignorando style_preset)")
        
        # Contexto de cena
        scene_type = page_data.get('scene_type', 'present')
        scene_desc = SCENE_DESCRIPTIONS.get(scene_type, SCENE_DESCRIPTIONS.get('present', ''))
        
        # Adiciona descrição de personagens
        char_prompt = ""
        if num_characters > 0:
            char_prompt = f"{num_characters} character(s), "
        
        # Paletas de cores: enriquece prompt com cores dos personagens
        color_prompt = ""
        if palettes:
            # APENAS usa paletas de REFERÊNCIA COLORIDA
            # Paletas extraídas de mangá B&W não são confiáveis para prompts
            if has_color_reference:
                ref_palettes = {k: v for k, v in palettes.items() 
                               if getattr(v, 'is_color_reference', False) or getattr(v, 'source_page', 0) == -1}
                if ref_palettes:
                    color_prompt = self._build_color_prompt(ref_palettes)
                    if self.logger:
                        print(f"[MangaPromptBuilder] Usando {len(ref_palettes)} paletas de referência colorida")
                else:
                    if self.logger:
                        print(f"[MangaPromptBuilder] Nenhuma paleta de referência válida encontrada")
            else:
                # NÃO usa paletas B&W - deixa o modelo/STYLE_PRESETS decidir
                if self.logger:
                    print(f"[MangaPromptBuilder] Ignorando paletas B&W (sem referências coloridas)")
        
        # Monta prompt final
        if color_prompt:
            prompt = f"{base}, {char_prompt}{color_prompt}, {scene_desc}, masterpiece, best quality"
        else:
            prompt = f"{base}, {char_prompt}{scene_desc}, masterpiece, best quality"
        
        # Registra prompts no logger se disponível
        if self.logger and hasattr(self.logger, 'log_prompts'):
            config_for_log = {
                "scene_type": scene_type,
                "style_preset": style_preset,
                "num_characters": num_characters,
                "has_color_reference": has_color_reference,
                "style_config_applied": style_config if style_applied else None,
                "color_prompt_used": bool(color_prompt)
            }
            self.logger.log_prompts(prompt, negative_prompt, config_for_log)
        
        return prompt

    def _build_color_prompt(self, palettes: Dict) -> str:
        """
        Constrói descrição de cores baseada nas paletas dos personagens.
        
        Args:
            palettes: Dict de char_id -> CharacterPalette
            
        Returns:
            String descrevendo as cores dos personagens
        """
        if not palettes:
            return ""
        
        color_terms = []
        
        for char_id, palette in palettes.items():
            if not hasattr(palette, 'regions'):
                continue
            
            char_colors = []
            
            # Extrai cor do cabelo
            if 'hair' in palette.regions:
                hair_lab = palette.regions['hair'].dominant_color
                hair_name = self._lab_to_color_name(hair_lab)
                if hair_name:
                    char_colors.append(f"{hair_name} hair")
                    if self.logger:
                        print(f"[ColorPrompt] {char_id} hair: LAB{hair_lab} -> {hair_name}")
            
            # Extrai cor da roupa
            if 'clothes_primary' in palette.regions:
                clothes_lab = palette.regions['clothes_primary'].dominant_color
                clothes_name = self._lab_to_color_name(clothes_lab)
                if clothes_name:
                    char_colors.append(f"{clothes_name} clothes")
                    if self.logger:
                        print(f"[ColorPrompt] {char_id} clothes: LAB{clothes_lab} -> {clothes_name}")
            
            # Extrai cor dos olhos
            if 'eyes' in palette.regions:
                eyes_lab = palette.regions['eyes'].dominant_color
                eyes_name = self._lab_to_color_name(eyes_lab)
                if eyes_name:
                    char_colors.append(f"{eyes_name} eyes")
                    if self.logger:
                        print(f"[ColorPrompt] {char_id} eyes: LAB{eyes_lab} -> {eyes_name}")
            
            if char_colors:
                color_terms.append(", ".join(char_colors))
        
        if not color_terms:
            return ""
        
        # Junta todas as descrições
        result = "; ".join(color_terms)
        if self.logger:
            print(f"[ColorPrompt] Final: {result}")
        return result
    
    def _lab_to_color_name(self, lab_color) -> str:
        """
        Converte cor CIELAB para nome aproximado em inglês.
        
        Versão melhorada com thresholds mais precisos para evitar
        excesso de classificações "orange" em tons de pele/cabelo.
        
        Args:
            lab_color: Tupla (L, a, b) ou lista
            
        Returns:
            Nome da cor em inglês
        """
        if lab_color is None or len(lab_color) < 3:
            return ""
        
        L, a, b = lab_color
        
        # Preto/Branco/Cinza (baseado em L e cromaticidade)
        chroma = (a**2 + b**2) ** 0.5
        
        if L > 90 and chroma < 15:
            return "white"
        if L < 20 and chroma < 15:
            return "black"
        if chroma < 10:
            return "gray"
        
        # Cores por quadrante - thresholds ajustados
        
        # Vermelho/Rosa (a alto positivo)
        if a > 15:
            if b > 25:
                # Laranja verdadeiro: alto a e alto b
                return "orange" if b > 40 else "coral"
            elif b > 5:
                # Vermelho a laranja-avermelhado
                return "red" if b < 20 else "red-orange"
            elif b < -15:
                return "magenta"
            else:
                # Rosa/Rosa-claro
                return "pink" if L > 60 else "rose"
        
        # Verde (a alto negativo)
        if a < -15:
            if b > 15:
                return "lime" if L > 70 else "yellow-green"
            elif b > -15:
                return "green"
            else:
                return "teal" if L > 50 else "dark-green"
        
        # Amarelo/Âmbar (b alto positivo, a próximo de zero)
        if b > 30:
            if abs(a) < 10:
                # Amarelo puro
                return "yellow" if L > 60 else "gold"
            elif a > 5:
                # Laranja-amarelado
                return "amber"
            else:
                return "yellow"
        
        # Laranja (b moderado-alto, a positivo moderado)
        if b > 20 and a > 5:
            if b > 35 and a > 15:
                return "orange"
            elif L > 60:
                return "peach"
            else:
                return "brown"
        
        # Azul (b negativo)
        if b < -20:
            if a < -5:
                return "cyan" if L > 60 else "blue"
            elif a > 10:
                return "purple"
            else:
                return "blue"
        
        # Roxo/Violeta (a positivo, b negativo moderado)
        if a > 10 and b < -5:
            return "purple" if L > 40 else "violet"
        
        # Fallback - tons de pele/comuns em mangá
        if L > 70 and a > 5 and b > 10:
            return "peach"  # Tom de pele claro
        elif L > 50 and a > 5 and b > 10:
            return "tan"    # Tom de pele médio
        elif L > 60:
            return "light"
        else:
            return "brown" if a > 0 else "gray"

    def build_prompt_for_character(self, character_desc: str, color_profile=None, scene_palette=None) -> str:
        """
        Constrói prompt com cores específicas para um personagem (ScenePalette).
        
        Args:
            character_desc: Descrição base do personagem (ex: "girl with long hair")
            color_profile: CharacterColorProfile optional
            scene_palette: ScenePalette optional
        """
        if not color_profile:
            return f"masterpiece, best quality, {character_desc}"
            
        # Converter HSL para nomes de cores aproximados
        primary_color = self.hue_to_color_name(color_profile.primary_hue)
        secondary_color = self.hue_to_color_name(color_profile.secondary_hue)
        
        prompt_parts = [
            "masterpiece", "best quality",
            character_desc,
            f"wearing {primary_color} clothes",
            f"{secondary_color} details"
        ]
        
        if scene_palette:
            prompt_parts.append(f"{scene_palette.temperature} lighting")
            
        prompt_parts.append("anime style")
        prompt_parts.append("flat colors")
        
        return ", ".join(prompt_parts)

    def hue_to_color_name(self, hue: int) -> str:
        """Mapeia hue (0-360) para nome de cor aproximado."""
        hue = hue % 360
        if 345 <= hue or hue < 15:
            return "red"
        elif 15 <= hue < 45:
            return "orange"
        elif 45 <= hue < 75:
            return "yellow"
        elif 75 <= hue < 105:
            return "lime"
        elif 105 <= hue < 135:
            return "green"
        elif 135 <= hue < 165:
            return "teal"
        elif 165 <= hue < 195:
            return "cyan"
        elif 195 <= hue < 225:
            return "sky blue"
        elif 225 <= hue < 255:
            return "blue"
        elif 255 <= hue < 285:
            return "purple"
        elif 285 <= hue < 315:
            return "magenta"
        elif 315 <= hue < 345:
            return "pink"
        return "red"

