import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

from core.domain.scene_palette import ScenePalette, CharacterColorProfile
from core.database.chapter_db import ChapterDatabase
from core.logging.setup import get_logger

logger = get_logger("ScenePaletteService")

class ScenePaletteService:
    def __init__(self, chapter_db: ChapterDatabase, output_dir: Path):
        self.db = chapter_db
        self.output_dir = Path(output_dir)
        self.palette_file = self.output_dir / "scene_palette.json"
        
        # In-memory cache
        self.scene_palette: Optional[ScenePalette] = None
        self.profiles: Dict[str, CharacterColorProfile] = {}
        
        self.load_cache()
    
    def load_cache(self):
        """Carrega cache do JSON local."""
        if self.palette_file.exists():
            try:
                with open(self.palette_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Load ScenePalette
                sp_data = data.get("scene_palette")
                if sp_data:
                    self.scene_palette = ScenePalette(**sp_data)
                
                # Load Profiles
                profiles_data = data.get("profiles", {})
                for char_id, p_data in profiles_data.items():
                    self.profiles[char_id] = CharacterColorProfile(**p_data)
                    
                logger.info(f"Loaded ScenePalette and {len(self.profiles)} profiles.")
            except Exception as e:
                logger.error(f"Failed to load scene_palette.json: {e}")
    
    def save_cache(self):
        """Salva cache no JSON local."""
        data = {
            "scene_palette": self.scene_palette.__dict__ if self.scene_palette else None,
            "profiles": {k: v.__dict__ for k, v in self.profiles.items()}
        }
        try:
            with open(self.palette_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save scene_palette.json: {e}")

    def initialize_scene_palette(self, protagonists_refs: List[Image.Image]):
        """Inicializa a paleta da cena baseada nas referências disponíveis."""
        if not self.scene_palette:
            self.scene_palette = ScenePalette.from_protagonists(protagonists_refs)
            self.save_cache()
            logger.info(f"Initialized ScenePalette: {self.scene_palette.temperature}")

    def get_profile(self, char_id: str) -> CharacterColorProfile:
        """Retorna ou gera profile determinístico para o personagem."""
        if char_id in self.profiles:
            return self.profiles[char_id]
        
        # Garante que temos uma ScenePalette (fallback se vazia)
        if not self.scene_palette:
            self.scene_palette = ScenePalette(
                primary_hues=[], 
                base_saturation=0.5, 
                base_lightness=0.5, 
                temperature="neutral"
            )
            
        # Gera novo profile
        profile = CharacterColorProfile.generate_from_seed(
            char_id=char_id,
            scene_palette=self.scene_palette,
            archetype="civilian" # TODO: Detect archetype if possible
        )
        
        self.profiles[char_id] = profile
        self.save_cache()
        
        return profile
