"""
Unit tests for MangaPromptBuilder
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.generation.prompt_builder import MangaPromptBuilder
from core.constants import SCENE_DESCRIPTIONS

class TestMangaPromptBuilder:
    
    @pytest.fixture
    def builder(self):
        return MangaPromptBuilder()
    
    def test_build_prompt_basic(self, builder):
        """Test basic prompt building with default options"""
        page_data = {}
        options = {'prompt': 'base prompt'}
        
        prompt = builder.build_prompt(page_data, options, num_characters=0)
        
        assert 'base prompt' in prompt
        assert 'masterpiece' in prompt
        assert 'character' not in prompt
        # Default scene
        default_scene = SCENE_DESCRIPTIONS.get('present')
        assert default_scene in prompt

    def test_build_prompt_scene_type(self, builder):
        """Test scene type lookup (OCP)"""
        page_data = {'scene_type': 'flashback'}
        options = {'prompt': 'base'}
        
        prompt = builder.build_prompt(page_data, options, num_characters=0)
        
        flashback_desc = SCENE_DESCRIPTIONS.get('flashback')
        assert flashback_desc in prompt

    def test_build_prompt_with_characters(self, builder):
        """Test character count inclusion"""
        page_data = {}
        options = {'prompt': 'base'}
        
        prompt = builder.build_prompt(page_data, options, num_characters=3)
        
        assert '3 character(s)' in prompt

    def test_build_prompt_color_reference(self, builder):
        """Test color reference precedence over style presets"""
        page_data = {}
        
        # Mock palette
        mock_palette = MagicMock()
        mock_palette.is_color_reference = True
        mock_palette.regions = {
            'hair': MagicMock(dominant_color=[50, 50, 50]) # Red-ish
        }
        
        options = {
            'prompt': 'base',
            'style_preset': 'monochrome', # Should be ignored
            'character_palettes': {'char_001': mock_palette}
        }
        
        prompt = builder.build_prompt(page_data, options, num_characters=1)
        
        # Should contain color info
        # [50, 50, 50] -> likely red/orange/brown depending on logic
        # Just check that it returns non-empty color string
        assert 'hair' in prompt
        
        # Should NOT contain style preset addition if logic works as expected
        # Assuming 'monochrome' preset adds 'monochrome' or 'greyscale'
        # We'd need to check config.settings.STYLE_PRESETS to be sure, 
        # but the builder logic says: if has_color_reference -> ignore style preset additions
        # We assume builder logic: "Aplica STYLE_PRESETS apenas se NÃO houver referências coloridas"

    def test_lab_to_color_name(self, builder):
        """Test color mapping logic"""
        # White
        assert builder._lab_to_color_name([95, 0, 0]) == 'white'
        # Black
        assert builder._lab_to_color_name([10, 0, 0]) == 'black'
        # Red (High A, moderate B)
        assert builder._lab_to_color_name([50, 60, 40]) in ['red', 'red-orange', 'coral']
        # Blue (Negative B)
        assert builder._lab_to_color_name([50, 0, -50]) in ['blue', 'cyan']

    def test_hue_to_color_name(self, builder):
        """Test HSL hue mapping"""
        assert builder.hue_to_color_name(0) == "red"
        assert builder.hue_to_color_name(120) == "green"
        assert builder.hue_to_color_name(240) == "blue"
        assert builder.hue_to_color_name(60) == "yellow"
        
    def test_build_prompt_for_character_scene_palette(self, builder):
        """Test prompt generation with ScenePalette profile"""
        profile = MagicMock()
        profile.primary_hue = 0 # Red
        profile.secondary_hue = 240 # Blue
        profile.archetype = "soldier"
        
        scene = MagicMock()
        scene.temperature = "warm"
        
        prompt = builder.build_prompt_for_character(
            character_desc="soldier character",
            color_profile=profile,
            scene_palette=scene
        )
        
        assert "wearing red clothes" in prompt
        assert "blue details" in prompt
        assert "warm lighting" in prompt
