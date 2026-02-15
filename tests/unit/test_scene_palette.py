
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from core.domain.scene_palette import CharacterColorProfile, ScenePalette
from core.generation.scene_palette_service import ScenePaletteService

def test_profile_determinism():
    # Scene agnostic
    p1 = CharacterColorProfile.generate_from_seed("char_123", None)
    p2 = CharacterColorProfile.generate_from_seed("char_123", None)
    assert p1.primary_hue == p2.primary_hue
    assert p1.secondary_hue == p2.secondary_hue
    
    p3 = CharacterColorProfile.generate_from_seed("char_456", None)
    # Probabilistic check (collisions possible but rare for simple hash)
    # With 360 hues, collision is 1/360.
    if p1.primary_hue == p3.primary_hue:
        assert p1.secondary_hue != p3.secondary_hue or p1.lightness != p3.lightness

def test_palette_harmonization():
    # Warm scene
    scene = ScenePalette(
        primary_hues=[0, 10], 
        base_saturation=0.8, 
        base_lightness=0.6, 
        temperature="warm"
    )
    
    profile = CharacterColorProfile.generate_from_seed("char_cold", scene)
    assert profile.saturation >= 0.2  # Should be clamped/influenced
    # Check if warmth bias works (hard to test without exact implementation details, but we can check it runs)

def test_service_initialization(tmp_path):
    mock_db = MagicMock()
    service = ScenePaletteService(mock_db, tmp_path)
    assert service.palette_file == tmp_path / "scene_palette.json"
    assert service.profiles == {}

def test_service_roundtrip(tmp_path):
    mock_db = MagicMock()
    service = ScenePaletteService(mock_db, tmp_path)
    
    # Generate profile
    profile = service.get_profile("char_A")
    assert profile.char_id == "char_A"
    
    # Save (implied by get_profile if it saves? No, usually save is manual in tests or on exit)
    service.save_cache()
    
    # Reload
    service2 = ScenePaletteService(mock_db, tmp_path)
    # It should load automatically in init
    
    # Mock db to ensure it doesn't try to fetch images if we call get_profile
    profile2 = service2.get_profile("char_A")
    
    assert profile.primary_hue == profile2.primary_hue
