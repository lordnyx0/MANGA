"""
Unit tests for Custom Exceptions
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.exceptions import (
    MangaColorError,
    AnalysisError,
    GenerationError,
    ModelLoadError,
    ResourceError
)

class TestExceptions:
    
    def test_manga_color_error_base(self):
        """Test base exception"""
        err = MangaColorError("Base error")
        assert str(err) == "Base error"
        assert isinstance(err, Exception)

    def test_analysis_error(self):
        """Test AnalysisError with page info"""
        err = AnalysisError("Failed to analyze", page_num=5)
        assert str(err) == "Failed to analyze"
        assert err.page_num == 5
        assert isinstance(err, MangaColorError)

    def test_generation_error(self):
        """Test GenerationError with page info"""
        err = GenerationError("Failed to generate", page_num=10)
        assert str(err) == "Failed to generate"
        assert err.page_num == 10
        assert isinstance(err, MangaColorError)

    def test_model_load_error(self):
        """Test ModelLoadError with model name"""
        err = ModelLoadError("Model not found", model_name="yolov8")
        assert str(err) == "Model not found"
        assert err.model_name == "yolov8"
        assert isinstance(err, MangaColorError)

    def test_resource_error(self):
        """Test ResourceError"""
        err = ResourceError("OOM")
        assert str(err) == "OOM"
        assert isinstance(err, MangaColorError)
