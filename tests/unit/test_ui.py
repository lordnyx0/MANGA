"""
Unit tests for UI module (ui/interface.py)
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock gradio before importing ui
gr_mock = MagicMock()
gr_mock.Progress = MagicMock()
gr_mock.themes.Soft = MagicMock(return_value=MagicMock())
sys.modules['gradio'] = gr_mock

from ui.interface import MangaColorizerUI, create_ui, launch_ui
from core.pipeline import ChapterAnalysis, GenerationOptions


class TestMangaColorizerUI:
    """Tests for MangaColorizerUI class"""
    
    @pytest.fixture
    def ui(self):
        """Create a fresh UI instance"""
        with patch('ui.interface.MangaColorizationPipeline'):
            return MangaColorizerUI()
    
    def test_initialization(self, ui):
        """Test UI initialization"""
        assert ui.pipeline is None
        assert ui.analysis is None
        assert ui.page_paths == []
        assert ui._page_info == {}
        assert ui.output_dir is not None
    
    def test_init_pipeline_lazy_loading(self, ui):
        """Test that pipeline is initialized lazily"""
        with patch('ui.interface.MangaColorizationPipeline') as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance
            
            # First call should create pipeline
            pipeline = ui._init_pipeline()
            assert pipeline is mock_instance
            MockPipeline.assert_called_once()
            
            # Second call should return same instance
            pipeline2 = ui._init_pipeline()
            assert pipeline2 is pipeline
            assert MockPipeline.call_count == 1
    
    def test_analyze_chapter_no_files(self, ui):
        """Test analyze_chapter with no files"""
        gr_mock.update = MagicMock(return_value="update")
        
        status, resumo, detalhes, gallery_update = ui.analyze_chapter([])
        
        assert "❌ Nenhuma imagem" in status
        assert resumo == ""
        assert detalhes == ""
    
    def test_analyze_chapter_success(self, ui):
        """Test successful chapter analysis"""
        # Setup mock files
        mock_file = MagicMock()
        mock_file.name = "/path/to/page_001.png"
        
        # Setup mock pipeline
        mock_analysis = MagicMock(spec=ChapterAnalysis)
        mock_analysis.num_pages = 5
        mock_analysis.num_characters = 3
        mock_analysis.characters = [
            {'appearances': 5, 'embedding_method': 'hybrid'},
            {'appearances': 3, 'embedding_method': 'clip'},
        ]
        mock_analysis.scene_breakdown = {
            'present': [1, 2, 3],
            'flashback': [4, 5]
        }
        mock_analysis.estimated_generation_time = 45.5
        
        with patch.object(ui, '_init_pipeline') as mock_init:
            mock_pipeline = MagicMock()
            mock_pipeline.process_chapter.return_value = mock_analysis
            mock_init.return_value = mock_pipeline
            
            status, resumo, detalhes, gallery = ui.analyze_chapter([mock_file])
            
            assert "✅ Análise completa" in status
            assert "5 páginas" in status
            assert "3 personagens" in status
            assert "📊 Resumo da Análise" in resumo
            assert "⚙️ Detalhes Técnicos" in detalhes
            assert ui.analysis is mock_analysis
    
    def test_analyze_chapter_error(self, ui):
        """Test analyze_chapter with error"""
        mock_file = MagicMock()
        mock_file.name = "/path/to/page.png"
        
        with patch.object(ui, '_init_pipeline') as mock_init:
            mock_pipeline = MagicMock()
            mock_pipeline.process_chapter.side_effect = Exception("Test error")
            mock_init.return_value = mock_pipeline
            
            status, resumo, detalhes, gallery = ui.analyze_chapter([mock_file])
            
            assert "❌ Erro inesperado" in status
            assert "Test error" in status
    
    def test_generate_page_no_analysis(self, ui):
        """Test generate_page without prior analysis"""
        status, image_path = ui.generate_page(
            page_num=1,
            style_preset="default",
            quality_mode="balanced",
            ip_scale=0.6,
            preserve_text=True,
            apply_narrative=True,
            seed=-1
        )
        
        assert "❌ Execute a análise primeiro" in status
        assert image_path is None
    
    def test_generate_page_invalid_page(self, ui):
        """Test generate_page with invalid page number"""
        ui.analysis = MagicMock(spec=ChapterAnalysis)
        ui.analysis.num_pages = 5
        
        status, image_path = ui.generate_page(
            page_num=10,  # Invalid
            style_preset="default",
            quality_mode="balanced",
            ip_scale=0.6,
            preserve_text=True,
            apply_narrative=True,
            seed=-1
        )
        
        assert "❌ Página inválida" in status
    
    def test_generate_page_success(self, ui):
        """Test successful page generation"""
        ui.analysis = MagicMock(spec=ChapterAnalysis)
        ui.analysis.num_pages = 5
        
        mock_result = MagicMock()
        mock_result.save = MagicMock()
        
        with patch.object(ui, '_init_pipeline') as mock_init:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_page.return_value = mock_result
            mock_init.return_value = mock_pipeline
            
            status, output_path = ui.generate_page(
                page_num=1,
                style_preset="vibrant",
                quality_mode="high",
                ip_scale=0.8,
                preserve_text=False,
                apply_narrative=True,
                seed=42
            )
            
            assert "✅ Página 1 gerada" in status
            assert output_path is not None
            mock_pipeline.generate_page.assert_called_once()
            
            # Check GenerationOptions was created correctly
            call_args = mock_pipeline.generate_page.call_args
            assert call_args[0][0] == 0  # page_num - 1 (0-based internally)
    
    def test_generate_all_pages_no_analysis(self, ui):
        """Test generate_all_pages without prior analysis"""
        status, paths = ui.generate_all_pages(
            style_preset="default",
            quality_mode="balanced",
            ip_scale=0.6,
            preserve_text=True,
            apply_narrative=True,
            seed=-1
        )
        
        assert "❌ Execute a análise primeiro" in status
        assert paths == []
    
    def test_generate_all_pages_success(self, ui):
        """Test successful generation of all pages"""
        ui.analysis = MagicMock(spec=ChapterAnalysis)
        ui.analysis.num_pages = 3
        
        mock_result = MagicMock()
        mock_result.save = MagicMock()
        
        with patch.object(ui, '_init_pipeline') as mock_init:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_page.return_value = mock_result
            mock_init.return_value = mock_pipeline
            
            status, paths = ui.generate_all_pages(
                style_preset="default",
                quality_mode="fast",
                ip_scale=0.5,
                preserve_text=True,
                apply_narrative=False,
                seed=123
            )
            
            assert "✅ 3 páginas geradas" in status
            assert len(paths) == 3
            assert mock_pipeline.generate_page.call_count == 3
    
    def test_set_scene_context_no_analysis(self, ui):
        """Test set_scene_context without prior analysis"""
        status = ui.set_scene_context(1, 5, "flashback")
        assert "❌ Execute a análise primeiro" in status
    
    def test_set_scene_context_invalid_range(self, ui):
        """Test set_scene_context with invalid page range"""
        ui.analysis = MagicMock(spec=ChapterAnalysis)
        ui.analysis.num_pages = 5
        
        status = ui.set_scene_context(10, 15, "flashback")
        assert "❌ Range inválido" in status
    
    def test_set_scene_context_start_greater_than_end(self, ui):
        """Test set_scene_context with start > end"""
        ui.analysis = MagicMock(spec=ChapterAnalysis)
        ui.analysis.num_pages = 10
        
        status = ui.set_scene_context(5, 3, "flashback")
        assert "❌ Página inicial deve ser menor" in status
    
    def test_set_scene_context_success(self, ui):
        """Test successful scene context setting"""
        ui.analysis = MagicMock(spec=ChapterAnalysis)
        ui.analysis.num_pages = 10
        
        with patch.object(ui, '_init_pipeline') as mock_init:
            mock_pipeline = MagicMock()
            mock_init.return_value = mock_pipeline
            
            status = ui.set_scene_context(1, 5, "flashback")
            
            assert "✅ Contexto 'flashback'" in status
            assert "cores desaturadas" in status
            mock_pipeline.set_scene_context.assert_called_once_with(
                page_range=(0, 4),  # 0-based
                context_type="flashback"
            )
    
    def test_format_analysis_summary(self, ui):
        """Test _format_analysis_summary"""
        mock_analysis = MagicMock(spec=ChapterAnalysis)
        mock_analysis.num_pages = 10
        mock_analysis.num_characters = 3
        mock_analysis.estimated_generation_time = 120.5
        mock_analysis.characters = [
            {'appearances': 10, 'embedding_method': 'hybrid'},
            {'appearances': 5, 'embedding_method': 'clip'},
            {'appearances': 3, 'embedding_method': 'clip'},
        ]
        mock_analysis.scene_breakdown = {
            'present': [1, 2, 3, 4, 5],
            'flashback': [6, 7, 8, 9, 10]
        }
        
        summary = ui._format_analysis_summary(mock_analysis)
        
        # Check key content is present (using unicode escapes to avoid encoding issues)
        assert "Resumo" in summary
        assert "10" in summary  # num_pages
        assert "3" in summary   # num_characters
        assert "120" in summary  # estimated time
        assert "Personagens" in summary
        assert "120s" in summary or "121s" in summary
        assert "👤 Personagens Principais" in summary
    
    def test_format_technical_details(self, ui):
        """Test _format_technical_details"""
        mock_analysis = MagicMock(spec=ChapterAnalysis)
        
        details = ui._format_technical_details(mock_analysis)
        
        assert "⚙️ Detalhes Técnicos" in details
        assert "Hardware" in details
        assert "Tile-Aware" in details
        assert "IP-Adapter" in details
        assert "Cache" in details


class TestCreateUI:
    """Tests for create_ui function"""
    
    def test_create_ui_returns_blocks(self):
        """Test that create_ui returns Gradio Blocks"""
        with patch('ui.interface.MangaColorizerUI'):
            with patch('ui.interface.gr') as mock_gr:
                mock_blocks = MagicMock()
                mock_gr.Blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks)
                mock_gr.Blocks.return_value.__exit__ = MagicMock(return_value=False)
                
                app = create_ui()
                
                assert app is not None
                mock_gr.Blocks.assert_called_once()


class TestLaunchUI:
    """Tests for launch_ui function"""
    
    def test_launch_ui(self):
        """Test launch_ui function"""
        with patch('ui.interface.create_ui') as mock_create:
            mock_app = MagicMock()
            mock_create.return_value = mock_app
            
            launch_ui(share=True, server_port=8080)
            
            mock_create.assert_called_once()
            mock_app.launch.assert_called_once_with(
                share=True,
                server_port=8080,
                show_error=True,
                quiet=False
            )


# Tests for GenerationOptions dataclass validation
class TestGenerationOptions:
    """Tests for GenerationOptions usage in UI"""
    
    def test_generation_options_creation(self):
        """Test GenerationOptions creation with various parameters"""
        # Default options
        opts1 = GenerationOptions()
        assert opts1.style_preset == "default"
        assert opts1.quality_mode == "balanced"
        assert opts1.seed is None
        
        # Custom options
        opts2 = GenerationOptions(
            style_preset="vibrant",
            quality_mode="high",
            preserve_original_text=False,
            apply_narrative_transforms=True,
            seed=42
        )
        assert opts2.style_preset == "vibrant"
        assert opts2.quality_mode == "high"
        assert opts2.preserve_original_text is False
        assert opts2.apply_narrative_transforms is True
        assert opts2.seed == 42
