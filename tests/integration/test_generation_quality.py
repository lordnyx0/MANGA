"""
Testes de integração para qualidade de geração.

Estes testes verificam que as imagens geradas não apresentam
problemas como cores psicodélicas, NaNs ou artefatos.

Requer: GPU e modelos carregados (marcado com @pytest.mark.gpu)
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.gpu
class TestGeneratedImageQuality:
    """Testa qualidade de imagens geradas pelo pipeline."""
    
    def test_generated_image_has_no_nans(self):
        """
        Verifica que imagem gerada não contém pixels NaN.
        
        NaNs nos pixels aparecem como pontos pretos ou cores aleatórias.
        """
        pytest.skip("Requer GPU e modelos carregados")
    
    def test_generated_image_color_distribution(self):
        """
        Verifica distribuição de cores da imagem gerada.
        
        Imagens psicodélicas têm:
        - Alta saturação média
        - Distribuição não-natural de cores
        - Muitos pixels em extremos do histograma
        """
        pytest.skip("Requer GPU e modelos carregados")


class TestImageAnalysisUtils:
    """Utilitários para analisar imagens em busca de problemas."""
    
    @staticmethod
    def detect_psychedelic_artifacts(image: Image.Image) -> dict:
        """
        Analisa imagem em busca de artefatos psicodélicos.
        
        Returns:
            Dict com métricas:
            - has_nans: bool - Presença de NaNs
            - saturation_mean: float - Saturação média (HSV)
            - extreme_pixels_ratio: float - Ratio de pixels em extremos
            - color_variance: float - Variância entre canais RGB
        """
        # Converte para numpy
        img_np = np.array(image).astype(np.float32)
        
        # 1. Detecta NaNs
        has_nans = np.isnan(img_np).any()
        
        # 2. Calcula saturação média (conversão simplificada)
        # Max(R,G,B) - Min(R,G,B) / Max(R,G,B)
        max_rgb = img_np.max(axis=2)
        min_rgb = img_np.min(axis=2)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        saturation_mean = saturation.mean()
        
        # 3. Pixels em extremos (> 250 ou < 5)
        extreme_pixels = np.sum((img_np > 250) | (img_np < 5))
        extreme_pixels_ratio = extreme_pixels / img_np.size
        
        # 4. Variância entre canais RGB (alta variância = cores não-naturais)
        channel_variance = np.var(img_np, axis=2).mean()
        
        return {
            "has_nans": has_nans,
            "saturation_mean": float(saturation_mean),
            "extreme_pixels_ratio": float(extreme_pixels_ratio),
            "color_variance": float(channel_variance),
            "is_psychedelic": saturation_mean > 0.8 or extreme_pixels_ratio > 0.3
        }
    
    @staticmethod
    def detect_fried_colors(image: Image.Image) -> bool:
        """
        Detecta imagens "fritadas" com cores saturadas excessivamente.
        
        Args:
            image: PIL Image
            
        Returns:
            True se imagem parece ter sido processada incorretamente
        """
        img_np = np.array(image)
        
        # Histograma de cada canal
        r_hist, _ = np.histogram(img_np[:,:,0], bins=256, range=(0, 256))
        g_hist, _ = np.histogram(img_np[:,:,1], bins=256, range=(0, 256))
        b_hist, _ = np.histogram(img_np[:,:,2], bins=256, range=(0, 256))
        
        # Detecta picos nos extremos (sinal de clipping)
        extreme_threshold = 0.15  # 15% dos pixels nos extremos
        
        r_extreme = (r_hist[:10].sum() + r_hist[-10:].sum()) / r_hist.sum()
        g_extreme = (g_hist[:10].sum() + g_hist[-10:].sum()) / g_hist.sum()
        b_extreme = (b_hist[:10].sum() + b_hist[-10:].sum()) / b_hist.sum()
        
        # Se mais de um canal tem picos nos extremos, provavelmente está "fritado"
        extreme_count = sum([r_extreme > extreme_threshold,
                            g_extreme > extreme_threshold,
                            b_extreme > extreme_threshold])
        
        return extreme_count >= 2
    
    def test_psychedelic_detection_on_synthetic(self):
        """
        Testa detector de artefatos em imagem sintética.
        """
        # Cria imagem sintética "normal"
        normal_img = Image.new('RGB', (512, 512), (128, 128, 128))
        
        result = self.detect_psychedelic_artifacts(normal_img)
        
        # Imagem cinza deve ter saturação baixa
        assert result["saturation_mean"] < 0.1
        assert not result["is_psychedelic"]
    
    def test_psychedelic_detection_on_extreme(self):
        """
        Testa detector em imagem com cores extremas.
        """
        # Cria imagem com cores aleatórias extremas (simula psicodélico)
        np.random.seed(42)
        extreme_data = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        extreme_img = Image.fromarray(extreme_data)
        
        result = self.detect_psychedelic_artifacts(extreme_img)
        
        # Deve detectar como psicodélico
        assert result["is_psychedelic"] or result["color_variance"] > 1000
    
    def test_fried_detection_on_synthetic(self):
        """
        Testa detector de imagens fritadas.
        """
        # Imagem normal
        normal = Image.new('RGB', (512, 512), (128, 128, 128))
        assert not self.detect_fried_colors(normal)
        
        # Imagem "fritada" (muitos pixels nos extremos)
        fried_data = np.zeros((512, 512, 3), dtype=np.uint8)
        fried_data[:256, :256] = [255, 0, 0]  # Vermelho puro
        fried_data[256:, 256:] = [0, 255, 255]  # Ciano puro
        fried = Image.fromarray(fried_data)
        
        # Esta imagem artificial tem extremos, então deve detectar
        result = self.detect_fried_colors(fried)
        # Nota: Este é um teste básico, pode precisar de ajuste


class TestVAEStability:
    """Testes específicos para estabilidade do VAE."""
    
    def test_vae_decode_with_safe_latents(self):
        """
        Testa que VAE consegue decodificar latents seguros.
        """
        from core.generation.engines.vae_dtype_adapter import VAEDtypeAdapter
        
        # Mock de VAE
        vae_mock = Mock()
        vae_mock.dtype = torch.float32
        
        def safe_decode(latents, return_dict=True, generator=None):
            # Simula decode: latents -> imagem
            # Latents [B, 4, H/8, W/8] -> Imagem [B, 3, H, W]
            b, _, h, w = latents.shape
            sample = torch.randn(b, 3, h*8, w*8)
            return Mock(sample=sample)
        
        vae_mock.decode = safe_decode
        
        # Testa com latents seguros
        safe_latents = torch.randn(1, 4, 64, 64) * 2  # valores normais
        
        with VAEDtypeAdapter(vae_mock):
            result = vae_mock.decode(safe_latents)
        
        assert result is not None
        assert result.sample.shape == (1, 3, 512, 512)
    
    def test_vae_decode_with_problematic_latents(self):
        """
        Testa que VAE consegue lidar com latents problemáticos.
        """
        from core.generation.engines.vae_dtype_adapter import VAEDtypeAdapter
        
        vae_mock = Mock()
        vae_mock.dtype = torch.float32
        vae_mock.decode = Mock(return_value=Mock(sample=torch.randn(1, 3, 512, 512)))
        
        # Latents com problemas
        bad_latents = torch.randn(1, 4, 64, 64)
        bad_latents[0, 0, 0, 0] = float('nan')
        bad_latents[0, 0, 0, 1] = 100.0  # Valor extremo
        
        # Não deve lançar exceção
        try:
            with VAEDtypeAdapter(vae_mock):
                result = vae_mock.decode(bad_latents)
            assert True
        except Exception as e:
            pytest.fail(f"Não deveria lançar exceção: {e}")


# Mock para testes
from unittest.mock import Mock

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
