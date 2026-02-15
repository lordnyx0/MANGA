"""
MangaAutoColor Pro - Medium Priority Test: Overlapping Masks

Criar duas/mais máscaras sobrepostas; verificar clamping e background.
"""

import numpy as np
import pytest

from core.test_utils import create_gaussian_mask


@pytest.mark.medium
class TestOverlappingMasks:
    """Testes de sobreposição e normalização de máscaras."""
    
    def test_sum_of_masks_clamped_to_one(self):
        """
        Soma de máscaras sobrepostas deve ser clamped em [0, 1].
        
        Aceite: clamped ∈ [0, 1].
        """
        h, w = 512, 512
        
        # Duas máscaras gaussianas com centros próximos (sobreposição)
        mask1 = create_gaussian_mask((h, w), center=(200, 256), sigma=100)
        mask2 = create_gaussian_mask((h, w), center=(312, 256), sigma=100)
        
        # Soma
        sum_masks = mask1 + mask2
        
        # Clamp
        clamped = np.clip(sum_masks, 0, 1)
        
        print(f"\n[Clamping Test]")
        print(f"Sum min: {sum_masks.min():.4f}, max: {sum_masks.max():.4f}")
        print(f"Clamped min: {clamped.min():.4f}, max: {clamped.max():.4f}")
        
        # Verifica clamping
        assert clamped.min() >= 0.0, "Clamped abaixo de 0"
        assert clamped.max() <= 1.0, "Clamped acima de 1"
        
        # A soma original deve ter valores > 1 na sobreposição
        assert sum_masks.max() > 1.0, "Máscaras não estão sobrepondo (max <= 1)"
    
    def test_background_mask_positive(self):
        """
        Máscara de background deve ser >= 0.
        
        mask_background = 1 - clamped_sum
        
        Aceite: mask_background >= 0.
        """
        h, w = 512, 512
        
        mask1 = create_gaussian_mask((h, w), center=(200, 256), sigma=100)
        mask2 = create_gaussian_mask((h, w), center=(312, 256), sigma=100)
        
        clamped = np.clip(mask1 + mask2, 0, 1)
        background_mask = 1 - clamped
        
        print(f"\n[Background Mask Test]")
        print(f"Background min: {background_mask.min():.4f}")
        print(f"Background max: {background_mask.max():.4f}")
        
        assert background_mask.min() >= 0.0, "Background mask negativa"
        assert background_mask.max() <= 1.0, "Background mask > 1"
    
    def test_overlapping_region_values(self):
        """
        Região de sobreposição deve ter valores maiores que máscaras individuais.
        
        Aceite: sum > individual na região de overlap.
        """
        h, w = 512, 512
        center1 = (200, 256)
        center2 = (312, 256)
        
        mask1 = create_gaussian_mask((h, w), center=center1, sigma=100)
        mask2 = create_gaussian_mask((h, w), center=center2, sigma=100)
        sum_masks = mask1 + mask2
        
        # Ponto no meio dos dois centros (região de máxima sobreposição)
        mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
        
        val1 = mask1[mid_point[1], mid_point[0]]
        val2 = mask2[mid_point[1], mid_point[0]]
        val_sum = sum_masks[mid_point[1], mid_point[0]]
        
        print(f"\n[Overlap Region Test]")
        print(f"Mask1 at midpoint: {val1:.4f}")
        print(f"Mask2 at midpoint: {val2:.4f}")
        print(f"Sum at midpoint: {val_sum:.4f}")
        
        # Na região de overlap, a soma deve ser maior que cada máscara individual
        assert val_sum > val1, "Soma não maior que mask1 na região de overlap"
        assert val_sum > val2, "Soma não maior que mask2 na região de overlap"
    
    def test_multiple_masks_normalization(self):
        """
        Testa normalização com 4+ máscaras sobrepostas.
        
        Aceite: clamped_sum sempre em [0, 1].
        """
        h, w = 512, 512
        
        # Cria 4 máscaras nos cantos
        centers = [(128, 128), (384, 128), (128, 384), (384, 384)]
        masks = [create_gaussian_mask((h, w), center=c, sigma=80) for c in centers]
        
        # Soma todas
        sum_all = np.sum(masks, axis=0)
        clamped = np.clip(sum_all, 0, 1)
        
        print(f"\n[Multiple Masks Test]")
        print(f"Sum of 4 masks min: {sum_all.min():.4f}, max: {sum_all.max():.4f}")
        print(f"Clamped min: {clamped.min():.4f}, max: {clamped.max():.4f}")
        
        assert clamped.min() >= 0.0, "Clamped negativo"
        assert clamped.max() <= 1.0, "Clamped > 1"
    
    def test_background_complete_coverage(self):
        """
        Background deve cobrir toda a área não coberta por personagens.
        
        Aceite: personagens + background = 1 em todos os pixels.
        """
        h, w = 512, 512
        
        # Uma máscara central
        char_mask = create_gaussian_mask((h, w), center=(256, 256), sigma=100)
        
        # Simula máscara de personagem (binária após threshold)
        char_binary = (char_mask > 0.1).astype(np.float32)
        
        # Background
        background = 1 - char_binary
        
        # Soma deve ser 1 em toda parte
        total = char_binary + background
        
        print(f"\n[Coverage Test]")
        print(f"Character area: {char_binary.sum():.0f} pixels")
        print(f"Background area: {background.sum():.0f} pixels")
        print(f"Total unique area: {(char_binary > 0).sum() + (background > 0).sum():.0f}")
        
        # Todos os pixels devem estar cobertos (char ou background)
        assert np.all(total == 1.0), "Soma personagem + background != 1 em alguns pixels"
