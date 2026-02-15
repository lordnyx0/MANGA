
import torch
import logging

logger = logging.getLogger("VAEDtypeAdapter")

class VAEDtypeAdapter:
    """
    Adaptador para garantir consistência de dtype entre latents e VAE.
    Necessário devido a edge case Windows+CUDA+SD1.5 não coberto por force_upcast.
    
    Issue: RuntimeError 'Input type (struct c10::Half) and bias type (float) should be the same'
           ao decodificar latents FP16 em VAE FP32.
    Solução: Cast explícito antes do decode via Context Manager.
    
    ADICIONAL: Detecta e corrige NaNs/overflows que causam imagens psicodélicas.
    """
    
    def __init__(self, vae):
        self.vae = vae
        self._original_decode = None
    
    def __enter__(self):
        self._original_decode = self.vae.decode
        
        def adapted_decode(latents, return_dict=True, generator=None):
            # Cast seguro: só converte se necessário
            if latents.dtype != self.vae.dtype:
                latents = latents.to(self.vae.dtype)
            
            # CORREÇÃO CRÍTICA: Detecta e corrige NaNs/Inf que causam cores psicodélicas
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                nan_count = torch.isnan(latents).sum().item()
                inf_count = torch.isinf(latents).sum().item()
                logger.warning(f"VAEDtypeAdapter: Detectados {nan_count} NaNs e {inf_count} Infs nos latents! Corrigindo...")
                # Substitui NaNs e Infs por valores seguros
                latents = torch.nan_to_num(latents, nan=0.0, posinf=3.0, neginf=-3.0)
            
            # Clamp para evitar valores extremos que podem causar artefatos
            latents = torch.clamp(latents, min=-5.0, max=5.0)
            
            return self._original_decode(latents, return_dict=return_dict, generator=generator)
        
        self.vae.decode = adapted_decode
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_decode:
            self.vae.decode = self._original_decode
    
    @staticmethod
    def decode_safe(vae, latents):
        """Método estático para uso sem context manager."""
        if latents.dtype != vae.dtype:
            latents = latents.to(vae.dtype)
        
        # CORREÇÃO: Detecta e corrige NaNs/Inf
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            latents = torch.nan_to_num(latents, nan=0.0, posinf=3.0, neginf=-3.0)
        
        latents = torch.clamp(latents, min=-5.0, max=5.0)
        return vae.decode(latents)
