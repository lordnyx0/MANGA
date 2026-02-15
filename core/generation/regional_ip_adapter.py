"""
Regional IP-Adapter com Early-Heavy Injection para SDXL-Lightning 4-Step.

Baseado em:
- T-GATE (ICML 2024): Early stopping de cross-attention em few-steps models
- ICAS (2025): Multi-embedding cyclic injection superior a simultaneous

Este módulo implementa IP-Adapter regional usando API nativa do Diffusers ≥0.29.0,
com estratégia de injeção temporal otimizada para mangá (anime).
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from PIL import Image
from dataclasses import dataclass
import gc

# Diffusers imports
try:
    from diffusers.image_processor import IPAdapterMaskProcessor
    from diffusers import StableDiffusionXLControlNetPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("[RegionalIP] Aviso: Diffusers não disponível. Regional IP-Adapter desativado.")


@dataclass
class RegionalCharacter:
    """
    Estrutura de dados para personagem regional.
    
    Args:
        char_id: Identificador único do personagem
        crop_image: PIL Image do crop do personagem (para IP-Adapter)
        mask: Array numpy (H, W) com valores 0.0-1.0
        scale: Scale inicial (será modificado pelo callback dinamicamente)
    """
    char_id: str
    crop_image: Image.Image
    mask: np.ndarray
    scale: float = 0.6


class EarlyHeavyRegionalIP:
    """
    Controlador de IP-Adapter com otimização temporal Early-Heavy para 4 steps.
    
    Estratégia para mangá:
    - Usa IP-Adapter Plus Face ViT-H para máxima identidade
    - Scale controlado dinamicamente por step (callback)
    - Step 0: 1.0 (máxima força para semântica)
    - Step 1: 0.6 (fade)
    - Steps 2-3: 0.0 (desligado, ControlNet domina)
    
    Para múltiplos personagens (máx 2 na RTX 3060):
    - Step 0: [1.0, 0.0] - Personagem A
    - Step 1: [0.0, 1.0] - Personagem B (ou fade se único)
    - Steps 2-3: [0.0, 0.0] - Desligado
    
    Args:
        pipeline: Pipeline SDXL já carregado com ControlNet
        device: Dispositivo ("cuda" ou "cpu")
        dtype: Tipo de dados (torch.float16 recomendado)
        model_name: Nome do modelo IP-Adapter (padrão: plus-face ViT-H)
    """
    
    def __init__(
        self,
        pipeline,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        model_name: str = "ip-adapter_sdxl.bin"  # Modelo padrão (ViT-L) - compatível com Diffusers
    ):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers não disponível. Instale: pip install diffusers>=0.29.0")
        
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        self.mask_processor = IPAdapterMaskProcessor()
        self._ip_adapter_loaded = False
        
        # Carregar IP-Adapter padrão (ViT-L) - compatível com Diffusers
        self._load_ip_adapter()
        
        # Otimizações de VRAM obrigatórias para RTX 3060 12GB
        self._setup_memory_optimizations()
    
    def _load_ip_adapter(self):
        """Carrega o modelo IP-Adapter padrão (2x para Regional IP)."""
        try:
            print(f"[RegionalIP] Carregando 2x {self.model_name} para Regional IP...")
            
            # Carregar 2 IP-Adapters (máximo para RTX 3060 12GB)
            # Usando modelo padrão ViT-L (1280 dims) - compatível com Diffusers
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name=[self.model_name, self.model_name],  # 2 IP-Adapters
                torch_dtype=self.dtype
            )
            
            self._ip_adapter_loaded = True
            print("[RegionalIP] 2 IP-Adapters carregados com sucesso!")
            
        except Exception as e:
            print(f"[RegionalIP] Erro ao carregar IP-Adapter: {e}")
            self._ip_adapter_loaded = False
            raise
    
    def _setup_memory_optimizations(self):
        """Configura otimizações de memória obrigatórias."""
        try:
            # CPU offload é essencial para economia de VRAM na RTX 3060
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
                print("[RegionalIP] CPU offload ativado")
            
            # VAE slicing para imagens grandes
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
                print("[RegionalIP] VAE slicing ativado")
            
            # VAE tiling para imagens muito grandes (>1536px)
            if hasattr(self.pipeline, 'enable_vae_tiling'):
                self.pipeline.enable_vae_tiling()
                print("[RegionalIP] VAE tiling ativado")
                
        except Exception as e:
            print(f"[RegionalIP] Aviso: Erro nas otimizações de memória: {e}")
    
    def generate_regional(
        self,
        prompt: str,
        negative_prompt: str = "",
        characters: Optional[List[RegionalCharacter]] = None,
        controlnet_image: Optional[Image.Image] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.2,
        height: int = 1024,
        width: int = 1408,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        """
        Gera imagem com IP-Adapter regional e Early-Heavy Injection.
        
        Args:
            prompt: Prompt de texto (ex: "manga colorization, anime style")
            negative_prompt: Prompt negativo
            characters: Lista de RegionalCharacter (máximo 2 para RTX 3060 12GB)
            controlnet_image: Imagem Canny para ControlNet
            num_inference_steps: Deve ser 4 (SDXL-Lightning)
            guidance_scale: Scale do CFG (1.2 para Lightning)
            height: Altura da imagem
            width: Largura da imagem
            generator: Gerador para reprodutibilidade
            
        Returns:
            Imagem PIL gerada
            
        Raises:
            ValueError: Se mais de 2 personagens (limitação VRAM)
            RuntimeError: Se IP-Adapter não foi carregado
        """
        if not self._ip_adapter_loaded:
            raise RuntimeError("IP-Adapter não foi carregado corretamente")
        
        # Sem personagens = geração base
        if not characters:
            return self._generate_base(
                prompt, negative_prompt, controlnet_image,
                num_inference_steps, guidance_scale, height, width, generator
            )
        
        # Limitação de VRAM para RTX 3060 12GB
        if len(characters) > 2:
            print(f"[RegionalIP] Aviso: {len(characters)} personagens detectados.")
            print("[RegionalIP] RTX 3060 12GB suporta máximo 2. Processando apenas os 2 primeiros.")
            characters = characters[:2]
        
        num_chars = len(characters)
        print(f"[RegionalIP] Gerando com {num_chars} personagem(ns) (Early-Heavy strategy)")
        
        # NOTA: Sempre carregamos 2 IP-Adapters, então precisamos sempre passar
        # 2 imagens e 2 scales, mesmo quando temos apenas 1 personagem.
        # O segundo IP-Adapter fica com scale 0 quando não há segundo personagem.
        
        # 1. Preparar imagens de referência (crops dos personagens)
        # SEMPRE passar 2 imagens (mesmo que uma seja dummy para o segundo IP-Adapter)
        if num_chars == 1:
            # Com 1 personagem: passar a mesma imagem para ambos os IP-Adapters
            # mas o segundo terá scale 0
            reference_images = [[characters[0].crop_image], [characters[0].crop_image]]
        else:
            # Com 2 personagens: cada um para seu IP-Adapter
            reference_images = [[char.crop_image] for char in characters]
        
        # 2. Preparar máscaras regionais
        # SEMPRE criar 2 máscaras (uma para cada IP-Adapter)
        masks = [char.mask for char in characters]
        
        # Se temos apenas 1 personagem, adicionar uma máscara vazia (zeros) para o segundo IP-Adapter
        if num_chars == 1:
            # Criar máscara vazia do mesmo tamanho
            empty_mask = np.zeros_like(masks[0])
            masks.append(empty_mask)
        
        processed_masks = self.mask_processor.preprocess(
            masks, height=height, width=width
        )
        
        # O Diffusers espera uma LISTA de máscaras, não um tensor 4D
        # Cada máscara deve ter shape [1, H, W] (um tensor por IP-Adapter)
        # processed_masks tem shape [2, 1, H, W] ou similar
        # Precisamos converter para lista de 2 tensores com shape [1, H, W]
        
        # Garantir que temos as dimensões corretas
        # O Diffusers espera: lista de tensores com shape [1, num_images, H, W]
        # Como cada IP-Adapter tem 1 imagem: shape deve ser [1, 1, H, W]
        print(f"[RegionalIP] Máscaras pré-processadas raw shape: {processed_masks.shape}, dim: {processed_masks.dim()}")
        
        # Garantir que temos 2 máscaras e cada uma com shape [1, 1, H, W]
        H, W = processed_masks.shape[-2], processed_masks.shape[-1]
        
        if processed_masks.dim() == 4 and processed_masks.shape[0] == 2:
            # Shape: [2, C, H, W] onde C pode ser 1 ou outro valor
            # Extrair cada máscara e garantir shape [1, 1, H, W]
            mask0 = processed_masks[0]
            mask1 = processed_masks[1]
            
            # Remover dimensões extras se necessário, depois adicionar as corretas
            # Garantir que termina com [1, 1, H, W]
            mask0 = mask0.reshape(1, 1, H, W)
            mask1 = mask1.reshape(1, 1, H, W)
            
            ip_adapter_masks = [mask0, mask1]
        elif processed_masks.dim() == 4 and processed_masks.shape[1] == 2:
            # Shape: [batch, 2, H, W] ou [1, 2, H, W]
            masks_squeezed = processed_masks.squeeze(0)  # [2, H, W]
            mask0 = masks_squeezed[0].reshape(1, 1, H, W)
            mask1 = masks_squeezed[1].reshape(1, 1, H, W)
            ip_adapter_masks = [mask0, mask1]
        elif processed_masks.dim() == 3:
            # Shape: [2, H, W] ou [num_masks, H, W]
            if processed_masks.shape[0] >= 2:
                mask0 = processed_masks[0].reshape(1, 1, H, W)
                mask1 = processed_masks[1].reshape(1, 1, H, W)
                ip_adapter_masks = [mask0, mask1]
            else:
                raise ValueError(f"Esperado pelo menos 2 máscaras, mas shape é {processed_masks.shape}")
        else:
            # Fallback: flatten e reshape
            masks_flat = processed_masks.reshape(-1, H, W)
            if masks_flat.shape[0] >= 2:
                mask0 = masks_flat[0].reshape(1, 1, H, W)
                mask1 = masks_flat[1].reshape(1, 1, H, W)
                ip_adapter_masks = [mask0, mask1]
            else:
                raise ValueError(f"Não foi possível extrair 2 máscaras de shape {processed_masks.shape}")
        
        print(f"[RegionalIP] Máscaras processadas: {len(ip_adapter_masks)} máscaras")
        for i, m in enumerate(ip_adapter_masks):
            print(f"  - Máscara {i}: shape={m.shape}")
        
        # 3. Configurar escala inicial (será modificada pelo callback)
        # SEMPRE passar 2 scales (para os 2 IP-Adapters carregados)
        if num_chars == 1:
            initial_scales = [[1.0], [0.0]]  # IP-Adapter 1 ativo, IP-Adapter 2 inativo
        else:
            initial_scales = [[1.0], [0.0]]  # Começa com personagem 1
        self.pipeline.set_ip_adapter_scale(initial_scales)
        
        # 4. Callback de Early-Heavy Injection (T-GATE + ICAS)
        def early_heavy_callback(pipe, step_index, timestep, callback_kwargs):
            """
            Estratégia Early-Heavy otimizada para 4 steps.
            SEMPRE retorna 2 scales (para os 2 IP-Adapters carregados).
            
            Com 2 personagens (cíclica):
            - Step 0: [[1.0], [0.0]] - Personagem A "carimba" identidade
            - Step 1: [[0.0], [1.0]] - Personagem B "carimba" identidade
            - Steps 2-3: [[0.0], [0.0]] - Desligado
            
            Com 1 personagem:
            - Step 0: [[1.0], [0.0]] - Máxima força no IP-Adapter 1
            - Step 1: [[0.6], [0.0]] - Fade no IP-Adapter 1
            - Steps 2-3: [[0.0], [0.0]] - Desligado
            """
            if step_index == 0:
                # Semantics planning: máxima força no primeiro personagem
                if num_chars == 1:
                    scales = [[1.0], [0.0]]
                    print(f"[RegionalIP] Step 0: Injetando personagem com força máxima")
                else:
                    scales = [[1.0], [0.0]]
                    print(f"[RegionalIP] Step 0: Injetando personagem 1 com força máxima")
                    
            elif step_index == 1:
                if num_chars == 1:
                    # Fade para personagem único
                    scales = [[0.6], [0.0]]
                    print(f"[RegionalIP] Step 1: Fade do personagem (scale 0.6)")
                else:
                    # Alternância cíclica para segundo personagem
                    scales = [[0.0], [1.0]]
                    print(f"[RegionalIP] Step 1: Injetando personagem 2 com força máxima")
            else:
                # Fidelity improving: desliga ambos os IP-Adapters
                scales = [[0.0], [0.0]]
                print(f"[RegionalIP] Step {step_index}: IP-Adapter desligado (ControlNet domina)")
            
            pipe.set_ip_adapter_scale(scales)
            return callback_kwargs
        
        # 5. Executar geração
        try:
            print(f"[RegionalIP] Iniciando geração ({height}x{width}, {num_inference_steps} steps)")
            
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=controlnet_image,
                ip_adapter_image=reference_images,
                cross_attention_kwargs={"ip_adapter_masks": ip_adapter_masks},
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                callback_on_step_end=early_heavy_callback,
            ).images[0]
            
            print("[RegionalIP] Geração concluída com sucesso!")
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"[RegionalIP] ERRO: Out of Memory!")
            print(f"[RegionalIP] Tentando fallback com 1 personagem...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Fallback: gerar com apenas 1 personagem
            if len(characters) > 1:
                return self.generate_regional(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    characters=[characters[0]],  # Apenas o primeiro
                    controlnet_image=controlnet_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                )
            else:
                # Último recurso: gerar sem IP-Adapter
                return self._generate_base(
                    prompt, negative_prompt, controlnet_image,
                    num_inference_steps, guidance_scale, height, width, generator
                )
        
        except Exception as e:
            print(f"[RegionalIP] Erro na geração: {e}")
            raise
    
    def _generate_base(
        self,
        prompt: str,
        negative_prompt: str,
        controlnet_image: Optional[Image.Image],
        num_inference_steps: int = 4,
        guidance_scale: float = 1.2,
        height: int = 1024,
        width: int = 1408,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        """
        Geração base sem IP-Adapter (fallback).
        """
        print("[RegionalIP] Usando geração base (sem IP-Adapter)")
        
        # Desliga IP-Adapter se estiver ativo (desliga ambos os 2 IP-Adapters)
        if self._ip_adapter_loaded:
            self.pipeline.set_ip_adapter_scale([[0.0], [0.0]])
        
        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=controlnet_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]
    
    def unload(self):
        """Descarrega modelos da VRAM."""
        if self._ip_adapter_loaded:
            try:
                self.pipeline.unload_ip_adapter()
                self._ip_adapter_loaded = False
                print("[RegionalIP] IP-Adapter descarregado")
            except Exception as e:
                print(f"[RegionalIP] Erro ao descarregar: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# Função auxiliar para criar máscaras gaussianas
def create_gaussian_mask(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    sigma: float = 20.0
) -> np.ndarray:
    """
    Cria máscara gaussiana para um personagem.
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box do personagem
        image_size: (width, height) tamanho da imagem
        sigma: Desvio padrão do gaussiano (maior = mais suave)
        
    Returns:
        Array numpy (H, W) com valores 0.0-1.0
    """
    width, height = image_size
    x1, y1, x2, y2 = bbox
    
    # Cria grid de coordenadas
    y, x = np.ogrid[:height, :width]
    
    # Centro do bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Sigma baseado no tamanho do personagem
    sigma_x = max((x2 - x1) / 2.5, sigma)
    sigma_y = max((y2 - y1) / 2.5, sigma)
    
    # Gaussiana 2D
    mask = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + 
                    (y - center_y)**2 / (2 * sigma_y**2)))
    
    return mask.astype(np.float32)


def create_background_mask(
    character_masks: List[np.ndarray]
) -> np.ndarray:
    """
    Cria máscara de background (inverso dos personagens).
    
    Args:
        character_masks: Lista de máscaras de personagens
        
    Returns:
        Máscara de background (0.0 = personagem, 1.0 = background)
    """
    if not character_masks:
        return np.ones_like(character_masks[0])
    
    # Combina todas as máscaras de personagens
    combined = np.maximum.reduce(character_masks)
    
    # Background é o inverso
    background = 1.0 - combined
    
    return background
