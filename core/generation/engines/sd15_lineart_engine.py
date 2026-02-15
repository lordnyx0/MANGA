
import torch
from PIL import Image, ImageChops
from typing import Dict, Any, Optional, Union, List
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)
from transformers import CLIPVisionModelWithProjection
import numpy as np

from config.settings import DEVICE, DTYPE
from config.settings import QUALITY_PRESETS, V3_STEPS, V3_STRENGTH, V3_GUIDANCE_SCALE, V3_CONTROL_SCALE

from core.generation.engines.vae_dtype_adapter import VAEDtypeAdapter
from core.generation.interfaces import ColorizationEngine
from core.exceptions import ModelLoadError, GenerationError
from core.logging.setup import get_logger

logger = get_logger("SD15LineartEngine")

class SD15LineartEngine(ColorizationEngine):
    """
    Motor v3.0: SD 1.5 + ControlNet Lineart Anime + IP-Adapter.
    Focado em fidelidade de traço e consistência via referências visuais.
    """
    
    def __init__(self, device: str = DEVICE, dtype: torch.dtype = DTYPE):
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self.models_loaded = False
        
        # Caminhos dos modelos
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.controlnet_id = "lllyasviel/control_v11p_sd15s2_lineart_anime"
        self.ip_adapter_repo = "h94/IP-Adapter"
        self.ip_adapter_file = "ip-adapter-plus-face_sd15.bin"
        self.image_encoder_path = "h94/IP-Adapter" # subfolder models/image_encoder
        
    def load_models(self):
        """Carrega pipeline completo na memória."""
        if self.models_loaded:
            return

        logger.info("Carregando modelos v3 (SD 1.5 + Lineart + IP-Adapter)...")
        
        try:
            # 1. Carrega ControlNet
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id,
                torch_dtype=self.dtype
            )
            
            # 2. Carrega Image Encoder e VAE
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.image_encoder_path,
                subfolder="models/image_encoder",
                torch_dtype=self.dtype
            )
            
            # Force VAE to float32 to avoid black images (NaNs) and use mse for better colors
            # Fixes "Solarization" / "Psychedelic" artifacts common with default SD1.5 VAE
            vae_id = "stabilityai/sd-vae-ft-mse"
            logger.info(f"Carregando VAE melhorado: {vae_id}")
            
            vae = AutoencoderKL.from_pretrained(
                vae_id,
                torch_dtype=torch.float32
            )
            # Habilita force_upcast para evitar NaNs e cores psicodélicas
            vae.config.force_upcast = True
            
            # 3. Pipeline Principal
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                vae=vae,
                controlnet=controlnet,
                image_encoder=image_encoder,
                torch_dtype=self.dtype,
                safety_checker=None,
                feature_extractor=None
            )
            
            # CRITICAL FIX for "Fried/Solarized" Colors:
            # The pipeline constructor automatically downcasts the VAE to self.dtype (FP16).
            # We must FORCE it back to FP32 to avoid numerical instability / overflow.
            logger.info("Forçando VAE para Float32 (Pós-Init Pipeline) para corrigir cores...")
            self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)
            
            # 4. Sampler Euler A (Agressivo e Nítido)
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule='scaled_linear'
            )
            
            # 5. VAE MSE (Color Fidelity)
            # Força o uso do VAE especializado para evitar desvios cromáticos
            vae_id = "stabilityai/sd-vae-ft-mse"
            logger.info(f"Forçando VAE de alta fidelidade: {vae_id}")
            self.pipe.vae = AutoencoderKL.from_pretrained(
                vae_id, 
                torch_dtype=self.dtype
            ).to(self.device)
            
            # 6. Carrega IP-Adapter
            self.pipe.load_ip_adapter(
                self.ip_adapter_repo,
                subfolder="models",
                weight_name=self.ip_adapter_file
            )
            
            # 6. Otimizações de Memória (Cruciais para 8-12GB VRAM)
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_vae_slicing()
                
                # Respect ENABLE_VAE_TILING from settings
                from config.settings import ENABLE_VAE_TILING
                if ENABLE_VAE_TILING:
                    logger.info("VAE Tiling HABILTADO via settings.")
                    self.pipe.enable_vae_tiling()
                else:
                    logger.info("VAE Tiling DESABILITADO (Evitando artefatos de borda).")
                    self.pipe.disable_vae_tiling()
                
            self.models_loaded = True
            logger.info("Motor v3 carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro fatal ao carregar modelos v3: {e}")
            raise ModelLoadError(f"Falha ao carregar SD15LineartEngine: {e}")

    def offload_models(self):
        """Libera VRAM (apenas se não estiver usando cpu_offload)"""
        if self.pipe and self.device == "cuda":
            # Com enable_model_cpu_offload, isso é gerenciado automaticamente,
            # mas podemos forçar garbage collection
            pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_page(self, page_image: Image.Image, options: Dict[str, Any]) -> Image.Image:
        """
        Gera página completa.
        Suporta PRE-CLEANING de balões se 'detections' for fornecido em options.
        """
        if not self.models_loaded:
            self.load_models()
            
        prompt = options.get("prompt", "manga coloring, high quality, vibrant colors")
        # Prompt negativo reforçado para evitar cores psicodélicas e artefatos
        default_neg = (
            "monochrome, greyscale, lowres, bad anatomy, worst quality, "
            "oversaturated, neon colors, psychedelic, distorted colors, "
            "blurry, watermark, signature, text, cropped, "
            "glitch, noise, grainy, dark spots"
        )
        neg_prompt = options.get("negative_prompt", default_neg)
        seed = options.get("seed", 42)
        
        # PRE-CLEANING: Limpa balões na imagem de entrada ANTES da IA ver
        # Isso deleta o texto e evita ghosting (a IA recebe um balão branco limpo)
        line_art = page_image.convert("RGB")
        detections = options.get("detections")
        if detections:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(line_art)
            cleaned = 0
            for det in detections:
                if det.get('class_id') == 3 or det.get('class_name') == 'text':
                    bbox = det.get('bbox')
                    if bbox:
                        # Preenche com BRANCO (255, 255, 255) para Lineart normal
                        # Nota: Se a imagem estiver invertida depois, isso vira o sinal de "vazio"
                        x1, y1, x2, y2 = bbox
                        draw.rectangle([x1-2, y1-2, x2+2, y2+2], fill=(255, 255, 255))
                        cleaned += 1
            if cleaned > 0:
                logger.info(f"Pre-Cleaning: {cleaned} balões limpos na imagem de entrada (Anti-Ghosting).")

        mask = Image.new("L", line_art.size, 255) # Máscara completa (toda a imagem)
        
        # Para full page generation, IP-Adapter precisa de uma referência.
        # Se não fornecida, usa prompt apenas (ip_scale=0)
        ref_img = options.get("reference_image")
        
        return self.generate_region(
            line_art=line_art,
            mask=mask,
            reference_image=ref_img,
            prompt=prompt,
            negative_prompt=neg_prompt,
            seed=seed,
            options=options
        )

    def generate_region(
        self,
        line_art: Image.Image,
        mask: Image.Image,
        reference_image: Optional[Image.Image],
        prompt: str,
        negative_prompt: str,
        seed: int,
        options: Dict[str, Any] = None
    ) -> Image.Image:
        """Core generation logic"""
        if not self.models_loaded:
            self.load_models()
        
        options = options or {}
            
        if 'generator' in options and isinstance(options['generator'], torch.Generator):
            generator = options['generator']
        else:
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            except Exception:
                generator = None
        
        # Configura IP-Adapter Scale
        # 0.0 se sem referência, 0.6-0.7 se com referência
        ip_scale = 0.7 if reference_image else 0.0
        self.pipe.set_ip_adapter_scale(ip_scale)
        
        # Prepara referência (se houver)
        ip_image = reference_image if reference_image else Image.new("RGB", (224, 224), (0,0,0))
        
        # Parâmetros de geração (Importados no topo)
        
        quality_mode = options.get("quality_mode", "balanced")
        preset = QUALITY_PRESETS.get(quality_mode, QUALITY_PRESETS["balanced"])
        steps = preset.get("steps", V3_STEPS)
        
        strength = V3_STRENGTH
        control_scale = options.get("control_scale", 0.7) # Reduzido de 0.8 para 0.7 para mais naturalidade
        guidance = V3_GUIDANCE_SCALE
        
        logger.debug(f"Gerando região com IP-Adapter scale={ip_scale}, Steps={steps}")
        
        # DOWN/UPSCALE STRATEGY (Safety for SD 1.5)
        # Avoids "double head" artifacts at high resolutions > 1024px
        original_size = line_art.size
        MAX_DIM = 1024
        
        needs_resize = max(original_size) > MAX_DIM
        
        control_image = None
        
        if needs_resize:
            ratio = MAX_DIM / max(original_size)
            new_w = int(original_size[0] * ratio)
            new_h = int(original_size[1] * ratio)
            target_size = (new_w, new_h)
            logger.info(f"Downscaling for generation: {original_size} -> {target_size}")
            control_image = line_art.resize(target_size, Image.LANCZOS)
        else:
            control_image = line_art
            
        if control_image is None:
             raise RuntimeError("Control image failed to initialize")

        # INVERT IMAGE FOR CONTROLNET (White Lines on Black Background)
        # Manga pages are typically Black Lines on White Background.
        # ControlNet Lineart models expect White Lines on Black Background.
        # Without this, the model sees the white background as "structure" and black lines as empty.
        
        # Check if image is somewhat "inverted" already (e.g. mostly black)
        # Simple heuristic: mean brightness.
        # Manga page is mostly white (> 200). Inverted lineart is mostly black (< 50).
        
        from PIL import ImageOps
        import numpy as np
        
        img_np = np.array(control_image.convert("L"))
        mean_brightness = np.mean(img_np)
        
        if mean_brightness > 127:
            # Likely Black-on-White (Standard Manga) -> INVERT
            # Refinement: Use Threshold to ensure clean lines
            bw_control = control_image.convert("L")
            control_image = ImageOps.invert(bw_control).convert("RGB")
        else:
            # Likely White-on-Black (Already Lineart Map) -> KEEP
            pass
            
        # REGIONAL IP-ADAPTER LOGIC
        # Handles list of images and masks for multi-character support
        cross_attention_kwargs = None
        
        # Extract ip_masks from options if present
        ip_masks = options.get('ip_adapter_masks', None)

        # DEBUG: Save Control Image & Masks
        try:
            import os
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_dir = "output/debug/images"
            lineart_dir = os.path.join(debug_dir, "lineart")
            masks_dir = os.path.join(debug_dir, "masks")
            
            os.makedirs(lineart_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            
            # Save Lineart (Control Image)
            lineart_path = os.path.join(lineart_dir, f"lineart_{timestamp}_{seed}.png")
            control_image.save(lineart_path)
            
            # Save Mask (Inpainting)
            if mask:
                mask_path = os.path.join(masks_dir, f"inpainting_mask_{timestamp}_{seed}.png")
                mask.save(mask_path)
            
            # Save IP-Adapter Masks (Regional)
            if ip_masks:
                for i, ip_mask in enumerate(ip_masks):
                    ip_mask_path = os.path.join(masks_dir, f"ip_mask_{i}_{timestamp}_{seed}.png")
                    ip_mask.save(ip_mask_path)
        except Exception as e:
            logger.warning(f"Failed to save debug images: {e}")
        
        if isinstance(ip_image, list) and not isinstance(ip_image[0], (int, float)):
             # Additional regional logic if needed (e.g. validating lengths)
             pass
        
        if isinstance(ip_image, list) and ip_masks:
             # Validate lengths
             if len(ip_image) != len(ip_masks):
                 logger.warning(f"Mismatch in IP-Adapter images ({len(ip_image)}) and masks ({len(ip_masks)}). Using global.")
             else:
                 # Resize masks to Latent Size
                 # Latent size based on control_image size (which is potentially resized)
                 w_lat, h_lat = control_image.size[0] // 8, control_image.size[1] // 8
                 
                 resized_masks = []
                 for m in ip_masks:
                     if hasattr(m, 'resize'): # PIL
                         m_resized = m.resize((w_lat, h_lat), Image.NEAREST)
                         resized_masks.append(m_resized)
                     else:
                         resized_masks.append(m) # Assume ready
                 
                 cross_attention_kwargs = {"ip_adapter_masks": resized_masks}
                 
                 # Apply scale from settings (default 0.7)
                 scale = options.get('ip_adapter_scale', 0.7)
                 self.pipe.set_ip_adapter_scale(scale)
                 
        # Dynamic IP-Adapter End Step Logic
        end_step_ratio = options.get('ip_adapter_end_step', 1.0)
        target_scale = options.get('ip_adapter_scale', 0.7)
        
        def ip_adapter_step_callback(pipe, step, timestep, callback_kwargs):
            # step is integer 0..num_inference_steps-1
            cutoff_step = int(steps * end_step_ratio)
            
            if step >= cutoff_step:
                pipe.set_ip_adapter_scale(0.0)
            else:
                pipe.set_ip_adapter_scale(target_scale)
            
            return callback_kwargs

        try:
            with torch.inference_mode():
                # Defensiva: Usa Adapter para garantir compatibilidade FP16/FP32 no VAE Decode
                with VAEDtypeAdapter(self.pipe.vae):
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=control_image,
                        num_inference_steps=steps,
                        strength=strength,
                        guidance_scale=guidance,
                        controlnet_conditioning_scale=control_scale,
                        cross_attention_kwargs=cross_attention_kwargs,
                        generator=generator,
                        output_type="pil",
                        ip_adapter_image=ip_image,
                        callback_on_step_end=ip_adapter_step_callback,
                        callback_on_step_end_tensor_inputs=["latents"]
                    )
            
            # Reset scale to avoid side-effects
            self.pipe.set_ip_adapter_scale(target_scale)
            
            # Obtém imagem resultante
            output_image = result.images[0]
            
            # Resize back if needed (CORREÇÃO: movido para antes do return)
            if needs_resize:
                logger.info(f"Upscaling result: {output_image.size} -> {original_size}")
                output_image = output_image.resize(original_size, Image.LANCZOS)
            
            return output_image
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("CUDA OOM in Parallel Regional IP-Adapter. Switching to Sequential Fallback.")
                torch.cuda.empty_cache()
                raise e
            raise

    def compose_final(self, base_image: Image.Image, colorized_image: Image.Image, detections: Optional[List[Dict]] = None) -> Image.Image:
        """
        Combina o traço original (base) com a cor gerada (colorized) usando
        Alpha Compositing (Phase 3 Fix).
        
        Preserva o traço original preto e nítido usando a luminância 
        do lineart como máscara alpha.
        """
        from PIL import ImageDraw, ImageFilter
        
        # Converte para modo compatível
        base = base_image.convert("RGB")
        color = colorized_image.convert("RGB")
        
        # Resize color se necessário (segurança)
        if base.size != color.size:
            color = color.resize(base.size, Image.LANCZOS)
            
        # 1. BUBBLE MASKING: Limpa áreas de texto na camada de cor
        if detections:
            draw = ImageDraw.Draw(color)
            for det in detections:
                if det.get('class_id') == 3 or det.get('class_name') == 'text':
                    bbox = det.get('bbox')
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        draw.rectangle([x1-4, y1-4, x2+4, y2+4], fill=(255, 255, 255))
        
        # 2. SOFT COMPOSITION: Suaviza a cor
        color = color.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # 3. ALPHA BLENDING: (Phase 3 Suggestion)
        # Usa o lineart como máscara: tom escuro -> mais opaco
        line_np = np.array(base).astype(np.float32) / 255.0
        color_np = np.array(color).astype(np.float32) / 255.0
        
        # Máscara de alpha baseada no brilho do lineart (Gray = (R+G+B)/3)
        # Quanto mais preto (0), maior o alpha (1.0).
        gray = np.mean(line_np, axis=2)
        alpha = 1.0 - np.clip(gray, 0.0, 1.0)
        alpha = alpha[..., np.newaxis]
        
        # Blend: color * (1 - alpha) + line * alpha
        # Mantém a cor nos brancos e o traço puro nos pretos
        out_np = color_np * (1.0 - alpha) + line_np * alpha
        
        composed = Image.fromarray((np.clip(out_np, 0.0, 1.0) * 255).astype(np.uint8))
        return composed
