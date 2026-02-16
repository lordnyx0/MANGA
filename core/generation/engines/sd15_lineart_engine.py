
import torch
from PIL import Image
from typing import Dict, Any, Optional, Union, List
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)
from transformers import CLIPVisionModelWithProjection
import numpy as np
import os
from datetime import datetime

from config.settings import DEVICE, DTYPE
from config.settings import (
    QUALITY_PRESETS,
    V3_STEPS,
    V3_STRENGTH,
    V3_GUIDANCE_SCALE,
    V3_CONTROL_SCALE,
    V3_IP_SCALE,
    IP_ADAPTER_END_STEP,
    V3_RETRY_ON_ARTIFACTS,
    V3_SAFE_GUIDANCE_SCALE,
    V3_SAFE_CONTROL_SCALE,
    V3_SAFE_IP_SCALE,
    V3_ARTIFACT_SATURATION_THRESHOLD,
    V3_ARTIFACT_EXTREME_PIXELS_THRESHOLD,
    V3_ARTIFACT_COLOR_STD_THRESHOLD,
    V3_LATENT_ABS_MAX,
    V3_REF_MIN_SIZE,
    V3_REF_MIN_STD,
    V3_COMPOSE_COLOR_BLUR_RADIUS,
    GENERATION_PROFILES_V3,
    SCHEDULER_PROFILES_V3,
)

from core.generation.engines.vae_dtype_adapter import VAEDtypeAdapter
from core.generation.engines.lineart_preprocessor import LineartPreprocessor
from core.generation.engines.composition_service import CompositionService
from core.generation.engines.memory_manager import MemoryManager
from core.generation.interfaces import ColorizationEngine
from core.generation.quality_gate import analyze_avqv_metrics, should_retry_safe
from core.exceptions import ModelLoadError, GenerationError
from core.logging.setup import get_logger

logger = get_logger("SD15LineartEngine")


class SD15LineartEngine(ColorizationEngine):
    """
    Motor v3.0: SD 1.5 + ControlNet Lineart Anime + IP-Adapter.
    Focado em fidelidade de traço e consistência via referências visuais.

    This class is an ORCHESTRATOR — it delegates specialized work to:
    - LineartPreprocessor: image analysis and ControlNet conditioning
    - CompositionService: final compositing and bubble masking
    - MemoryManager: VRAM lifecycle management
    """

    def __init__(
        self,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
        lineart_preprocessor: Optional[LineartPreprocessor] = None,
        composition_service: Optional[CompositionService] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self.models_loaded = False
        self.current_generation_profile = "balanced"

        # Dependency Injection with sensible defaults
        self._preprocessor = lineart_preprocessor or LineartPreprocessor()
        self._compositor = composition_service or CompositionService()
        self._memory = memory_manager or MemoryManager()

        # Model paths
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.controlnet_id = "lllyasviel/control_v11p_sd15s2_lineart_anime"
        self.ip_adapter_repo = "h94/IP-Adapter"
        self.ip_adapter_file = "ip-adapter-plus-face_sd15.bin"
        self.image_encoder_path = "h94/IP-Adapter"  # subfolder models/image_encoder

    # ── Model Lifecycle ──────────────────────────────────────────────────

    def load_models(self):
        """Carrega pipeline completo na memória."""
        if self.models_loaded:
            return

        logger.info("Carregando modelos v3 (SD 1.5 + Lineart + IP-Adapter)...")

        try:
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id, torch_dtype=self.dtype
            )

            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.image_encoder_path,
                subfolder="models/image_encoder",
                torch_dtype=self.dtype,
            )

            # Force VAE to float32 — fixes solarization/psychedelic artifacts
            vae_id = "stabilityai/sd-vae-ft-mse"
            logger.info(f"Carregando VAE melhorado: {vae_id}")
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float32)
            vae.config.force_upcast = True

            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                vae=vae,
                controlnet=controlnet,
                image_encoder=image_encoder,
                torch_dtype=self.dtype,
                safety_checker=None,
                feature_extractor=None,
            )

            # CRITICAL: Force VAE back to FP32 after pipeline constructor downcasts it
            logger.info("Forçando VAE para Float32 (Pós-Init Pipeline)...")
            self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)

            self._apply_scheduler_profile("balanced")

            # Reinforce VAE FP32
            logger.info("Reforçando VAE em Float32 para estabilidade de cor.")
            self.pipe.vae = self.pipe.vae.to(device=self.device, dtype=torch.float32)
            self.pipe.vae.config.force_upcast = True

            self.pipe.load_ip_adapter(
                self.ip_adapter_repo,
                subfolder="models",
                weight_name=self.ip_adapter_file,
            )

            # Delegate memory optimizations
            self._memory.setup_optimizations(self.pipe, self.device)

            self.models_loaded = True
            logger.info("Motor v3 carregado com sucesso.")

        except Exception as e:
            logger.error(f"Erro fatal ao carregar modelos v3: {e}")
            raise ModelLoadError(f"Falha ao carregar SD15LineartEngine: {e}")

    def offload_models(self):
        """Libera VRAM (delega ao MemoryManager)."""
        self._memory.offload()

    # ── Scheduler ────────────────────────────────────────────────────────

    def _apply_scheduler_profile(self, profile: str):
        cfg = SCHEDULER_PROFILES_V3.get(profile, SCHEDULER_PROFILES_V3.get("balanced", {}))
        beta_start = float(cfg.get("beta_start", 0.00085))
        beta_end = float(cfg.get("beta_end", 0.012))
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
        )
        self.current_generation_profile = profile

    # ── Reference Validation ─────────────────────────────────────────────

    @staticmethod
    def _reference_is_valid(image: Image.Image) -> bool:
        if image is None:
            return False
        if min(image.size) < V3_REF_MIN_SIZE:
            return False
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        if float(np.std(arr)) < V3_REF_MIN_STD:
            return False
        return True

    @staticmethod
    def _normalize_reference_image(image: Image.Image) -> Image.Image:
        """Normaliza referência para CLIP/IP-Adapter (RGB 224x224)."""
        img = image.convert("RGB")
        w, h = img.size
        side = min(w, h)
        left = max(0, (w - side) // 2)
        top = max(0, (h - side) // 2)
        img = img.crop((left, top, left + side, top + side))
        return img.resize((224, 224), Image.LANCZOS)

    @staticmethod
    def _normalize_ip_adapter_mask(mask: Any) -> Optional[Image.Image]:
        """Normaliza máscara regional para PIL L (0-255)."""
        if isinstance(mask, Image.Image):
            return mask.convert("L")

        if isinstance(mask, np.ndarray):
            arr = np.asarray(mask)
            if arr.ndim == 3:
                arr = arr[..., 0]
            if arr.ndim != 2:
                return None
            if arr.dtype != np.uint8:
                scale = 255.0 if float(np.nanmax(arr)) <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="L")

        return None

    # ── Quality Analysis (delegates to quality_gate) ─────────────────────

    # Keep _compute_lineart_metrics as a static delegation for backward compat
    @staticmethod
    def _compute_lineart_metrics(image_gray: Image.Image) -> Dict[str, float]:
        return LineartPreprocessor.compute_metrics(image_gray)

    @staticmethod
    def _analyze_image_artifacts(image: Image.Image) -> Dict[str, float]:
        """Retorna métricas AVQV mínimas para detectar saída psicodélica/frita."""
        return analyze_avqv_metrics(image)

    @staticmethod
    def _is_psychedelic_output(metrics: Dict[str, float]) -> bool:
        return should_retry_safe(
            metrics,
            {
                "saturation_mean": V3_ARTIFACT_SATURATION_THRESHOLD,
                "extreme_pixels_ratio": V3_ARTIFACT_EXTREME_PIXELS_THRESHOLD,
                "color_std": V3_ARTIFACT_COLOR_STD_THRESHOLD,
            },
        )

    # ── Latent Safety ────────────────────────────────────────────────────

    def _compute_dynamic_latent_abs_limit(self, step: int) -> float:
        """
        Retorna limite dinâmico para magnitude de latents.

        Em schedulers ancestrais (ex.: Euler A), o ruído inicial tem sigma alto,
        logo a magnitude absoluta dos latents no começo pode ser naturalmente
        muito maior que o limiar base estático. Usar limite proporcional ao sigma
        evita falso-positivo no step inicial e mantém proteção nos steps finais.
        """
        limit = float(V3_LATENT_ABS_MAX)
        scheduler = getattr(self.pipe, "scheduler", None)
        sigmas = getattr(scheduler, "sigmas", None)
        if sigmas is None:
            return limit

        try:
            idx = max(0, min(int(step), len(sigmas) - 1))
            sigma = float(sigmas[idx])
            return max(limit, sigma * 6.0)
        except (TypeError, ValueError, RuntimeError, IndexError):
            return limit

    # ── Generation Core ──────────────────────────────────────────────────

    def _run_generation_pass(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        control_image: Image.Image,
        steps: int,
        strength: float,
        guidance: float,
        control_scale: float,
        cross_attention_kwargs: Optional[Dict[str, Any]],
        generator: Optional[torch.Generator],
        ip_image: Any,
        callback,
    ):
        with torch.inference_mode():
            with VAEDtypeAdapter(self.pipe.vae):
                return self.pipe(
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
                    callback_on_step_end=callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                )

    # ── Public API ───────────────────────────────────────────────────────

    def generate_page(self, page_image: Image.Image, options: Dict[str, Any]) -> Image.Image:
        """
        Gera página completa.
        Suporta PRE-CLEANING de balões se 'detections' for fornecido em options.
        """
        if not self.models_loaded:
            self.load_models()

        prompt = options.get("prompt", "manga coloring, high quality, vibrant colors")
        default_neg = (
            "monochrome, greyscale, lowres, bad anatomy, worst quality, "
            "oversaturated, neon colors, psychedelic, distorted colors, "
            "blurry, watermark, signature, text, cropped, "
            "glitch, noise, grainy, dark spots"
        )
        neg_prompt = options.get("negative_prompt", default_neg)
        seed = options.get("seed", 42)

        # Pre-cleaning: delegate bubble masking to CompositionService
        line_art = page_image.convert("RGB")
        detections = options.get("detections")
        if detections:
            line_art = self._compositor.clean_bubble_regions(line_art, detections)

        mask = Image.new("L", line_art.size, 255)
        ref_img = options.get("reference_image")

        return self.generate_region(
            line_art=line_art,
            mask=mask,
            reference_image=ref_img,
            prompt=prompt,
            negative_prompt=neg_prompt,
            seed=seed,
            options=options,
        )

    def generate_region(
        self,
        line_art: Image.Image,
        mask: Image.Image,
        reference_image: Optional[Image.Image],
        prompt: str,
        negative_prompt: str,
        seed: int,
        options: Dict[str, Any] = None,
    ) -> Image.Image:
        """Core generation logic"""
        if not self.models_loaded:
            self.load_models()

        options = options or {}

        if "generator" in options and isinstance(options["generator"], torch.Generator):
            generator = options["generator"]
        else:
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            except Exception:
                generator = None

        # Reference validation
        if isinstance(reference_image, list):
            valid_refs = [
                self._normalize_reference_image(img)
                for img in reference_image
                if isinstance(img, Image.Image) and self._reference_is_valid(img)
            ]
            if len(valid_refs) != len(reference_image):
                logger.warning("Referências inválidas removidas do IP-Adapter regional.")
            reference_image = valid_refs if valid_refs else None
        elif isinstance(reference_image, Image.Image):
            if not self._reference_is_valid(reference_image):
                logger.warning("Referência inválida detectada. IP-Adapter será desativado.")
                reference_image = None
            else:
                reference_image = self._normalize_reference_image(reference_image)

        # Generation profile
        generation_profile = options.get("generation_profile", "balanced")
        profile_cfg = GENERATION_PROFILES_V3.get(
            generation_profile, GENERATION_PROFILES_V3["balanced"]
        )
        if generation_profile != self.current_generation_profile:
            self._apply_scheduler_profile(generation_profile)

        # IP-Adapter scale
        default_ip_scale = float(profile_cfg.get("ip_scale", V3_IP_SCALE))
        ip_scale = (
            options.get("ip_adapter_scale", default_ip_scale)
            if reference_image
            else 0.0
        )
        self.pipe.set_ip_adapter_scale(ip_scale)

        ip_image = reference_image if reference_image else Image.new("RGB", (224, 224), (0, 0, 0))

        # Generation parameters
        quality_mode = options.get("quality_mode", "balanced")
        preset = QUALITY_PRESETS.get(quality_mode, QUALITY_PRESETS["balanced"])
        steps = int(options.get("steps", preset.get("steps", V3_STEPS)))
        strength = V3_STRENGTH
        control_scale = options.get(
            "control_scale",
            float(profile_cfg.get("control_scale", V3_CONTROL_SCALE)),
        )
        guidance = options.get(
            "guidance_scale",
            float(profile_cfg.get("guidance_scale", V3_GUIDANCE_SCALE)),
        )

        logger.info(
            "Generation profile=%s steps=%s guidance=%.2f control=%.2f ip_scale=%.2f",
            generation_profile, steps, guidance, control_scale, ip_scale,
        )

        # Delegate lineart preprocessing to LineartPreprocessor
        control_image, needs_resize, original_size = (
            self._preprocessor.prepare_control_image(line_art)
        )

        if control_image is None:
            raise RuntimeError("Control image failed to initialize")

        # Regional IP-Adapter logic
        cross_attention_kwargs = None
        ip_masks = options.get("ip_adapter_masks", None)

        # Debug: save control image & masks
        self._save_debug_images(control_image, mask, ip_masks, seed)

        if isinstance(ip_image, list) and ip_image and not isinstance(ip_image[0], (int, float)):
            pass

        if isinstance(ip_image, list) and ip_masks:
            if len(ip_image) != len(ip_masks):
                logger.warning(
                    f"Mismatch in IP-Adapter images ({len(ip_image)}) "
                    f"and masks ({len(ip_masks)}). Using global."
                )
            else:
                w_lat = control_image.size[0] // 8
                h_lat = control_image.size[1] // 8

                resized_masks = []
                for m in ip_masks:
                    normalized_mask = self._normalize_ip_adapter_mask(m)
                    if normalized_mask is None:
                        logger.warning(
                            "Máscara IP-Adapter com tipo/formato inválido: %s",
                            type(m).__name__,
                        )
                        continue
                    resized_masks.append(
                        normalized_mask.resize((w_lat, h_lat), Image.NEAREST)
                    )

                cross_attention_kwargs = {"ip_adapter_masks": resized_masks}

                scale = (
                    options.get("ip_adapter_scale", V3_IP_SCALE)
                    if reference_image
                    else 0.0
                )
                self.pipe.set_ip_adapter_scale(scale)
                ip_scale = scale

        # Dynamic IP-Adapter end step
        end_step_ratio = float(options.get("ip_adapter_end_step", IP_ADAPTER_END_STEP))
        end_step_ratio = max(0.0, min(1.0, end_step_ratio))
        target_scale = ip_scale if end_step_ratio > 0.0 else 0.0

        def ip_adapter_step_callback(pipe, step, timestep, callback_kwargs):
            latents = (
                callback_kwargs.get("latents")
                if isinstance(callback_kwargs, dict)
                else None
            )
            if isinstance(latents, torch.Tensor):
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    raise GenerationError("Latents inválidos detectados (NaN/Inf)")
                max_abs = float(torch.max(torch.abs(latents)).detach().cpu().item())
                dynamic_limit = self._compute_dynamic_latent_abs_limit(step)
                if max_abs > dynamic_limit:
                    raise GenerationError(
                        f"Latents com magnitude extrema detectados "
                        f"(max_abs={max_abs:.2f}, limite={dynamic_limit:.2f}, step={step})"
                    )

            if end_step_ratio <= 0.0:
                pipe.set_ip_adapter_scale(0.0)
                return callback_kwargs

            cutoff_step = int(steps * end_step_ratio)
            if step >= cutoff_step:
                pipe.set_ip_adapter_scale(0.0)
            else:
                pipe.set_ip_adapter_scale(target_scale)

            return callback_kwargs

        try:
            result = self._run_generation_pass(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,
                steps=steps,
                strength=strength,
                guidance=guidance,
                control_scale=control_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                generator=generator,
                ip_image=ip_image,
                callback=ip_adapter_step_callback,
            )

            self.pipe.set_ip_adapter_scale(target_scale)

            output_image = result.images[0]
            metrics = self._analyze_image_artifacts(output_image)
            logger.info(
                "Output metrics: sat=%.3f extreme=%.3f color_std=%.3f",
                metrics["saturation_mean"],
                metrics["extreme_pixels_ratio"],
                metrics["color_std"],
            )

            # Quality gate + retry with safe profile
            if V3_RETRY_ON_ARTIFACTS and self._is_psychedelic_output(metrics):
                logger.warning("Artifact gate acionado. Reexecutando com profile SAFE.")
                safe_profile = GENERATION_PROFILES_V3.get("safe", {})
                safe_guidance = min(
                    guidance,
                    float(safe_profile.get("guidance_scale", V3_SAFE_GUIDANCE_SCALE)),
                )
                safe_control_scale = min(
                    control_scale,
                    float(safe_profile.get("control_scale", V3_SAFE_CONTROL_SCALE)),
                )
                safe_default_ip = float(
                    safe_profile.get("ip_scale", V3_SAFE_IP_SCALE)
                )
                safe_ip_scale = (
                    min(target_scale, safe_default_ip) if reference_image else 0.0
                )
                target_scale = safe_ip_scale
                self.pipe.set_ip_adapter_scale(safe_ip_scale)

                safe_result = self._run_generation_pass(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    control_image=control_image,
                    steps=steps,
                    strength=strength,
                    guidance=safe_guidance,
                    control_scale=safe_control_scale,
                    cross_attention_kwargs=cross_attention_kwargs,
                    generator=generator,
                    ip_image=ip_image,
                    callback=ip_adapter_step_callback,
                )
                output_image = safe_result.images[0]
                safe_metrics = self._analyze_image_artifacts(output_image)
                logger.info(
                    "SAFE output metrics: sat=%.3f extreme=%.3f color_std=%.3f",
                    safe_metrics["saturation_mean"],
                    safe_metrics["extreme_pixels_ratio"],
                    safe_metrics["color_std"],
                )
                self.pipe.set_ip_adapter_scale(target_scale)

            # Resize back if needed
            if needs_resize:
                logger.info(f"Upscaling result: {output_image.size} -> {original_size}")
                output_image = output_image.resize(original_size, Image.LANCZOS)

            return output_image

        except (RuntimeError, GenerationError) as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    "CUDA OOM in Parallel Regional IP-Adapter. "
                    "Switching to Sequential Fallback."
                )
                torch.cuda.empty_cache()
                raise e
            raise

    def compose_final(
        self,
        base_image: Image.Image,
        colorized_image: Image.Image,
        detections: Optional[List[Dict]] = None,
    ) -> Image.Image:
        """
        Combina o traço original (base) com a cor gerada (colorized).
        Delegates to CompositionService.
        """
        return self._compositor.compose_final(
            base_image,
            colorized_image,
            detections=detections,
            blur_radius=float(V3_COMPOSE_COLOR_BLUR_RADIUS),
        )

    # ── Debug Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _save_debug_images(
        control_image: Image.Image,
        mask: Image.Image,
        ip_masks: Optional[list],
        seed: int,
    ) -> None:
        """Save debug images for control/mask inspection."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_dir = "output/debug/images"
            lineart_dir = os.path.join(debug_dir, "lineart")
            masks_dir = os.path.join(debug_dir, "masks")

            os.makedirs(lineart_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            lineart_path = os.path.join(lineart_dir, f"lineart_{timestamp}_{seed}.png")
            control_image.save(lineart_path)

            if mask:
                mask_path = os.path.join(masks_dir, f"inpainting_mask_{timestamp}_{seed}.png")
                mask.save(mask_path)

            if ip_masks:
                for i, ip_mask in enumerate(ip_masks):
                    ip_mask_path = os.path.join(
                        masks_dir, f"ip_mask_{i}_{timestamp}_{seed}.png"
                    )
                    ip_mask.save(ip_mask_path)
        except Exception as e:
            logger.warning(f"Failed to save debug images: {e}")
