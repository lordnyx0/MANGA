"""
MangaAutoColor Pro - Logger Detalhado de Geração

Registra todo o processo de geração incluindo:
- Prompts usados em cada etapa
- Configurações aplicadas
- Detecções e embeddings
- Tempos de processamento
- Erros e warnings
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class GenerationStep:
    """Representa uma etapa do processo de geração"""
    step_name: str
    start_time: float
    end_time: Optional[float] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    config: Optional[Dict] = None
    input_info: Optional[Dict] = None
    output_info: Optional[Dict] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    def finish(self):
        """Finaliza a etapa registrando o tempo"""
        self.end_time = time.time()
    
    def add_error(self, error: str):
        """Adiciona erro à etapa"""
        self.errors.append(f"[{datetime.now().isoformat()}] {error}")
    
    def add_warning(self, warning: str):
        """Adiciona warning à etapa"""
        self.warnings.append(f"[{datetime.now().isoformat()}] {warning}")
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        duration = (self.end_time - self.start_time) if self.end_time else None
        return {
            "step_name": self.step_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": round(duration, 2) if duration else None,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "config": self.config,
            "input_info": self.input_info,
            "output_info": self.output_info,
            "errors": self.errors,
            "warnings": self.warnings
        }


class GenerationLogger:
    """
    Logger detalhado do processo de geração.
    
    Cria arquivos JSON estruturados na pasta logs do capítulo.
    """
    
    def __init__(self, chapter_id: str, output_dir: str):
        self.chapter_id = chapter_id
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Diretórios para imagens de debug
        self.images_dir = self.output_dir / "images"
        self.crops_dir = self.images_dir / "crops"
        self.canny_dir = self.images_dir / "canny"
        self.input_dir = self.images_dir / "input"
        self.detections_dir = self.images_dir / "detections"
        self.masks_dir = self.images_dir / "masks"  # ADR 004
        
        # Cria diretórios de imagens
        for d in [self.crops_dir, self.canny_dir, self.input_dir, self.detections_dir, self.masks_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Dados do log
        self.session_start = time.time()
        self.steps: List[GenerationStep] = []
        self.current_step: Optional[GenerationStep] = None
        self.global_config: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            "chapter_id": chapter_id,
            "start_time": datetime.now().isoformat(),
            "output_directory": str(output_dir)
        }
        
        # Arquivos de log
        self.main_log_file = self.logs_dir / "generation_log.json"
        self.prompts_file = self.logs_dir / "prompts_used.txt"
        self.timeline_file = self.logs_dir / "timeline.txt"
        
        print(f"[GenerationLogger] Inicializado: {self.logs_dir}")
    
    def set_global_config(self, config: Dict):
        """Define configurações globais da sessão"""
        self.global_config = config
        self._save_main_log()
    
    def start_step(self, step_name: str, input_info: Optional[Dict] = None) -> GenerationStep:
        """Inicia uma nova etapa de geração"""
        # Finaliza etapa anterior se existir
        if self.current_step and not self.current_step.end_time:
            self.current_step.finish()
            self.steps.append(self.current_step)
        
        # Cria nova etapa
        self.current_step = GenerationStep(
            step_name=step_name,
            start_time=time.time(),
            input_info=input_info
        )
        
        print(f"[GenerationLogger] Iniciando: {step_name}")
        return self.current_step
    
    def log_prompts(self, prompt: str, negative_prompt: str = "", config: Optional[Dict] = None):
        """Registra prompts usados na etapa atual"""
        if self.current_step:
            self.current_step.prompt = prompt
            self.current_step.negative_prompt = negative_prompt
            self.current_step.config = config or {}
        
        # Também salva no arquivo de prompts
        timestamp = datetime.now().strftime("%H:%M:%S")
        step_name = self.current_step.step_name if self.current_step else "unknown"
        
        with open(self.prompts_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{timestamp}] ETAPA: {step_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write(f"NEGATIVE PROMPT:\n{negative_prompt}\n\n")
            if config:
                f.write(f"CONFIGURAÇÕES:\n{json.dumps(config, indent=2, ensure_ascii=False)}\n")
            f.write(f"{'='*60}\n\n")
    
    def log_config(self, config: Dict):
        """Registra configurações da etapa atual"""
        if self.current_step:
            self.current_step.config = config
    
    def log_error(self, error: str):
        """Registra erro na etapa atual"""
        if self.current_step:
            self.current_step.add_error(error)
        print(f"[GenerationLogger] ERRO: {error}")
    
    def log_warning(self, warning: str):
        """Registra warning na etapa atual"""
        if self.current_step:
            self.current_step.add_warning(warning)
        print(f"[GenerationLogger] AVISO: {warning}")
    
    def finish_step(self, output_info: Optional[Dict] = None):
        """Finaliza a etapa atual"""
        if self.current_step:
            self.current_step.output_info = output_info
            self.current_step.finish()
            self.steps.append(self.current_step)
            
            duration = self.current_step.end_time - self.current_step.start_time
            print(f"[GenerationLogger] Finalizado: {self.current_step.step_name} ({duration:.2f}s)")
            
            # Atualiza timeline
            self._update_timeline()
            # Salva log principal
            self._save_main_log()
            
            self.current_step = None
    
    def log_detection_info(self, page_num: int, detections: List[Dict]):
        """Registra informações de detecção"""
        detection_file = self.logs_dir / f"detections_page_{page_num:03d}.json"
        
        data = {
            "page_number": page_num,
            "timestamp": datetime.now().isoformat(),
            "total_detections": len(detections),
            "detections": detections
        }
        
        with open(detection_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def log_embedding_info(self, char_id: str, embedding_type: str, shape: tuple, 
                           source_images: List[str]):
        """Registra informações de embedding"""
        embedding_file = self.logs_dir / "embeddings_info.json"
        
        # Carrega existente ou cria novo
        if embedding_file.exists():
            with open(embedding_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"embeddings": []}
        
        data["embeddings"].append({
            "char_id": char_id,
            "type": embedding_type,
            "shape": list(shape),
            "source_images": source_images,
            "timestamp": datetime.now().isoformat()
        })
        
        with open(embedding_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def log_pass1_summary(self, summary: Dict):
        """Registra resumo do Pass 1"""
        summary_file = self.logs_dir / "pass1_summary.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    # ========================================
    # Métodos para salvar imagens de debug
    # ========================================
    
    def save_input_image(self, page_num: int, image):
        """Salva imagem de entrada original"""
        try:
            from PIL import Image
            import numpy as np
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            output_path = self.input_dir / f"page_{page_num:03d}_input.png"
            image.save(output_path)
            print(f"[GenerationLogger] Input salvo: {output_path.name}")
            return str(output_path)
        except Exception as e:
            print(f"[GenerationLogger] Erro ao salvar input: {e}")
            return None
    
    def save_canny_image(self, page_num: int, canny_array, tile_idx: int = None):
        """Salva imagem canny (controle de linhas)"""
        try:
            from PIL import Image
            import numpy as np
            
            # Canny pode ser array numpy ou já uma imagem
            if isinstance(canny_array, np.ndarray):
                # Normaliza para 0-255 se necessário
                if canny_array.max() <= 1.0:
                    canny_array = (canny_array * 255).astype(np.uint8)
                image = Image.fromarray(canny_array)
            else:
                image = canny_array
            
            suffix = f"_tile{tile_idx}" if tile_idx is not None else ""
            output_path = self.canny_dir / f"page_{page_num:03d}{suffix}_canny.png"
            image.save(output_path)
            print(f"[GenerationLogger] Canny salvo: {output_path.name}")
            return str(output_path)
        except Exception as e:
            print(f"[GenerationLogger] Erro ao salvar canny: {e}")
            return None
    
    def save_crop_image(self, page_num: int, char_id: str, crop_image, crop_type: str = "body"):
        """Salva crop de personagem"""
        try:
            from PIL import Image
            import numpy as np
            
            if isinstance(crop_image, np.ndarray):
                crop_image = Image.fromarray(crop_image)
            
            output_path = self.crops_dir / f"page_{page_num:03d}_{char_id}_{crop_type}.png"
            crop_image.save(output_path)
            print(f"[GenerationLogger] Crop salvo: {output_path.name}")
            return str(output_path)
        except Exception as e:
            print(f"[GenerationLogger] Erro ao salvar crop: {e}")
            return None
    
    def save_detection_visualization(self, page_num: int, image, detections: List[Dict]):
        """Salva imagem com bboxes de detecção desenhadas"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Cria cópia para desenhar
            vis_image = image.copy()
            draw = ImageDraw.Draw(vis_image)
            
            # Cores por classe
            colors = {
                0: (0, 255, 0),   # body - verde
                1: (255, 0, 0),   # face - vermelho
                2: (0, 0, 255),   # frame - azul
                3: (255, 255, 0)  # text - amarelo
            }
            
            for det in detections:
                bbox = det.get('bbox', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    class_id = det.get('class_id', 0)
                    color = colors.get(class_id, (128, 128, 128))
                    
                    # Desenha retângulo
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Adiciona label
                    label = det.get('class_name', f'class_{class_id}')
                    char_id = det.get('char_id', '')
                    conf = det.get('confidence', 0)
                    text = f"{label} {conf:.2f}"
                    if char_id:
                        text += f" [{char_id[-6:]}]"
                    
                    draw.text((x1, y1 - 15), text, fill=color)
            
            output_path = self.detections_dir / f"page_{page_num:03d}_detections.png"
            vis_image.save(output_path)
            print(f"[GenerationLogger] Detections salvo: {output_path.name}")
            return str(output_path)
        except Exception as e:
            print(f"[GenerationLogger] Erro ao salvar detection viz: {e}")
            return None
    
    # ========================================
    # ADR 004: Máscara debug images
    # ========================================
    
    def save_mask_image(self, page_num: int, char_id: str, mask_array, mask_type: str = "sam"):
        """Salva máscara individual de um personagem como imagem grayscale.
        
        Args:
            page_num: Número da página
            char_id: ID do personagem
            mask_array: Array numpy (H, W) — uint8 (0-255) ou float32 (0.0-1.0)
            mask_type: Tipo da máscara ('sam', 'processed', 'bbox_fallback')
        """
        try:
            from PIL import Image
            import numpy as np
            
            mask = mask_array
            if mask.dtype == np.float32 or mask.dtype == np.float64:
                mask = (mask * 255).clip(0, 255).astype(np.uint8)
            
            image = Image.fromarray(mask, mode='L')
            output_path = self.masks_dir / f"page_{page_num:03d}_{char_id}_{mask_type}.png"
            image.save(output_path)
            print(f"[GenerationLogger] Mask salva: {output_path.name}")
            return str(output_path)
        except Exception as e:
            print(f"[GenerationLogger] Erro ao salvar mask: {e}")
            return None
    
    def save_mask_composite(self, page_num: int, original_image, masks: dict,
                            depth_order: list = None, alpha: float = 0.45):
        """Salva visualização composta: original + todas as máscaras coloridas sobrepostas.
        
        Cada personagem recebe uma cor distinta. Útil para verificar se os recortes
        SAM estão corretos e se o z-ordering resolve oclusões.
        
        Args:
            page_num: Número da página
            original_image: Imagem PIL ou numpy (H, W, 3)
            masks: Dict char_id -> np.ndarray (H, W) float32 0-1 ou uint8 0-255
            depth_order: Lista de char_ids (frente→fundo), se disponível
            alpha: Transparência do overlay (0 = invisível, 1 = opaco)
        """
        try:
            from PIL import Image
            import numpy as np
            
            if isinstance(original_image, np.ndarray):
                base = original_image.copy()
                if base.ndim == 2:  # grayscale
                    base = np.stack([base]*3, axis=-1)
            else:
                base = np.array(original_image.convert('RGB'))
            
            h, w = base.shape[:2]
            overlay = base.astype(np.float32).copy()
            
            # Paleta de cores distintas para até 10 personagens
            palette = [
                (0, 200, 255),    # ciano
                (255, 80, 80),    # vermelho
                (80, 255, 80),    # verde
                (255, 200, 0),    # amarelo
                (200, 80, 255),   # roxo
                (255, 128, 0),    # laranja
                (0, 255, 200),    # teal
                (255, 0, 200),    # magenta
                (128, 200, 0),    # lima
                (0, 128, 255),    # azul
            ]
            
            # Ordem de renderização: fundo→frente para ver a oclusão
            render_order = list(masks.keys())
            if depth_order:
                render_order = [cid for cid in reversed(depth_order) if cid in masks]
                # Adiciona qualquer char_id extra que não estava no depth_order
                for cid in masks:
                    if cid not in render_order and cid != 'background':
                        render_order.append(cid)
            else:
                render_order = [cid for cid in render_order if cid != 'background']
            
            legend_lines = []
            for idx, char_id in enumerate(render_order):
                mask = masks[char_id]
                color = palette[idx % len(palette)]
                
                # Normaliza máscara para float 0-1
                if mask.dtype == np.uint8:
                    mask_f = mask.astype(np.float32) / 255.0
                else:
                    mask_f = mask.astype(np.float32)
                
                # Redimensiona máscara se necessário
                if mask_f.shape[:2] != (h, w):
                    import cv2
                    mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Aplica overlay colorido
                for c in range(3):
                    overlay[:, :, c] = np.where(
                        mask_f > 0.05,
                        overlay[:, :, c] * (1 - alpha) + color[c] * alpha * mask_f,
                        overlay[:, :, c]
                    )
                
                depth_label = f"z={render_order.index(char_id)}" if depth_order else ""
                legend_lines.append(f"{char_id[-8:]}: rgb{color} {depth_label}")
            
            result = np.clip(overlay, 0, 255).astype(np.uint8)
            result_img = Image.fromarray(result)
            
            output_path = self.masks_dir / f"page_{page_num:03d}_mask_composite.png"
            result_img.save(output_path)
            
            # Salva legenda em txt
            legend_path = self.masks_dir / f"page_{page_num:03d}_mask_legend.txt"
            with open(legend_path, 'w', encoding='utf-8') as f:
                f.write(f"Máscara composite — página {page_num}\n")
                f.write(f"Render order (fundo→frente): {' → '.join(cid[-8:] for cid in render_order)}\n")
                for line in legend_lines:
                    f.write(f"  {line}\n")
            
            print(f"[GenerationLogger] Mask composite salvo: {output_path.name} ({len(render_order)} personagens)")
            return str(output_path)
        except Exception as e:
            print(f"[GenerationLogger] Erro ao salvar mask composite: {e}")
            return None
    
    def save_output_image(self, page_num: int, image, suffix: str = "output"):
        """Salva imagem de saída (resultado da geração)"""
        try:
            from PIL import Image
            import numpy as np
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            output_path = self.images_dir / f"page_{page_num:03d}_{suffix}.png"
            image.save(output_path)
            print(f"[GenerationLogger] Output salvo: {output_path.name}")
            return str(output_path)
        except Exception as e:
            print(f"[GenerationLogger] Erro ao salvar output: {e}")
            return None
    
    def finalize(self, final_stats: Optional[Dict] = None):
        """Finaliza o logging da sessão"""
        # Finaliza etapa atual se existir
        if self.current_step and not self.current_step.end_time:
            self.current_step.finish()
            self.steps.append(self.current_step)
        
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_duration_seconds"] = round(time.time() - self.session_start, 2)
        self.metadata["final_stats"] = final_stats or {}
        
        self._save_main_log()
        self._update_timeline()
        
        print(f"[GenerationLogger] Sessão finalizada. Logs salvos em: {self.logs_dir}")
    
    def _save_main_log(self):
        """Salva o log principal em JSON"""
        data = {
            "metadata": self.metadata,
            "global_config": self.global_config,
            "steps": [step.to_dict() for step in self.steps],
            "current_step": self.current_step.to_dict() if self.current_step else None
        }
        
        with open(self.main_log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def _update_timeline(self):
        """Atualiza arquivo de timeline em formato legível"""
        with open(self.timeline_file, 'w', encoding='utf-8') as f:
            f.write(f"MANGAAUTOCOLOR PRO - TIMELINE DE GERAÇÃO\n")
            f.write(f"Capítulo: {self.chapter_id}\n")
            f.write(f"Início: {self.metadata['start_time']}\n")
            f.write(f"{'='*70}\n\n")
            
            for i, step in enumerate(self.steps, 1):
                duration = step.end_time - step.start_time if step.end_time else 0
                f.write(f"{i}. {step.step_name}\n")
                f.write(f"   Duração: {duration:.2f}s\n")
                
                if step.prompt:
                    prompt_preview = step.prompt[:80] + "..." if len(step.prompt) > 80 else step.prompt
                    f.write(f"   Prompt: {prompt_preview}\n")
                
                if step.errors:
                    f.write(f"   ⚠️  Erros: {len(step.errors)}\n")
                if step.warnings:
                    f.write(f"   ⚡ Warnings: {len(step.warnings)}\n")
                
                f.write(f"\n")
            
            if self.current_step and not self.current_step.end_time:
                f.write(f">>> EM ANDAMENTO: {self.current_step.step_name}\n")


class NullLogger:
    """Logger nulo para quando logging está desabilitado"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def set_global_config(self, config):
        pass
    
    def start_step(self, step_name, input_info=None):
        class DummyStep:
            def add_error(self, e): pass
            def add_warning(self, w): pass
            def finish(self): pass
        return DummyStep()
    
    def log_prompts(self, prompt, negative_prompt="", config=None):
        pass
    
    def log_config(self, config):
        pass
    
    def log_error(self, error):
        pass
    
    def log_warning(self, warning):
        pass
    
    def finish_step(self, output_info=None):
        pass
    
    def log_detection_info(self, page_num, detections):
        pass
    
    def log_embedding_info(self, char_id, embedding_type, shape, source_images):
        pass
    
    def log_pass1_summary(self, summary):
        pass
    
    def finalize(self, final_stats=None):
        pass
    
    # Stubs para métodos de imagem (não salvam nada)
    def save_input_image(self, page_num, image):
        return None
    
    def save_canny_image(self, page_num, canny_array, tile_idx=None):
        return None
    
    def save_crop_image(self, page_num, char_id, crop_image, crop_type="body"):
        return None
    
    def save_detection_visualization(self, page_num, image, detections):
        return None
    
    def save_mask_image(self, page_num, char_id, mask_array, mask_type="sam"):
        return None
    
    def save_mask_composite(self, page_num, original_image, masks, depth_order=None, alpha=0.45):
        return None
    
    def save_output_image(self, page_num, image, suffix="output"):
        return None

