"""
MangaAutoColor Pro - Interface Gradio v2.0
Interface Two-Pass com suporte a Tile-Aware Processing e Regional IP-Adapter
"""

import gradio as gr
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import tempfile
import os

from core.pipeline import (
    MangaColorizationPipeline, 
    GenerationOptions,
    ChapterAnalysis,
    AnalysisError,
    GenerationError
)
from config.settings import (
    DEVICE, DTYPE, STYLE_PRESETS, TIPOS_DE_CENA,
    TILE_SIZE, TILE_OVERLAP, MAX_REF_PER_TILE,
    IP_ADAPTER_SCALE_DEFAULT, IP_ADAPTER_END_STEP,
    get_device_properties, get_optimal_batch_size
)


class MangaColorizerUI:
    """
    Interface Gradio para o MangaAutoColor Pro.
    
    Implementa workflow Two-Pass com:
    - Pass 1: Análise com detecção YOLO + cache de embeddings
    - Pass 2: Geração Tile-Aware com Regional IP-Adapter
    """
    
    def __init__(self):
        self.pipeline: Optional[MangaColorizationPipeline] = None
        self.analysis: Optional[ChapterAnalysis] = None
        self.page_paths: List[str] = []
        self.output_dir = tempfile.mkdtemp(prefix="manga_output_")
        
        # Cache de informações por página
        self._page_info: Dict[int, Dict] = {}
        
    def _init_pipeline(self) -> MangaColorizationPipeline:
        """Inicializa pipeline com lazy loading"""
        if self.pipeline is None:
            self.pipeline = MangaColorizationPipeline(
                device=DEVICE,
                dtype=DTYPE,
                cache_dir="./data/cache",
                enable_xformers=True,
                enable_cpu_offload=True
            )
        return self.pipeline
    
    def analyze_chapter(
        self, 
        files: List[str],
        progress: gr.Progress = gr.Progress()
    ) -> Tuple[str, str, str, gr.update]:
        """
        PASSO 1: Análise completa do capítulo com Tile-Aware preprocessing.
        
        Detecta personagens, extrai embeddings CLIP/ArcFace, calcula tiles
        e pré-computa máscaras gaussianas para cada personagem.
        
        Args:
            files: Lista de arquivos de imagem
            progress: Callback de progresso do Gradio
            
        Returns:
            Tuple com (status, resumo, detalhes técnicos, update para galeria)
        """
        if not files:
            return (
                "❌ Nenhuma imagem selecionada",
                "", 
                "",
                gr.update(visible=False)
            )
        
        try:
            pipeline = self._init_pipeline()
            
            # Ordena arquivos por nome
            self.page_paths = sorted([f.name if hasattr(f, 'name') else f for f in files])
            
            # Configura callback de progresso
            def on_progress(page_num: int, stage: str, pct: float):
                stage_names = {
                    "analyzing": "🔍 Analisando",
                    "detection": "👤 Detectando personagens",
                    "embedding": "🧠 Extraindo embeddings",
                    "tiling": "🧩 Calculando tiles",
                    "consolidating": "📊 Consolidando",
                    "complete": "✅ Completo"
                }
                stage_name = stage_names.get(stage, stage)
                progress(pct / 100, desc=f"Página {page_num + 1}: {stage_name}")
            
            pipeline.set_progress_callback(on_progress)
            
            # Executa análise
            progress(0, desc="Iniciando análise...")
            self.analysis = pipeline.process_chapter(self.page_paths)
            
            # Formata resultados
            status = f"✅ Análise completa: {self.analysis.num_pages} páginas, {self.analysis.num_characters} personagens"
            
            resumo = self._format_analysis_summary(self.analysis)
            detalhes_tecnicos = self._format_technical_details(self.analysis)
            
            return status, resumo, detalhes_tecnicos, gr.update(visible=True)
            
        except AnalysisError as e:
            return f"❌ Erro na análise: {e}", "", "", gr.update(visible=False)
        except Exception as e:
            return f"❌ Erro inesperado: {e}", "", "", gr.update(visible=False)
    
    def _format_analysis_summary(self, analysis: ChapterAnalysis) -> str:
        """Formata resumo da análise para exibição"""
        lines = [
            f"## 📊 Resumo da Análise",
            f"",
            f"**Páginas analisadas:** {analysis.num_pages}",
            f"**Personagens detectados:** {analysis.num_characters}",
            f"**Tempo estimado de geração:** {analysis.estimated_generation_time:.0f}s",
            f"",
            f"### 👤 Personagens Principais",
        ]
        
        # Ordena por número de aparições
        sorted_chars = sorted(
            analysis.characters,
            key=lambda x: x.get('appearances', 1),
            reverse=True
        )
        
        for i, char in enumerate(sorted_chars[:10]):
            appearances = char.get('appearances', 1)
            method = char.get('embedding_method', 'N/A')
            lines.append(f"- **Personagem {i+1}**: {appearances} páarições (método: {method})")
        
        if len(sorted_chars) > 10:
            lines.append(f"- ... e mais {len(sorted_chars) - 10} personagens")
        
        lines.extend([
            f"",
            f"### 🎬 Contexto Narrativo Detectado",
        ])
        
        for scene_type, pages in analysis.scene_breakdown.items():
            if pages:
                page_list = ', '.join(map(str, pages[:5]))
                if len(pages) > 5:
                    page_list += f", ... ({len(pages)} total)"
                lines.append(f"- **{scene_type}**: páginas {page_list}")
        
        return "\n".join(lines)
    
    def _format_technical_details(self, analysis: ChapterAnalysis) -> str:
        """Formata detalhes técnicos da análise"""
        device_info = get_device_properties()
        
        lines = [
            f"## ⚙️ Detalhes Técnicos",
            f"",
            f"### Hardware",
            f"- **Dispositivo:** {device_info.get('name', 'CPU')}",
            f"- **Memória VRAM:** {device_info.get('total_memory_gb', 0):.1f} GB",
            f"- **Dtype:** {DTYPE}",
            f"- **Batch size ótimo:** {get_optimal_batch_size()}",
            f"",
            f"### Configurações Tile-Aware",
            f"- **Tamanho do tile:** {TILE_SIZE}x{TILE_SIZE}px",
            f"- **Overlap:** {TILE_OVERLAP}px",
            f"- **Máx. personagens/tile:** {MAX_REF_PER_TILE}",
            f"",
            f"### IP-Adapter (Regional Identity)",
            f"- **Scale padrão:** {IP_ADAPTER_SCALE_DEFAULT}",
            f"- **End step (temporal decay):** {IP_ADAPTER_END_STEP:.0%}",
            f"- **Background isolation:** Ativado",
            f"",
            f"### Cache",
            f"- **Diretório:** `./data/chapters/<chapter_id>/`",
            f"- **Formato embeddings:** PyTorch .pt (FP16)",
            f"- **Indexação:** FAISS (Inner Product)",
        ]
        
        return "\n".join(lines)
    
    def generate_page(
        self,
        page_num: int,
        style_preset: str,
        quality_mode: str,
        ip_scale: float,
        preserve_text: bool,
        apply_narrative: bool,
        seed: int,
        progress: gr.Progress = gr.Progress()
    ) -> Tuple[str, Optional[str]]:
        """
        PASSO 2: Geração de uma página específica com Tile-Aware Processing.
        
        Processa a página em tiles 1024x1024, carregando apenas os embeddings
        dos personagens presentes em cada tile (Top-K por prominence).
        
        Args:
            page_num: Número da página (1-based para UI)
            style_preset: Preset de estilo
            quality_mode: Modo de qualidade
            ip_scale: Escala do IP-Adapter
            preserve_text: Preservar texto original
            apply_narrative: Aplicar transformações narrativas
            seed: Seed para reprodutibilidade
            progress: Callback de progresso
            
        Returns:
            Tuple com (status, caminho da imagem gerada)
        """
        if self.analysis is None:
            return "❌ Execute a análise primeiro (Passo 1)", None
        
        if page_num < 1 or page_num > self.analysis.num_pages:
            return f"❌ Página inválida. Escolha entre 1 e {self.analysis.num_pages}", None
        
        try:
            pipeline = self._init_pipeline()
            
            # Configura opções
            options = GenerationOptions(
                style_preset=style_preset,
                quality_mode=quality_mode,
                preserve_original_text=preserve_text,
                apply_narrative_transforms=apply_narrative,
                seed=seed if seed >= 0 else None
            )
            
            # Callback de progresso com informações de tile
            def on_progress(pn: int, stage: str, pct: float):
                stage_names = {
                    "loading": "📂 Carregando modelos",
                    "tiles": "🧩 Processando tiles",
                    "blending": "🎨 Blending final",
                    "complete": "✅ Completo"
                }
                stage_name = stage_names.get(stage, stage)
                progress(pct / 100, desc=f"{stage_name}")
            
            pipeline.set_progress_callback(on_progress)
            
            # Gera página (0-based internamente)
            progress(0, desc="Iniciando geração Tile-Aware...")
            result = pipeline.generate_page(page_num - 1, options)
            
            # Salva resultado
            output_path = Path(self.output_dir) / f"page_{page_num:03d}.png"
            result.save(output_path, quality=95)
            
            return f"✅ Página {page_num} gerada com sucesso!", str(output_path)
            
        except GenerationError as e:
            return f"❌ Erro na geração: {e}", None
        except Exception as e:
            return f"❌ Erro inesperado: {e}", None
    
    def generate_all_pages(
        self,
        style_preset: str,
        quality_mode: str,
        ip_scale: float,
        preserve_text: bool,
        apply_narrative: bool,
        seed: int,
        progress: gr.Progress = gr.Progress()
    ) -> Tuple[str, List[str]]:
        """
        Gera todas as páginas em sequência.
        
        Returns:
            Tuple com (status, lista de caminhos das imagens)
        """
        if self.analysis is None:
            return "❌ Execute a análise primeiro (Passo 1)", []
        
        try:
            pipeline = self._init_pipeline()
            
            options = GenerationOptions(
                style_preset=style_preset,
                quality_mode=quality_mode,
                preserve_original_text=preserve_text,
                apply_narrative_transforms=apply_narrative,
                seed=seed if seed >= 0 else None
            )
            
            output_paths = []
            
            for i in range(self.analysis.num_pages):
                progress(
                    i / self.analysis.num_pages, 
                    desc=f"Gerando página {i + 1}/{self.analysis.num_pages}"
                )
                
                result = pipeline.generate_page(i, options)
                
                output_path = Path(self.output_dir) / f"page_{i+1:03d}.png"
                result.save(output_path, quality=95)
                output_paths.append(str(output_path))
            
            progress(1.0, desc="Completo!")
            
            return f"✅ {len(output_paths)} páginas geradas com Tile-Aware Processing!", output_paths
            
        except Exception as e:
            return f"❌ Erro: {e}", []
    
    def set_scene_context(
        self,
        start_page: int,
        end_page: int,
        context_type: str
    ) -> str:
        """
        Define contexto narrativo para um range de páginas.
        
        Aplica transformações de estilo (flashback = desaturado,
        dream = ethereal, etc) nas páginas especificadas.
        
        Args:
            start_page: Página inicial (1-based)
            end_page: Página final (1-based)
            context_type: Tipo de cena
            
        Returns:
            Mensagem de status
        """
        if self.analysis is None:
            return "❌ Execute a análise primeiro"
        
        if start_page < 1 or end_page > self.analysis.num_pages:
            return f"❌ Range inválido. Páginas: 1-{self.analysis.num_pages}"
        
        if start_page > end_page:
            return "❌ Página inicial deve ser menor ou igual à final"
        
        try:
            pipeline = self._init_pipeline()
            pipeline.set_scene_context(
                page_range=(start_page - 1, end_page - 1),  # 0-based
                context_type=context_type
            )
            
            # Descrição do efeito
            effects = {
                "present": "cores normais",
                "flashback": "cores desaturadas/nostálgicas",
                "dream": "cores etéreas/brilhantes",
                "nightmare": "cores escuras/distorcidas",
                "hell": "tons de vermelho/preto",
                "memory": "cores suaves/lembrança"
            }
            effect = effects.get(context_type, context_type)
            
            return f"✅ Contexto '{context_type}' ({effect}) definido para páginas {start_page}-{end_page}"
            
        except Exception as e:
            return f"❌ Erro: {e}"


def create_ui() -> gr.Blocks:
    """
    Cria a interface Gradio para o MangaAutoColor Pro.
    
    Returns:
        gr.Blocks: Aplicação Gradio
    """
    ui = MangaColorizerUI()
    
    # Obtém info do dispositivo
    device_info = get_device_properties()
    device_str = f"{device_info.get('name', 'CPU')}"
    if device_info.get('total_memory_gb', 0) > 0:
        device_str += f" ({device_info['total_memory_gb']:.1f} GB)"
    
    with gr.Blocks(
        title="MangaAutoColor Pro v2.0",
        theme=gr.themes.Soft(),
        css="""
            .main-title { text-align: center; margin-bottom: 1em; }
            .step-header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
            .tech-info { font-family: monospace; font-size: 0.9em; }
            .highlight { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }
        """
    ) as app:
        
        gr.Markdown(
            """
            # 🎨 MangaAutoColor Pro v2.0
            ### Sistema Two-Pass com Tile-Aware Processing e Regional IP-Adapter
            """,
            elem_classes=["main-title"]
        )
        
        gr.Markdown(
            f"**🖥️ Dispositivo:** {device_str} | "
            f"**⚡ Engine:** SDXL-Lightning 4-Step | "
            f"**🧩 Tiles:** {TILE_SIZE}px",
            elem_classes=["tech-info"]
        )
        
        with gr.Tabs():
            # ==================== TAB 1: ANÁLISE ====================
            with gr.TabItem("📊 Passo 1: Análise", id="analysis"):
                gr.Markdown(
                    """
                    <div class="step-header">
                    <b>Passo 1: Análise Completa do Capítulo</b><br>
                    Detecta personagens (YOLOv8), extrai embeddings (CLIP/ArcFace), 
                    calcula tiles e pré-computa máscaras gaussianas.
                    </div>
                    """,
                    sanitize_html=False
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        input_files = gr.File(
                            label="📁 Páginas do Capítulo",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath"
                        )
                        
                        gr.Markdown("""
                        **Formatos suportados:** PNG, JPG, WEBP, BMP  
                        **Ordem:** Ordenado automaticamente por nome de arquivo
                        """)
                        
                        analyze_btn = gr.Button(
                            "🔍 Analisar Capítulo",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=3):
                        analysis_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        
                        with gr.Tabs():
                            with gr.TabItem("Resumo"):
                                analysis_summary = gr.Markdown(
                                    label="Resumo da Análise",
                                    visible=True
                                )
                            
                            with gr.TabItem("Detalhes Técnicos"):
                                analysis_details = gr.Markdown(
                                    label="Detalhes Técnicos",
                                    visible=True,
                                    elem_classes=["tech-info"]
                                )
                
                # Seção de contexto narrativo
                with gr.Accordion("🎬 Definir Contexto Narrativo (Opcional)", open=False):
                    gr.Markdown(
                        "Marque páginas como flashback, sonho, etc. para ajustar a colorização. "
                        "Cada tipo de cena aplica um estilo visual específico automaticamente."
                    )
                    
                    with gr.Row():
                        context_start = gr.Number(
                            label="Página Inicial",
                            value=1,
                            minimum=1,
                            precision=0
                        )
                        context_end = gr.Number(
                            label="Página Final",
                            value=1,
                            minimum=1,
                            precision=0
                        )
                        context_type = gr.Dropdown(
                            label="Tipo de Cena",
                            choices=TIPOS_DE_CENA,
                            value="present"
                        )
                        context_btn = gr.Button("Aplicar Contexto", variant="secondary")
                    
                    context_status = gr.Textbox(
                        label="Status do Contexto",
                        interactive=False
                    )
            
            # ==================== TAB 2: GERAÇÃO ====================
            with gr.TabItem("🖌️ Passo 2: Geração", id="generation"):
                gr.Markdown(
                    """
                    <div class="step-header">
                    <b>Passo 2: Geração Tile-Aware</b><br>
                    Processa em tiles <span class="highlight">1024×1024</span> com 
                    <span class="highlight">Regional IP-Adapter</span>. 
                    Máximo <span class="highlight">2 personagens/tile</span> com 
                    <span class="highlight">temporal decay</span>.
                    </div>
                    """,
                    sanitize_html=False
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configurações")
                        
                        style_preset = gr.Dropdown(
                            label="Estilo Visual",
                            choices=list(STYLE_PRESETS.keys()),
                            value="default"
                        )
                        
                        quality_mode = gr.Radio(
                            label="Modo de Qualidade",
                            choices=["fast", "balanced", "high"],
                            value="balanced"
                        )
                        
                        ip_scale_slider = gr.Slider(
                            label="IP-Adapter Scale (Força da Identidade)",
                            minimum=0.0,
                            maximum=1.0,
                            value=IP_ADAPTER_SCALE_DEFAULT,
                            step=0.05
                        )
                        
                        gr.Markdown(
                            f"""
                            ℹ️ **Dica:** Valores maiores preservam mais a identidade do personagem.
                            Padrão: {IP_ADAPTER_SCALE_DEFAULT}
                            """
                        )
                        
                        preserve_text = gr.Checkbox(
                            label="📝 Preservar texto original (balões)",
                            value=True
                        )
                        
                        apply_narrative = gr.Checkbox(
                            label="🎭 Aplicar transformações narrativas",
                            value=True
                        )
                        
                        seed = gr.Number(
                            label="🎲 Seed (-1 para aleatório)",
                            value=-1,
                            precision=0
                        )
                        
                        gr.Markdown("---")
                        
                        # Geração individual
                        gr.Markdown("### 📄 Gerar Página Individual")
                        page_num = gr.Number(
                            label="Número da Página",
                            value=1,
                            minimum=1,
                            precision=0
                        )
                        generate_one_btn = gr.Button(
                            "🎨 Gerar Página",
                            variant="primary"
                        )
                        
                        gr.Markdown("---")
                        
                        # Geração em lote
                        gr.Markdown("### 📚 Gerar Capítulo Completo")
                        generate_all_btn = gr.Button(
                            "🚀 Gerar Todas as Páginas",
                            variant="secondary"
                        )
                    
                    with gr.Column(scale=2):
                        generation_status = gr.Textbox(
                            label="Status da Geração",
                            interactive=False
                        )
                        
                        output_image = gr.Image(
                            label="Resultado da Geração",
                            type="filepath"
                        )
                        
                        output_gallery = gr.Gallery(
                            label="Galeria de Páginas Geradas",
                            columns=4,
                            height="auto",
                            visible=True
                        )
            
            # ==================== TAB 3: ARQUITETURA ====================
            with gr.TabItem("🏗️ Arquitetura", id="architecture"):
                gr.Markdown(
                    """
                    ## 🏗️ Arquitetura Two-Pass com Differential Diffusion
                    
                    ### Passo 1: Análise (CPU/IO Bound)
                    
                    ```
                    ┌─────────────────────────────────────────────────────────┐
                    │  1. Detecção (YOLOv8)                                    │
                    │     └── keremberke/yolov8m-manga-10k                    │
                    │                                                         │
                    │  2. Identidade (Hybrid Encoder)                          │
                    │     ├── ArcFace (InsightFace) - quando disponível       │
                    │     └── CLIP Image Encoder - IP-Adapter reference       │
                    │                                                         │
                    │  3. Paleta (CIELAB + K-means)                            │
                    │     └── Regiões: hair, skin, eyes, clothes              │
                    │                                                         │
                    │  4. Tile Pre-computation                                 │
                    │     ├── Divide página em 1024×1024 tiles                │
                    │     ├── Calcula Top-K personagens por tile              │
                    │     └── Gera máscaras Gaussianas                        │
                    │                                                         │
                    │  📦 Persistência: FAISS + Parquet + .pt tensors         │
                    └─────────────────────────────────────────────────────────┘
                    ```
                    
                    ### Passo 2: Geração (VRAM Bound)
                    
                    ```
                    ┌─────────────────────────────────────────────────────────┐
                    │  Engine: SDXL-Lightning (4 steps)                        │
                    │  ControlNet: Canny edges                                │
                    │                                                         │
                    │  Para cada Tile:                                         │
                    │  1. Carrega apenas embeddings dos Top-K chars           │
                    │  2. Aplica Regional IP-Adapter com máscaras Gaussianas  │
                    │  3. Steps 0-60%: IP-Adapter ativo (identidade)          │
                    │  4. Steps 60-100%: Apenas SDXL + ControlNet (refino)    │
                    │  5. Descarrega embeddings da VRAM                       │
                    │                                                         │
                    │  Blending Final:                                         │
                    │  ├── Multi-band blending entre tiles                    │
                    │  ├── Chroma isolation para consistência de cor          │
                    │  └── Poisson blending (opcional) para harmonização      │
                    └─────────────────────────────────────────────────────────┘
                    ```
                    
                    ### Princípios Differential Diffusion
                    
                    1. **Máscaras Progressivas (Gradient Masks)**
                       - Centro do personagem: força 1.0
                       - Bordas: decaimento Gaussiano para 0.0
                       - Background: IP-Adapter scale = 0.0
                    
                    2. **Temporal Decay (IP-Adapter Step Control)**
                       - Steps 0-2: Injeta identidade (cores globais)
                       - Steps 3-4: Refinamento com ControlNet (estrutura)
                    
                    3. **Tile-Aware Locality**
                       - Cada tile carrega apenas personagens presentes
                       - Top-K = 2 personagens por tile (economia de VRAM)
                    """
                )
            
            # ==================== TAB 4: SOBRE ====================
            with gr.TabItem("ℹ️ Sobre", id="about"):
                gr.Markdown(
                    f"""
                    ## 🎨 MangaAutoColor Pro v2.0
                    
                    Sistema enterprise de colorização automática de mangá com arquitetura
                    **Two-Pass** e **Differential Diffusion**.
                    
                    ### ✨ Funcionalidades Principais
                    
                    | Feature | Descrição |
                    |---------|-----------|
                    | **🔄 Two-Pass** | Análise completa antes da geração (navegação não-linear) |
                    | **🧩 Tile-Aware** | Processamento em tiles 1024×1024 para alta resolução |
                    | **👤 Regional IP-Adapter** | Identidade por região com máscaras Gaussianas |
                    | **🎭 Contexto Narrativo** | Flashbacks, sonhos, cenas especiais |
                    | **💾 Cache Persistente** | Embeddings salvos em .pt (imutabilidade) |
                    | **⚡ SDXL-Lightning** | Geração em 4 steps com qualidade profissional |
                    
                    ### 📊 Performance Esperada
                    
                    | Hardware | Análise/página | Geração/página | VRAM |
                    |----------|----------------|----------------|------|
                    | RTX 3060 | ~2s | ~8s | 10-12GB |
                    | RTX 4090 | ~0.8s | ~3s | 20-24GB |
                    
                    ### 🔧 Tecnologias
                    
                    - **Detecção:** YOLOv8 (keremberke/yolov8m-manga-10k)
                    - **Base:** SDXL-Lightning (ByteDance)
                    - **Identidade:** CLIP + ArcFace (InsightFace)
                    - **Control:** ControlNet Canny SDXL
                    - **Database:** FAISS + Parquet
                    
                    ### 📚 Documentação
                    
                    - [Arquitetura](docs/ARCHITECTURE.md)
                    - [API Reference](docs/API.md)
                    - [Setup Guide](docs/SETUP.md)
                    
                    ### ⚠️ Aviso
                    
                    Este projeto é para fins educacionais. Respeite os direitos autorais
                    dos criadores de mangá.
                    """
                )
        
        # ==================== EVENT HANDLERS ====================
        
        # Análise
        analyze_btn.click(
            fn=ui.analyze_chapter,
            inputs=[input_files],
            outputs=[analysis_status, analysis_summary, analysis_details, output_gallery]
        )
        
        # Contexto narrativo
        context_btn.click(
            fn=ui.set_scene_context,
            inputs=[context_start, context_end, context_type],
            outputs=[context_status]
        )
        
        # Geração individual
        generate_one_btn.click(
            fn=ui.generate_page,
            inputs=[
                page_num, style_preset, quality_mode, ip_scale_slider,
                preserve_text, apply_narrative, seed
            ],
            outputs=[generation_status, output_image]
        )
        
        # Geração em lote
        generate_all_btn.click(
            fn=ui.generate_all_pages,
            inputs=[
                style_preset, quality_mode, ip_scale_slider,
                preserve_text, apply_narrative, seed
            ],
            outputs=[generation_status, output_gallery]
        )
    
    return app


def launch_ui(share: bool = False, server_port: int = 7860):
    """
    Lança a interface Gradio.
    
    Args:
        share: Se True, cria link público
        server_port: Porta do servidor
    """
    app = create_ui()
    app.launch(
        share=share,
        server_port=server_port,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    launch_ui()
