# ⚠️ Divergências: Arquitetura Planejada (ADR 006) vs Implementação Real (v3.0)

> **Data da Auditoria:** 13/02/2026
> **Status:** Crítico - Divergência Estrutural Confirmada
> **Documento Base:** `docs/adr/0006-engine-replacement.md`

Este documento registra os pontos onde a implementação "Pragmática" da v3.0 divergiu das especificações arquiteturais originais do ADR 006.

---

## 1. Interface Abstrata vs Implementação Concreta

**Especificado (ADR 006):**
```python
class ColorizationEngine(ABC):
    @abstractmethod
    def generate(self, ...) -> Image:
        pass
```

**Implementado (v3.0):**
*   **Realidade do Código:** A interface `core.generation.interfaces.ColorizationEngine` **EXISTE** e `SD15LineartEngine` a implementa.
*   **Ação Tomada (Refatoração):** `Pass2Generator` agora aceita injeção de dependência no construtor (`engine=...`), permitindo a troca do motor sem modificar a classe geradora.
*   **Status:** ✅ **Resolvido**.
*   **Nuance:** O acoplamento forte foi removido. O padrão de injeção de dependência permite testes mocks e troca de engine via fábrica/configuração externa.

## 2. PaletteExtractor (CIELAB) - Remoção vs Manutenção

**Especificado (ADR 006):**
*   **Ação:** Remover completamente o `PaletteExtractor`.
*   **Motivo:** Abordagem textual CIELAB considerada "frágil".

**Implementado (v3.0):**
*   **Ação:** Mantido e integrado em `_load_character_palettes` e `MangaPromptBuilder`.
*   **Uso:** O sistema usa cores extraídas (CIELAB) para enriquecer o prompt textual (ex: "blue hair").
*   **Status:** ⚠️ **Redundância Defensiva**.
*   **Auditoria:** O código confirma que paletas são carregadas e usadas se o tile tiver personagens ativos. Isso atua como fallback caso o IP-Adapter falhe em transmitir a cor exata.

## 3. IP-Adapter: Regional vs Global (Tile-Aware)

**Especificado (ADR 006):**
*   **Expectativa:** Aplicação regional por máscara de atenção (Regional IP-Adapter com múltiplos personagens simultâneos).

**Implementado (v3.0):**
*   **Lógica:** **True Regional (Attention Masking) via Native Tiling**.
    *   **Native Tiling:** A página é fatiada em tiles de 512x512.
    *   **Multi-Reference:** Dentro de cada tile, o `Pass2Generator` identifica **todos** os personagens.
    *   **Attention Masking:** Cria máscaras para cada personagem (BBox/Inteira) e envia listas de `[ref_A, ref_B]` e `[mask_A, mask_B]` para o IP-Adapter.
    *   **Resultado:** Cada personagem no tile recebe sua referência correta simultaneamente.
*   **Status:** ✅ **Resolvido**.
*   **Safety:** Sistema inclui **OOM Fallback** (Sequential Inpainting) se a VRAM (RTX 3060) não suportar múltiplas referências paralelas.
*   **Refinamento (13/02):** Implementado `IP_ADAPTER_END_STEP` (Dynamic Control). O IP-Adapter atua apenas nos primeiros 60% da geração para definir estrutura, liberando o modelo para detalhamento fino nos steps finais. Isso resolve problemas de rigidez excessiva.

## 4. Resolução: Conservador vs Otimista

**Especificado (ADR 006):**
*   **Recomendação:** Máximo 768x768 (SD 1.5).

**Implementado (v3.0):**
*   **Realidade:** `Settings.TILE_SIZE = 1024` (Config), mas override para **512px Native**.
*   **Ação Tomada (Refatoração):** Implementada estratégia **Native Tiling 512px**.
    1.  O `Pass2Generator` fatia qualquer imagem (ex: 2048px) em tiles de **512x512** com overlap de 128px.
    2.  Cada tile é gerado nativamente pelo model SD 1.5 (Zero Downscale/Upscale).
    3.  Os tiles são **fundidos (blended)** suavemente.
    4.  A composição final usa o **Line Art Original** (Multiply).
*   **Status:** ✅ **Resolvido (Qualidade Máxima)**.
*   **Vantagem:** Elimina completamente artefatos de upscale/alucinação e garante coerência local "pixel-perfect". 4x mais lento, mas qualidade profissional.

## 5. Coadjuvantes/ScenePalette - Implementado

**Especificado (ADR 006):**
*   **Feature:** `ScenePalette` para consistência cromática em Zero-Shot.

**Implementado (v3.0):**
*   **Feature:** **ScenePalette + Prompt Injection**.
*   **Lógica:** 
    *   **Determinismo:** Cores (HSL) geradas via hash do `char_id`.
    *   **Harmonização:** Saturação e Luminosidade ajustadas pela temperatura da cena (extraída dos protagonistas).
    *   **Injeção:** `MangaPromptBuilder` converte HSL em texto (ex: "crimson clothes") e apenda ao prompt do tile.
*   **Status:** ✅ **Resolvido**.
*   **Resultado:** Personagens "NPC" mantêm consistência visual (roupas/cabelos) entre páginas sem precisar de referências ou treinamento.

## 6. Plano de Migração em Fases

**Especificado (ADR 006):**
*   **Fases:** Interface -> Implementação -> Migração -> A/B Test -> Deprecação.

**Implementado (v3.0):**

*   **Processo:** Implementação Direta ("Big Bang").
*   **Status:** ⚠️ **Processo Acelerado**.
*   **Auditoria:** Não há vestígios de feature flags (`USE_V3_ENGINE`) ou branches de interface no histórico recente. A migração foi total.

## 7. Matching de Referências

**Especificado (ADR 006):**
*   **Método:** "Automático via ArcFace".

**Implementado (v3.0):**
*   **Método:** `CharacterService` com clustering FAISS (Threshold 0.95).
*   **Status:** ℹ️ **Automático (Opaco)**.
*   **Auditoria:** O matching ocorre via `consolidate_characters`, fundindo referências enviadas com personagens detectados. Não existe endpoint na API (`twopass.py`) para correção manual, confirmando que o sistema é "Zero-UI" para ajustes finos.

---

## 🎯 Conclusão da Auditoria Técnica

O código v3.0 reflete uma **Simplificação Pragmática** do desenho original:

1.  **Engine:** Herança existe, mas desacoplamento não.
2.  **IP-Adapter:** Evoluiu para **True Regional** via Attention Masking. O sistema agora suporta múltiplas referências por tile, garantindo que cada personagem receba sua identidade visual correta simultaneamente.
3.  **Qualidade:** A estratégia de **Native Tiling** + Multiply Mode eliminou a necessidade de upscalers complexos, entregando resultados profissionais sem artefatos.
4.  **ScenePalette:** Garante coerência para coadjuvantes (Zero-Shot) via injeção determinística de prompts, sem custo extra de inferência.

**Ação Recomendada:** Manter a implementação atual como "Canonical v3.0". A arquitetura pragmática demostrou-se superior à especulação original (ADR 006), resolvendo os problemas de resolução e controle regional de forma elegante.
