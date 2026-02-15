"""
Script de verificacao ADR 005

Verifica se Point Correspondence e Temporal Consistency estao funcionando.
Execute apos uma analise de capitulo.
"""

import sys
from pathlib import Path

def check_adr005():
    print("=" * 70)
    print("VERIFICACAO ADR 005 - Point Correspondence & Temporal Consistency")
    print("=" * 70)
    
    # 1. Verificar configuracoes
    print("\n[1] Verificando configuracoes...")
    from config.settings import (
        PCTC_ENABLED, PCTC_POINT_ENABLED, PCTC_TEMPORAL_ENABLED
    )
    
    if PCTC_ENABLED and PCTC_POINT_ENABLED and PCTC_TEMPORAL_ENABLED:
        print("   [OK] ADR 005 esta ATIVADO nas configuracoes")
    else:
        print("   [AVISO] ADR 005 esta DESATIVADO em config/settings.py")
        print(f"           PCTC_ENABLED={PCTC_ENABLED}")
        print(f"           PCTC_POINT_ENABLED={PCTC_POINT_ENABLED}")
        print(f"           PCTC_TEMPORAL_ENABLED={PCTC_TEMPORAL_ENABLED}")
    
    # 2. Verificar caches existentes
    print("\n[2] Verificando caches de capitulos...")
    cache_dir = Path("chapter_cache")
    
    if not cache_dir.exists():
        print("   [ERRO] Pasta chapter_cache nao encontrada")
        return
    
    chapter_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    
    if not chapter_dirs:
        print("   [AVISO] Nenhum capitulo em cache")
        return
    
    print(f"   {len(chapter_dirs)} capitulo(s) encontrado(s)")
    
    # 3. Verificar dados ADR 005 em cada capitulo
    for ch_dir in chapter_dirs:
        print(f"\n   Capitulo: {ch_dir.name}")
        
        # Verificar arquivos de atencao
        attention_files = list(ch_dir.glob("*attention_mask*.npy"))
        if attention_files:
            print(f"      [OK] {len(attention_files)} attention mask(s)")
        else:
            print(f"      [INFO] Nenhum attention mask (executar Pass 1 novamente)")
        
        # Verificar arquivos temporais
        temporal_files = list(ch_dir.glob("*temporal*.npy"))
        if temporal_files:
            print(f"      [OK] {len(temporal_files)} arquivo(s) temporal")
        else:
            print(f"      [INFO] Nenhum dado temporal (executar Pass 1 novamente)")
        
        # Verificar pages.parquet
        pages_file = ch_dir / "pages.parquet"
        if pages_file.exists():
            import pandas as pd
            df = pd.read_parquet(pages_file)
            
            has_attention = 'attention_mask_paths' in df.columns
            has_temporal = 'temporal_data' in df.columns
            
            if has_attention and has_temporal:
                print(f"      [OK] Database tem colunas ADR 005")
            else:
                print(f"      [INFO] Database sem colunas ADR 005 (executar Pass 1 novamente)")
    
    # 4. Instrucoes
    print("\n" + "=" * 70)
    print("INSTRUCOES:")
    print("=" * 70)
    print("""
Se ADR 005 nao aparecer nos caches:

1. Exclua o cache antigo:
   rmdir /s chapter_cache\\<chapter_id>

2. Execute Pass 1 novamente pela API ou CLI

3. Verifique os logs por mensagens:
   - "Point Correspondence: X masks geradas"
   - "Temporal Consistency: continuous (SSIM=Y.YY)"
   - "ADR 005: X attention masks carregadas"

4. Execute este script novamente para confirmar
""")

if __name__ == "__main__":
    check_adr005()
