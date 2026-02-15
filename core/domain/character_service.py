from typing import List, Tuple, Optional, Set, TYPE_CHECKING
import json
import pandas as pd

if TYPE_CHECKING:
    from core.database.chapter_db import ChapterDatabase

class CharacterService:
    """
    Serviço de domínio para operações com personagens.
    Extrai lógica de negócio do ChapterDatabase.
    """
    
    def __init__(self, db: "ChapterDatabase"):
        self.db = db
        
    def consolidate_characters(self, threshold: float = 0.95):
        """
        Consolida personagens similares (merge).
        
        Usa FAISS para encontrar embeddings muito similares (>0.95)
        e mescla personagens que provavelmente são o mesmo personagem
        em diferentes páginas.
        """
        # Acesso direto a membros protegidos para refatoração inicial
        # Idealmente, métodos públicos seriam adicionados ao ChapterDatabase
        if self.db._characters_df is None:
            self.db._load_parquets()
            
        if self.db._characters_df is None or len(self.db._characters_df) <= 1:
            return
        
        print(f"[CharacterService] Consolidando {len(self.db._characters_df)} personagens...")
        
        # Agrupa por similaridade usando FAISS
        merged_count = 0
        chars_to_merge = []
        
        # Itera sobre uma cópia para evitar problemas de índice
        chars_list = self.db._characters_df.to_dict('records')
        
        for row in chars_list:
            char_id = row['char_id']
            
            # Carrega embedding
            emb = self.db.load_embedding(char_id, 'clip')
            if emb is None:
                continue
            
            # Busca similares (excluindo o próprio)
            similars = self.db.find_similar_characters(emb, top_k=3, threshold=0.90)
            similars = [(sid, score) for sid, score in similars if sid != char_id]
            
            if similars:
                # Marca para merge com o mais similar
                target_id, score = similars[0]
                if score > threshold:
                    chars_to_merge.append((char_id, target_id, score))
        
        # Executa merges
        merged_ids = set()
        
        # Carrega tiles se necessário para atualizar referências
        if self.db._tiles_df is None:
            tile_parquet = self.db.cache_dir / "tiles.parquet"
            if tile_parquet.exists():
                self.db._tiles_df = pd.read_parquet(tile_parquet)
        
        for char_id, target_id, score in chars_to_merge:
            if char_id in merged_ids or target_id in merged_ids:
                # Evita encadeamento complexo (A->B, B->C) em um único passo por simplicidade
                continue
            
            print(f"[CharacterService] Mesclando {char_id} -> {target_id} (score={score:.3f})")
            
            # 1. Atualiza referências nos tiles
            if self.db._tiles_df is not None:
                for idx, row in self.db._tiles_df.iterrows():
                    active_ids = row.get('active_char_ids', [])
                    if isinstance(active_ids, str):
                        active_ids = json.loads(active_ids)
                    
                    if char_id in active_ids:
                        # Substitui ID antigo pelo novo
                        active_ids = [target_id if cid == char_id else cid for cid in active_ids]
                        # Remove duplicatas que possam ter surgido (se ambos estavam no tile)
                        active_ids = list(set(active_ids))
                        self.db._tiles_df.at[idx, 'active_char_ids'] = json.dumps(active_ids)
            
            # 2. Atualiza contagem e metadados do target
            mask = self.db._characters_df['char_id'] == target_id
            if mask.any():
                idx = self.db._characters_df[mask].index[0]
                self.db._characters_df.at[idx, 'bbox_count'] += 1
                # Poderíamos fundir paletas aqui também se necessário
            
            merged_ids.add(char_id)
            merged_count += 1
        
        # Remove personagens mesclados do DataFrame principal
        if merged_ids:
            self.db._characters_df = self.db._characters_df[~self.db._characters_df['char_id'].isin(merged_ids)].reset_index(drop=True)
            print(f"[CharacterService] {merged_count} personagens mesclados, "
                  f"{len(self.db._characters_df)} restantes")
            
            # Salva alterações imediatamente para persistir o merge
            self.db.save_all()
            
            # Recarrega FAISS index para remover os deletados
            # Reconstrói índice usando VectorIndex manual ADD (o Index original não suporta delete fácil no SimpleIndex)
            # Como a implementação do VectorIndex pode variar, vamos reconstruir se possível
            # A chamada original era self._vector_index = VectorIndex(...)
            # Aqui precisamos chamar algo no DB para resetar o index.
            
            # Assumindo que podemos acessar e recriar o index no DB
            from core.database.vector_index import VectorIndex
            if hasattr(self.db, '_vector_index'):
                 # Recria
                 self.db._vector_index = VectorIndex(self.db.embedding_dim)
                 chars_list = self.db._characters_df.to_dict('records')
                 for row in chars_list:
                     char_id = row['char_id']
                     emb = self.db.load_embedding(char_id, 'clip')
                     if emb is not None:
                         self.db._vector_index.add(char_id, emb)
