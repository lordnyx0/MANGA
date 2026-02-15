import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Tenta importar FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[VectorIndex] AVISO: FAISS não disponível. Funcionalidade de busca vetorial desabilitada.")

class VectorIndex:
    """
    Wrapper para índice FAISS de personagens.
    Gerencia adição, busca e persistência de embeddings vetoriais.
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = None
        self.id_map: Dict[int, str] = {}  # index_id -> char_id
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (Cosine similarity if normalized)
            
    def add(self, char_id: str, embedding: torch.Tensor):
        """
        Adiciona um embedding ao índice.
        
        Args:
            char_id: ID único do personagem
            embedding: Tensor do embedding (será normalizado)
        """
        if not FAISS_AVAILABLE or self.index is None:
            return
            
        # Normaliza embedding para busca por cosseno via Inner Product
        vec = embedding.cpu().numpy().astype('float32')
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        # Adiciona ao índice
        idx = self.index.ntotal
        self.index.add(vec.reshape(1, -1))
        self.id_map[idx] = char_id
        
    def search(
        self, 
        query_embedding: torch.Tensor, 
        top_k: int = 5, 
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Busca itens similares no índice.
        
        Args:
            query_embedding: Embedding de consulta
            top_k: Número máximo de resultados
            threshold: Score mínimo de similaridade (0-1)
            
        Returns:
            Lista de tuplas (char_id, score)
        """
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return []
        
        # Normaliza query
        vec = query_embedding.cpu().numpy().astype('float32')
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            
        # Busca
        # top_k não pode ser maior que o número de vetores
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec.reshape(1, -1), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                char_id = self.id_map.get(int(idx))
                if char_id:
                    results.append((char_id, float(score)))
                    
        return results

    def save(self, directory: Path):
        """Salva o índice e o mapa de IDs no diretório especificado."""
        if not FAISS_AVAILABLE or self.index is None:
            return
            
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Salva índice FAISS
        faiss.write_index(self.index, str(directory / "index.faiss"))
        
        # Salva mapa de IDs
        with open(directory / "id_map.json", 'w') as f:
            json.dump(self.id_map, f)
            
    def load(self, directory: Path):
        """Carrega o índice e o mapa de IDs do diretório."""
        if not FAISS_AVAILABLE:
            return
            
        directory = Path(directory)
        index_path = directory / "index.faiss"
        map_path = directory / "id_map.json"
        
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            
        if map_path.exists():
            with open(map_path, 'r') as f:
                self.id_map = {int(k): v for k, v in json.load(f).items()}
                
    def rebuild(self, embeddings: Dict[str, torch.Tensor]):
        """
        Reconstroi o índice do zero com os embeddings fornecidos.
        Útil após consolidação/deleção de personagens.
        """
        if not FAISS_AVAILABLE:
            return
            
        # Reseta índice
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.id_map = {}
        
        # Re-adiciona todos
        for char_id, emb in embeddings.items():
            self.add(char_id, emb)
            
    @property
    def count(self) -> int:
        """Retorna número de vetores no índice"""
        return self.index.ntotal if self.index else 0
