from typing import List, Dict, Optional, TYPE_CHECKING
from core.constants import SceneType

if TYPE_CHECKING:
    from core.database.chapter_db import ChapterDatabase

class NarrativeService:
    """
    Serviço de domínio para análise narrativa.
    Extrai lógica de negócio do ChapterDatabase.
    """
    
    def __init__(self, db: "ChapterDatabase"):
        self.db = db
        
    def detect_narrative_arcs(self):
        """
        Detecta arcos narrativos contínuos baseados nos tipos de cena das páginas.
        Suaviza transições e gera um relatório de estrutura.
        """
        if self.db._pages_df is None:
            self.db._load_parquets()
            
        if self.db._pages_df is None or len(self.db._pages_df) == 0:
            return

        print("[NarrativeService] Detectando arcos narrativos...")
        
        # Ordena por número da página
        pages = self.db._pages_df.sort_values('page_num').to_dict('records')
        
        current_arc_type = None
        arc_start_page = 0
        arcs = []
        
        for page in pages:
            scene_type = page.get('scene_type', SceneType.PRESENT.value)
            page_num = page['page_num']
            
            if current_arc_type is None:
                current_arc_type = scene_type
                arc_start_page = page_num
            elif scene_type != current_arc_type:
                # Fim do arco anterior
                arcs.append({
                    'type': current_arc_type,
                    'start': arc_start_page,
                    'end': page_num - 1,
                    'length': (page_num - 1) - arc_start_page + 1
                })
                # Início do novo
                current_arc_type = scene_type
                arc_start_page = page_num
        
        # Adiciona último arco
        if current_arc_type is not None:
            arcs.append({
                'type': current_arc_type,
                'start': arc_start_page,
                'end': pages[-1]['page_num'],
                'length': pages[-1]['page_num'] - arc_start_page + 1
            })
            
        # Imprime relatório
        print(f"[NarrativeService] Estrutura Narrativa Identificada:")
        for arc in arcs:
            print(f"  - Páginas {arc['start']+1}-{arc['end']+1} ({arc['length']} pgs): {str(arc['type']).upper()}")
            
        # Poderíamos salvar isso em um metadata.json se necessário
