from enum import Enum, auto

class SceneType(str, Enum):
    """Tipos de cena suportados pelo sistema."""
    PRESENT = "present"
    FLASHBACK = "flashback"
    DREAM = "dream"
    NIGHTMARE = "nightmare"
    INDOORS = "indoors"
    OUTDOORS = "outdoors"
    NIGHT = "night"
    SUNSET = "sunset"
    
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

class DetectionClass(int, Enum):
    """IDs de classes detectadas pelo YOLO."""
    BODY = 0
    FACE = 1
    # ID 2 reservado (ex: mãos/acessórios futuros)
    TEXT = 3

class GenerationMode(str, Enum):
    """Modos de geração."""
    SINGLE_TILE = "single"
    MULTI_TILE = "multi"

# Configuração de prompts por tipo de cena (Extensível)
SCENE_DESCRIPTIONS = {
    SceneType.PRESENT: 'present day scene, clear colors',
    SceneType.FLASHBACK: 'flashback scene, desaturated colors, nostalgic',
    SceneType.DREAM: 'dream scene, ethereal, glowing',
    SceneType.NIGHTMARE: 'nightmare scene, dark, ominous',
    SceneType.INDOORS: 'indoor scene, detailed background',
    SceneType.OUTDOORS: 'outdoor scene, natural lighting',
    SceneType.NIGHT: 'night scene, dark atmosphere, moon lighting',
    SceneType.SUNSET: 'sunset scene, warm colors, orange sky'
}
