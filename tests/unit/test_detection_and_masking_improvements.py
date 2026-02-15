import numpy as np

from core.analysis.mask_processor import MaskProcessor
from core.detection.yolo_detector import YOLODetector, DetectionResult
from core.utils.image_ops import extract_canny_edges
from core.pass2_generator import Pass2Generator
from core.pass1_analyzer import Pass1Analyzer


def test_extract_canny_edges_auto_threshold_detects_lineart():
    img = np.ones((64, 64), dtype=np.uint8) * 255
    img[20:22, 10:54] = 0

    edges = extract_canny_edges(img, low_threshold=None, high_threshold=None)

    assert edges.shape == img.shape
    assert edges.dtype == np.uint8
    assert edges.sum() > 0


def test_mask_dilation_does_not_invade_foreground():
    processor = MaskProcessor(overlap_dilation=2)

    front = np.zeros((30, 30), dtype=np.uint8)
    back = np.zeros((30, 30), dtype=np.uint8)
    front[5:15, 5:15] = 255
    back[10:20, 10:20] = 255

    masks = {
        "front": (front.copy(), False, 0),
        "back": (back.copy(), True, 10),
    }

    result = processor._apply_overlap_dilation(masks, ["front", "back"])
    back_after = result["back"][0]

    # Back não deve reocupar pixels do front após dilatação
    assert np.logical_and(back_after > 0, front > 0).sum() == 0


def test_yolo_overlap_dedup_keeps_highest_confidence():
    detector = YOLODetector.__new__(YOLODetector)
    detector.CHARACTER_CLASSES = {0, 1}

    dets = [
        DetectionResult(bbox=(10, 10, 100, 100), confidence=0.95, class_id=0, class_name="body", prominence_score=0.8),
        DetectionResult(bbox=(12, 12, 98, 98), confidence=0.70, class_id=0, class_name="body", prominence_score=0.75),
        DetectionResult(bbox=(120, 10, 180, 80), confidence=0.88, class_id=3, class_name="text", prominence_score=0.0),
    ]

    out = detector._deduplicate_by_overlap(dets)

    bodies = [d for d in out if d.class_id == 0]
    assert len(bodies) == 1
    assert bodies[0].confidence == 0.95


def test_character_mask_matches_bbox_area_without_one_pixel_bleed():
    generator = Pass2Generator.__new__(Pass2Generator)

    # bbox de 10x10 dentro do tile
    mask = generator._create_character_mask(
        char_bbox=(10, 10, 20, 20),
        tile_bbox=(0, 0, 64, 64),
        tile_size=(64, 64),
    )

    assert mask is not None
    arr = np.array(mask)
    assert int((arr > 0).sum()) == 100


def test_pass1_character_filter_excludes_text_and_frame():
    assert Pass1Analyzer._is_character_detection(0) is True
    assert Pass1Analyzer._is_character_detection(1) is True
    assert Pass1Analyzer._is_character_detection(2) is False
    assert Pass1Analyzer._is_character_detection(3) is False


def test_yolo_class_threshold_uses_per_class_thresholds():
    detector = YOLODetector.__new__(YOLODetector)
    detector.conf_threshold = 0.30
    detector.face_conf_threshold = 0.22
    detector.text_conf_threshold = 0.20

    assert detector._class_conf_threshold(0) == 0.30
    assert detector._class_conf_threshold(1) == 0.22
    assert detector._class_conf_threshold(2) == 0.30
    assert detector._class_conf_threshold(3) == 0.20
