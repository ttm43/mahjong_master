import numpy as np
from src.vision.classifier import TileClassifier

def test_classifier():
    # Model doesn't exist, will use random weights but should not crash
    classifier = TileClassifier(model_path="dummy_mobilenet.pt", device='cpu')
    
    # Dummy RGB patch
    dummy_patch = np.zeros((100, 80, 3), dtype=np.uint8)
    label = classifier.classify(dummy_patch)
    
    assert isinstance(label, str)
    assert label in classifier.TILE_CLASSES
