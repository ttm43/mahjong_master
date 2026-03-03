import cv2
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small
from pathlib import Path
from src.app_logging import get_logger

class TileClassifier:
    TILE_CLASSES = [
        '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
        '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
        '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
        'E', 'S', 'W', 'N', 'P', 'F', 'C' # 东, 南, 西, 北, 白(Haku/P), 发(Hatsu/F), 中(Chun/C)
    ]
    
    def __init__(self, model_path, device='cpu'):
        self.logger = get_logger("classifier")
        self.device = torch.device(device)
        self.model = mobilenet_v3_small(num_classes=34)
        self.weights_loaded = False
        
        # Load weights gracefully if they exist; otherwise keep random initialization.
        try:
            path = Path(model_path)
            if path.exists():
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.weights_loaded = True
            else:
                self.logger.warning("Classifier weights not found at %s. Using random initialization.", model_path)
        except Exception as exc:
            self.logger.warning(
                "Failed to load classifier weights at %s: %s. Using random initialization.",
                model_path,
                exc,
            )
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((96, 96)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify(self, patch_img):
        """
        patch_img: numpy array of shape (H, W, 3) representing BGR image from OpenCV
        Returns predicted class label string
        """
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
        
        tensor_img = self.transform(rgb_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor_img)
            _, predicted = torch.max(outputs, 1)
            
        idx = predicted.item()
        return self.TILE_CLASSES[idx]
