import sys
import torch
from PIL import Image

from model import build_model, get_device
from dataset import get_transforms


CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

DISPLAY_NAMES = {
    'akiec': 'Actinic Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevus',
    'vasc':  'Vascular Lesion',
}


def load_model(model_path, device=None):
    if device is None:
        device = get_device()
    model = build_model(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict(image, model, device):
    transform = get_transforms('val')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    results = {
        DISPLAY_NAMES[cls]: round(float(prob) * 100, 2)
        for cls, prob in zip(CLASS_NAMES, probs)
    }
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python predict.py <image_path> [model_path]')
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'models/best_model.pth'

    model, device = load_model(model_path)
    image = Image.open(image_path).convert('RGB')
    results = predict(image, model, device)

    print('\nTop 3 predictions:')
    for i, (name, confidence) in enumerate(list(results.items())[:3], 1):
        print(f'  {i}. {name}: {confidence:.1f}%')
