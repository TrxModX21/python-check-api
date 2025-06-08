import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import os

MODEL_PATH = os.path.join("model", "final_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sesuaikan dengan arsitektur model kamu
class EfficientNetB0Gray(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetB0Gray, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=False)
        self.efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

# Load model
model = EfficientNetB0Gray(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform gambar sama seperti waktu training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Fungsi prediksi
def predict_image(image: Image.Image) -> str:
    # image_tensor = Image.open(image).convert('RGB')  # tetap RGB agar bisa dikonversi ke grayscale
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return "stunting" if predicted.item() == 1 else "normal"