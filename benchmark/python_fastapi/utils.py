import io
import json

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

model = models.resnet18(pretrained=True)
model.eval()
model = model.cuda().half()
imagenet_class_index = json.load(open('./imagenet_class_index.json'))
traced_model_cuda = torch.jit.script(model)
frozen_traced_model_cuda = torch.jit.freeze(traced_model_cuda)
optimized_frozen_traced_model_cuda = torch.jit.optimize_for_inference(frozen_traced_model_cuda)


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


@torch.no_grad()
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes).cuda().half()
    outputs = torch.softmax(optimized_frozen_traced_model_cuda(tensor), 1)
    value, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    torch.cuda.empty_cache()
    return imagenet_class_index[predicted_idx], value.item()


def get_result(image_file, is_api=False):
    image_bytes = image_file.file.read()
    (_, class_name), confidence = get_prediction(image_bytes)
    result = {
        "status": "success",
        "message": f"Class found for image was {class_name} with confidence {confidence:.2f}."
    }
    return result
