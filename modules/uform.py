import os
import uform
import torch

embedding_dim = int(os.getenv("EMBEDDING_DIM", 256))
image_preprocess = None
text_preprocess = None

processors, models = uform.get_model(
    os.getenv("UFORM_MODEL", "unum-cloud/uform3-image-text-multilingual-base")
)

model_text = models[uform.Modality.TEXT_ENCODER]
model_image = models[uform.Modality.IMAGE_ENCODER]
processor_text = processors[uform.Modality.TEXT_ENCODER]
processor_image = processors[uform.Modality.IMAGE_ENCODER]

def image_forward_fn(model, images, device, transform):
    model = model_image
    images = processor_image(images)#.to(device)
    return model.encode(images)

def text_forward_fn(model, texts, device, transform):
    model = model_text
    texts = processor_text(texts)
    # texts = {k: v.to(device) for k, v in texts.items()}
    return model.encode(texts)

# NOTE: Dummy model used to satisfy the interface. Not applicable for the now disjoint modalities.
model = torch.nn.Module()
