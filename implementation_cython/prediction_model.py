import torch
def predict_model(model, processor, img, device):
    inputs = processor(images=img, return_tensors="pt",do_rescale=False).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)

    logits = outputs.logits
    prediction = logits.argmax(dim=1)[0].cpu().numpy()
    return prediction