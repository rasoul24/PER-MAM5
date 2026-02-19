import torch
import torch.nn.functional as F

def predict_model2(model, processor, img, device):
    inputs = processor(images=img, return_tensors="pt",do_rescale=False).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)

    logits = outputs.logits
    prediction = logits.argmax(dim=1)[0].cpu().numpy()
    return prediction


def predict_model(model, processor, img, device):
    # Préparation des entrées
    inputs = processor(images=img, return_tensors="pt", do_rescale=False).to(device)

    with torch.no_grad():
        # Note : torch.amp.autocast('cuda') est la version moderne si tu veux nettoyer le warning
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)

    logits = outputs.logits  # Forme actuelle : [1, num_classes, 128, 128]

    # --- AJOUT : Redimensionnement aux dimensions de l'image d'origine ---
    # img.shape[:2] donne (hauteur, largeur)
    upsampled_logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)

    # Calcul de la classe dominante par pixel sur l'image redimensionnée
    prediction = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    return prediction