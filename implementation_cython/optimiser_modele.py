import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def quantize_and_prune_model(model, save_path, pruning_amount=0.2):

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name="weight", amount=pruning_amount)
            prune.remove(module, 'weight')

    model.cpu()
    model_quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    torch.save(model_quantized.state_dict(), save_path)
    print(f"✅ Modèle pruné et quantisé sauvegardé dans : {save_path}")

    return model_quantized

if __name__ == "__main__":
    MODEL_DIR = r"C:\Users\rasou\Desktop\PER\Modele_Complet"
    SAVE_PATH = r"C:\Users\rasou\Desktop\PER\Modele_Complet\modele_quant_prune"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    quantize_and_prune_model(model=model, save_path=SAVE_PATH, pruning_amount=0.2)
