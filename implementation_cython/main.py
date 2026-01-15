import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import torch
import time

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from medial_axis import trajectoire,vecteur_directeur,check_collisions,generer_alertes_collision
from description_image import decode_segmap_cython
from bounding_box import get_bounding_boxes,generer_description
from prediction_model import predict_model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#lien vers le dossier qui contient le modèle pré-entrainé
dossier_modele_fine_tune = r"C:\Users\rasou\Desktop\PER\Modele_Complet"

#définition du processor qui traite les images avant de les envoyer au modèle
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")

#import du modèle pré entrainé
model_finetuned = SegformerForSemanticSegmentation.from_pretrained(dossier_modele_fine_tune).to(device)
model_finetuned.eval()

dict_korean_color = {0: (255, 128, 255), 1: (0, 0, 255), 2: (230, 170, 255), 3: (0, 0, 0), 4: (255, 155, 155), 5: (255, 255, 0), 6: (0, 255, 0)}

palette = {
    0: (128, 64, 128),    # Road (Violet Cityscapes)
    1: (244, 35, 232),    # Sidewalk (Rose Cityscapes) - Surface principale
    2: (110, 110, 110),   # Alley (Gris sombre) - Pour différencier de la route
    3: (200, 0, 0),       # Bike Lane (Rouge Foncé) - Zone de danger
    4: (0, 255, 255),     # Braille Blocks (CYAN ELECTRIQUE) - Doit péter sur le rose !
    5: (255, 165, 0),     # Caution Zone (Orange Vif) - Attention danger
    6: (70, 70, 70),      # Building (Gris foncé Cityscapes)
    7: (190, 153, 153),   # Fence (Beige Cityscapes)
    8: (220, 20, 60),     # Pedestrian (Rouge Cityscapes)
    9: (153, 153, 153),   # Pole (Gris clair Cityscapes)
    10: (107, 142, 35),   # Vegetation (Vert Cityscapes)
    11: (0, 0, 142),      # Vehicle (Bleu foncé Cityscapes)
    12: (102, 102, 156),  # Wall (Bleu gris Cityscapes)
    13: (220, 220, 0),     # Traffic Sign (Jaune Cityscapes)

}
id2label = {
    0: "La Route",
    4: "Les Bandes podotactiles",
    5: "La Zone de danger",
    8: "La Personne",
    9: "Le Poteau",
    11: "Le Véhicule",
    13: "Le Panneau de signalisation",
}

palette_np = np.zeros((14, 3), dtype=np.uint8)
for class_id, color in palette.items():
    palette_np[class_id] = color


TARGET_CLASSES_NP = np.array([0, 4, 5, 8, 9, 11, 13], dtype=np.int32)

if __name__ == "__main__":
    start = time.perf_counter()

    # 1. Chargement et exécution
    path_img = r"C:\Users\rasou\Desktop\PER\DATA\Polygon_14_new\Polygon_1481\ZED2_KSC_114777_L_P039920.png"
    #path_img = r"C:\Users\rasou\Desktop\PER\DATA\Bbox_30_new\Bbox_2412\ZED3_KSC_130023_L_P034854.png"
    img = cv2.imread(path_img)

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    prediction_model = predict_model(model=model_finetuned,processor=processor,img=img,device=device)
    prediction_short = prediction_model.astype(np.int16)
    img_rgb = decode_segmap_cython(prediction_short,palette_np)
    trajectoire = trajectoire(img_rgb,5)
    trajectoire_float = trajectoire.astype(np.float32)
    vecteur_directeur = vecteur_directeur(trajectoire_float)


    # Obtenir les données
    boxes = get_bounding_boxes(prediction_short, TARGET_CLASSES_NP, min_area=30)
    description_image = generer_description(boxes, vecteur_directeur, id2label)

    collisions = check_collisions(trajectoire_float, boxes)
    allerte_collisions = generer_alertes_collision(collisions, id2label)
    end = time.perf_counter()


    # Exemple d'utilisation pour ta navigation
    for obj in boxes:
        print(f"Classe {obj['id']} détectée en {obj['bbox']}")

    for phrase in description_image:
        print(phrase)

    for alerte in allerte_collisions:
        print(alerte)

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(trajectoire)
    plt.subplot(1,3,3)
    plt.imshow(img_rgb)
    plt.show()
    ##traj = trajectoire(mask, 20)

    print(f"Durée précise : {end - start:.6f} secondes")