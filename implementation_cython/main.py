import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import torch
import time

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from medial_axis import trajectoire,vecteur_directeur,check_collisions,generer_alertes_collision
from description_image import decode_segmap_cython
from bounding_box import get_bounding_boxes,generer_description, get_position_objets
from prediction_model import predict_model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#lien vers le dossier qui contient le modèle pré-entrainé
#dossier_modele_fine_tune = r"/home/rasoul/Bureau/PER/Modele_Complet"
dossier_modele_fine_tune = r"C:\Users\rasou\Desktop\PER\Modele_Complet"
dossier_modele_dist = r"C:\Users\rasou\Desktop\PER\model_11"

#définition du processor qui traite les images avant de les envoyer au modèle
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
processor_dist = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")

#import du modèle pré entrainé
model_finetuned = SegformerForSemanticSegmentation.from_pretrained(dossier_modele_fine_tune).to(device)
model_finetuned.eval()

model_dist = SegformerForSemanticSegmentation.from_pretrained(
        dossier_modele_dist,
        local_files_only=True
    ).to(device)

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


id2label = {0:"Road",1:"SideWalk",2:"Alley",3:"Bike lane", 4:"Braille Blocks", 5:"Caution Zone",6:"Building",7:"Fence", 8:"Pedestrian", 9:"Pole",10:"Vegetation", 11:"Vehicle", 12:"Wall" ,13:"Traffic Sign"}

palette_np = np.zeros((14, 3), dtype=np.uint8)
for class_id, color in palette.items():
    palette_np[class_id] = color


TARGET_CLASSES = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13], dtype=np.int32)
CLASSES_AFFICHE = np.array([8,9,11,12], dtype=np.int32)

if __name__ == "__main__":
    start = time.perf_counter()

    # 1. Chargement et exécution
    #path_img = r"C:\Users\rasou\Desktop\PER\DATA\Bbox_30_new\Bbox_2412\ZED3_KSC_130023_L_P034854.png"
    path_img = r"C:\Users\rasou\Desktop\PER\DATA\Polygon_14_new\Polygon_1502\MP_SEL_PN001377.jpg"
    img = cv2.imread(path_img)

    img_512 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    img_display_512 = cv2.cvtColor(img_512, cv2.COLOR_BGR2RGB)


    plt.imshow(img_display_512)
    plt.show()

    if img is None:
        print("Erreur : Image non trouvée.")
    else:
        # Conversion pour l'affichage (OpenCV lit en BGR, Matplotlib affiche en RGB)
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # 2. Inférence et calculs Cython
        prediction_model = predict_model(model=model_finetuned, processor=processor, img=img, device=device)
        prediction_model_dist = predict_model(model=model_dist, processor=processor_dist, img=img, device=device)
        prediction_short = prediction_model.astype(np.int16)
        prediction_short_dist = prediction_model_dist.astype(np.int16)

        # img_rgb contient le masque coloré
        img_rgb = decode_segmap_cython(prediction_short, palette_np)
        img_rgb_dist = decode_segmap_cython(prediction_short_dist, palette_np)

        # --- Affichage du Modèle B0 ---
        plt.figure(figsize=(12, 8)) # Format large pour bien voir les détails
        plt.imshow(img_rgb_dist)
        plt.title("Segmentation avec le modèle B0\n(Modèle de base distillé)", fontsize=12, fontweight='bold', pad=20)
        plt.axis('off') # Supprime les axes pour un rendu "pro"
        plt.tight_layout()
        plt.show()

        # --- Affichage du Modèle B2 ---
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title("Segmentation avec le modèle B2\n(Modèle Fine-tuné)", fontsize=12, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        # Trajectoire
        traj_mask = trajectoire(img_rgb, 5) # On renomme pour éviter de masquer la fonction
        trajectoire_float = traj_mask.astype(np.float32)
        v_dir = vecteur_directeur(trajectoire_float)

        # 3. Détection des boîtes et collisions
        boxes = get_bounding_boxes(prediction_short, TARGET_CLASSES)
        collisions = check_collisions(trajectoire_float, boxes)
        message_collision = generer_alertes_collision(collisions, id2label)

        positions = get_position_objets(boxes, v_dir, id2label)
        phrases = generer_description(positions)
        print("phrases = ", phrases)

        # --- AJOUT DES BOUNDING BOXES SUR LE MASQUE ---
        # --- AJOUT DES BOUNDING BOXES ET DU VECTEUR ---
        for obj in boxes:

            if obj["id"] in CLASSES_AFFICHE:
                x1, y1, x2, y2 = obj["bbox"]
                label_name = id2label.get(obj["id"], f"ID:{obj['id']}")

                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(img_rgb, label_name, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)



        # 3. Détection des boîtes et collisions
        boxes_dist = get_bounding_boxes(prediction_short_dist, TARGET_CLASSES)
        collisions_dist = check_collisions(trajectoire_float, boxes)
        message_collision_dist = generer_alertes_collision(collisions_dist, id2label)

        positions_dist = get_position_objets(boxes_dist, v_dir, id2label)
        phrases_dist = generer_description(positions_dist)

        print("phrases_dist = ",phrases_dist )

        # --- AJOUT DES BOUNDING BOXES SUR LE MASQUE ---
        # --- AJOUT DES BOUNDING BOXES ET DU VECTEUR ---
        for obj in boxes_dist:
            if obj["id"] in CLASSES_AFFICHE:
                x1, y1, x2, y2 = obj["bbox"]
                label_name = id2label.get(obj["id"], f"ID:{obj['id']}")

                cv2.rectangle(img_rgb_dist, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(img_rgb_dist, label_name, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)


         # --- Affichage du Modèle B2 ---
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title("Segmentation avec le modèle B2\n + Bounding Boxes", fontsize=12, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

              # --- Affichage du Modèle B2 ---
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb_dist)
        plt.title("Segmentation avec le modèle B0\n + Bounding Boxes", fontsize=12, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


