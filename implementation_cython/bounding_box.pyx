import numpy as np
cimport numpy as np
cimport cython
import cv2
from libc.math cimport sqrt

# On définit des seuils spécifiques (exprimés en % de la surface totale)
# Poteaux (9) et Panneaux (13) sont fins, Véhicules (11) sont gros.
CLASS_THRESHOLDS = {
    4: 0.1,  # Braille Blocks (5%)
    5: 0.1,   # Caution Zone (20%)
    8: 0.05,   # Pedestrian (30%)
    9: 0.02,  # Pole (2%) - Très fin
    11: 0.1,  # Vehicle (10%)
    13: 0.02  # Traffic Sign (2%)
}

@cython.boundscheck(False)
@cython.wraparound(False)
def get_bounding_boxes(short[:, :] pred_mask, int[:] target_ids, float default_min_area_ratio=0.1):
    cdef int h = pred_mask.shape[0]
    cdef int w = pred_mask.shape[1]
    cdef float total_area = <float>(h * w)
    cdef int i, r, c, class_id, n_targets = target_ids.shape[0]
    cdef float current_min_area

    results = []

    # Buffer pour le masque binaire
    cdef np.ndarray[np.uint8_t, ndim=2] binary_mask_np = np.zeros((h, w), dtype=np.uint8)
    cdef unsigned char[:, :] binary_mask = binary_mask_np

    # Noyau pour l'ouverture morphologique (nettoyage du bruit)

    for i in range(n_targets):
        class_id = target_ids[i]

        # Déterminer le seuil d'aire pour cette classe précise
        # Si la classe n'est pas dans le dictionnaire, on utilise le ratio par défaut
        ratio = CLASS_THRESHOLDS.get(class_id, default_min_area_ratio)
        current_min_area = ratio * total_area

        with nogil:
            for r in range(h):
                for c in range(w):
                    if pred_mask[r, c] == class_id:
                        binary_mask[r, c] = 255
                    else:
                        binary_mask[r, c] = 0

        # --- ÉTAPE DE NETTOYAGE ---


        # Extraction des contours
        contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 90000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                results.append({
                    "id": class_id,
                    "bbox": (int(x), int(y), int(x + bw), int(y + bh)),
                    "area": int(area)
                })

    return results



@cython.boundscheck(False)
@cython.wraparound(False)
def get_position_objets(list boxes, float[:] vecteur_directeur, dict id2label):
    cdef int H = 720
    cdef int W = 480
    cdef float x_0 = W / 2.0
    cdef float y_0 = <float>H

    cdef list results = []
    cdef float a = vecteur_directeur[0]
    cdef float b = vecteur_directeur[1]

    # Normalisation
    cdef float norme_dir = sqrt(a*a + b*b)
    if norme_dir > 0:
        a /= norme_dir
        b /= norme_dir

    for obj in boxes:
        x1, y1, x2, y2 = obj["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        vx = cx - x_0
        vy = cy - y_0
        norme = sqrt(vx*vx + vy*vy)

        if norme < 1e-6: continue

        vx_norm = vx / norme
        vy_norm = vy / norme

        # Produit scalaire & Déterminant
        ps = vx_norm * a + vy_norm * b
        det = a * vy_norm - b * vx_norm

        pos_horiz = "on the left" if det < -0.1 else ("on the right" if det > 0.1 else "aligned")
        pos_vert = "in front"

        label_name = id2label.get(obj["id"], "object")

        results.append({
            "label": label_name,
            "description": f" {pos_vert} {pos_horiz}",
            "bbox": obj["bbox"],
            "area": obj["area"]
        })

    return results




#id2label = {4:"Braille Blocks", 5:"Caution Zone", 8:"Pedestrian", 9:"Pole", 11:"Vehicle", 13:"Traffic Sign"}

def generer_description(list position_objets):
    """
    Takes the list of positioned objects and builds a voice announcement in English.
    """
    cdef list morceaux = []
    cdef str annonce = ""
    cdef str label, desc

    cdef float coeff_danger = 0.0
    cdef float surface_image = 720*480

    # On utilise bien 'position_objets' comme défini dans l'argument juste au-dessus
    for obj in position_objets:
        label = obj["label"]
        desc = obj["description"]
        area = obj["area"]

        if label == "Pedestrian":
            morceaux.append(f"Caution, pedestrian {desc}")
            coeff_danger += 1*area

        elif label == "Vehicle":
            morceaux.append(f"Vehicle detected {desc}")
            coeff_danger += 2*area

        elif label == "Braille Blocks":
            morceaux.append(f"Tactile paving {desc}")

        elif label == "Traffic Sign":
            morceaux.append(f"Traffic sign {desc}")

        else:
            #morceaux.append(f"{label} {desc}")
            continue



    #Il faut filtre annonce d'une certaine manière

    print("coeff_danger = ",(coeff_danger/surface_image)*100)

    if((coeff_danger/surface_image)*100 >= 0.3):
        annonce = ". ".join(morceaux)
    else:
        annonce = "No danger"

    return annonce




@cython.boundscheck(False)
@cython.wraparound(False)
def generer_description2(list boxes, float[:] vecteur_directeur, dict id2label):

    cdef int H = 128
    cdef int W = 128

    # Point central (bas de l'image)
    cdef float x_0 = W / 2.0
    cdef float y_0 = <float>H

    cdef list phrases = []
    cdef float a = vecteur_directeur[0]
    cdef float b = vecteur_directeur[1]

    # Normalisation du vecteur directeur s'il ne l'est pas
    cdef float norme_dir = sqrt(a*a + b*b)
    if norme_dir > 0:
        a /= norme_dir
        b /= norme_dir

    cdef dict obj
    cdef float x, y, x2, y2, cx, cy, vx, vy, norme, vx_norm, vy_norm, ps, det
    cdef str devant_derriere, gauche_droite, label_name

    for obj in boxes:
        # Extraction des coordonnées
        bbox = obj["bbox"]
        x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        cx = (x + x2) / 2.0
        cy = (y + y2) / 2.0

        vx = cx - x_0
        vy = cy - y_0
        norme = sqrt(vx*vx + vy*vy)

        if norme < 1e-6: # Éviter division par zéro
            continue

        vx_norm = vx / norme
        vy_norm = vy / norme

        # Produit scalaire (Z-axis / Proximité)
        ps = vx_norm * a + vy_norm * b
        devant_derriere = "devant" if ps > 0 else "derrière"

        # Déterminant (X-axis / Latéral)
        det = a * vy_norm - b * vx_norm
        if det < -0.1: # Ajout d'une petite zone morte pour "aligné"
            gauche_droite = "à gauche"
        elif det > 0.1:
            gauche_droite = "à droite"
        else:
            gauche_droite = "aligné"

        label_name = id2label.get(obj["id"], "objet inconnu")


        phrases.append(f"{label_name} est {devant_derriere} {gauche_droite}")

    return phrases