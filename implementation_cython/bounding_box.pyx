import numpy as np
cimport numpy as np
cimport cython
import cv2
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def get_bounding_boxes(short[:, :] pred_mask, int[:] target_ids, int min_area=50):
    """
    Optimisé : Retourne une liste de dicts (label, coords) pour l'embarqué.
    """
    cdef int h = pred_mask.shape[0]
    cdef int w = pred_mask.shape[1]
    cdef int i, class_id, n_targets = target_ids.shape[0]

    # Liste pour stocker les résultats
    results = []

    # On crée un buffer réutilisable pour éviter les réallocations mémoire
    cdef np.ndarray[np.uint8_t, ndim=2] binary_mask_np = np.zeros((h, w), dtype=np.uint8)
    cdef unsigned char[:, :] binary_mask = binary_mask_np

    for i in range(n_targets):
        class_id = target_ids[i]

        with nogil:
            for r in range(h):
                for c in range(w):
                    if pred_mask[r, c] == class_id:
                        binary_mask[r, c] = 255
                    else:
                        binary_mask[r, c] = 0

        # Extraction des contours (OpenCV)
        contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)

                results.append({
                    "id": class_id,
                    "bbox": (int(x), int(y), int(x + bw), int(y + bh)), # x1, y1, x2, y2
                    "area": int(area)
                })

    return results


def generer_description(list boxes, float[:] vecteur_directeur, dict id2label):
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