
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from skimage.morphology import medial_axis, dilation, disk
import cv2
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import sys
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import math

def chemin_milieu(mask_blue, plot=True):
    """
    mask_blue : bool (H, W), True = trottoir
    Retourne un squelette 'mince' qui suit le milieu du trottoir.
    """

    # 1) squelette complet
    skel, dist = medial_axis(mask_blue, return_distance=True)

    H, W = skel.shape
    main_skel = np.zeros_like(skel, dtype=bool)

    # 2) pour chaque ligne, garder le point de squelette
    #    le plus proche du centre horizontal
    cx = W // 2
    for y in range(H):
        xs = np.where(skel[y])[0]
        if len(xs) == 0:
            continue
        # x le plus proche du centre
        x_mid = xs[np.argmin(np.abs(xs - cx))]
        main_skel[y, x_mid] = True

    if plot:
        plt.figure(dpi=150)
        plt.imshow(mask_blue, cmap="gray", vmin=0, vmax=1)
        plt.contour(main_skel, levels=[0.5], colors="r", linewidths=2)
        plt.title("Chemin au milieu du trottoir")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return main_skel
def chemin_milieu_continu(mask_blue, plot=True):
    """
    mask_blue : bool (H, W), True = trottoir
    Retourne un chemin continu (main_skel) qui suit le milieu du trottoir.
    """

    # 1) squelette du trottoir
    skel, dist = medial_axis(mask_blue, return_distance=True)

    H, W = skel.shape
    main_skel = np.zeros_like(skel, dtype=bool)

    # 2) point de départ : en bas, pixel de squelette le plus "central" dans le trottoir
    y0 = H - 1
    xs = np.where(skel[y0])[0]
    if len(xs) == 0:
        # remonter jusqu'à trouver une ligne avec squelette
        found = False
        for y0 in range(H - 2, -1, -1):
            xs = np.where(skel[y0])[0]
            if len(xs) > 0:
                found = True
                break
        if not found:
            return main_skel  # squelette vide

    # centre du trottoir sur cette ligne: moyenne des xs
    cx0 = int(xs.mean())
    x_cur = cx0
    y_cur = y0
    main_skel[y_cur, x_cur] = True

    # 3) remonter ligne par ligne en cherchant, à chaque pas, le voisin "le plus central"
    for y in range(y_cur - 1, -1, -1):  # du bas vers le haut
        # on regarde les pixels de squelette sur cette ligne
        xs = np.where(skel[y])[0]
        if len(xs) == 0:
            continue

        # on ne veut pas sauter trop loin : on favorise la proximité en x
        # et le "centre" local du trottoir
        cx_line = int(xs.mean())
        # on définit une cible: mix entre continuité (x_cur) et centre du trottoir (cx_line)
        x_target = 0.5 * x_cur + 0.5 * cx_line

        # on choisit le pixel de squelette le plus proche de x_target
        x_next = xs[np.argmin(np.abs(xs - x_target))]
        main_skel[y, x_next] = True
        x_cur = x_next

    if plot:
        plt.figure(dpi=150)
        plt.imshow(mask_blue, cmap="gray", vmin=0, vmax=1)
        plt.contour(main_skel, levels=[0.5], colors="red", linewidths=2)
        plt.title("Chemin continu au milieu du trottoir")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return main_skel


def extract_main_skeleton_branch(skel):
    """
    Retourne uniquement la branche principale du squelette :
    celle qui part d'en haut et va le plus profond possible vers le bas.
    """
    # Coordonnées des pixels du squelette
    y, x = np.where(skel)
    points = list(zip(y, x))
    N = len(points)

    # Indexation rapide point -> index
    index = {p: i for i, p in enumerate(points)}

    # 8-voisinage
    neigh = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]

    # Construction du graphe
    rows = []
    cols = []
    data = []

    for p in points:
        i = index[p]
        py, px = p

        for dy, dx in neigh:
            q = (py + dy, px + dx)
            if q in index:  # voisin appartenant au squelette
                j = index[q]
                rows.append(i)
                cols.append(j)
                data.append(1)  # poids = 1

    graph = csr_matrix((data, (rows, cols)), shape=(N, N))

    # Points les plus hauts (y minimal)
    top_points = np.where(y == y.min())[0]

    # Points les plus bas (y maximal)
    bottom_points = np.where(y == y.max())[0]

    # Calcul chemin le plus long entre haut et bas
    dist_matrix, predecessors = shortest_path(
        graph,
        directed=False,
        return_predecessors=True
    )

    best_path = []
    best_len = 0

    for s in top_points:
        for t in bottom_points:
            d = dist_matrix[s, t]
            if np.isfinite(d) and d > best_len:
                best_len = d
                # reconstruction du chemin
                path = []
                cur = t
                while cur != -9999:
                    path.append(points[cur])
                    cur = predecessors[s, cur]
                best_path = path[::-1]

    # Création d'un masque de squelette filtré
    main_skel = np.zeros_like(skel, dtype=bool)
    for (py, px) in best_path:
        main_skel[py, px] = True

    return main_skel


def trajectoire(mask,dist):

  # Sélection du bleu pur (trottoir)
  m = mask.astype(np.float32) / 255.0
  eps = 0.1
  #mask_blue = (m[..., 2] = 1)  & (m[..., 0]=0 ) & (m[..., 1] = 0)
  #mask_blue = (m[..., 2] == 1) & (m[..., 0] == 0) & (m[..., 1] == 0)

  colors = [
     (0,0,255),
    (230,170,255)
    ]
  colors2 = [
     (0,0,255),
    (230,170,255),
    (208,88,255),
    (138,60,200),
    (88,38,128)
    ]
  # masque initial vide
  mask_filtre = np.zeros(m.shape[:2], dtype=bool)

  # ajout des couleurs
  for c in colors:
    mask_filtre |= (mask == c).all(axis=-1)

  mask_filtre = mask_filtre.astype(np.float32) / 255.0
  print("Bleu trouvé :", mask_filtre.sum())

  # Calcul du squelette + distances
  skel, distance = medial_axis(mask_filtre, return_distance=True)

  # Masque des obstacles (noir pur)
  obstacle_mask = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)

  main_skel = skel
  #main_skel = extract_main_skeleton_branch(skel)
  #main_skel = chemin_milieu(skel)
  #main_skel = chemin_milieu_continu(skel)
  print("Pixels squelette :", skel.sum())
  print("Pixels main squelette :", main_skel.sum())

  dist_on_main = distance * main_skel

  # Distance uniquement sur squelette
  #dist_on_skel = distance * skel

  # --- Protection : zone dangereuse autour des obstacles ---
  rayon_secu = dist      # en pixels
  zone_dangereuse = dilation(obstacle_mask, disk(rayon_secu))

  # Collision = squelette qui touche zone interdite
  collision = main_skel & zone_dangereuse

  # ------------------------------
  #       MESSAGE FINAL
  # ------------------------------

  if collision.any():
      print("⚠️ Collision (ou zone trop étroite) détectée ! Décallez vous a gauche")
  else:
      print("👍 Aucun risque de collision.")

  return main_skel



def vecteur_directeur(main_skel):
  idx1 = np.argmax(main_skel[0])
  idx2 = np.argmax(main_skel[-1])

  point1 = np.array([idx1, 0])
  point2 = np.array([idx2, 150])

  vecteur_dir = point1 - point2

  norme = np.linalg.norm(vecteur_dir)

  # Éviter la division par zéro
  if norme == 0:
      vecteur_dir = np.array([0, 0])
  else:
      vecteur_dir = vecteur_dir / norme
  return vecteur_dir


def segmentation_box(image,processor,model,vecteur_directeur):
  # Préparer l'image et prédire
  inputs = processor(images=image, return_tensors="pt").to(device)
  outputs = model(**inputs)
  # Post-traitement pour récupérer les boîtes

  H, W = image.shape[:2]         # récupérer hauteur et largeur
  target_sizes = torch.tensor([[H, W]],device=device)
  results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

  # Coordonnées du point central (correct x/y)
  x_0 = W // 2
  y_0 = H

  fig, ax = plt.subplots(1, 1, figsize=(12, 8))
  ax.imshow(image)

  Phrases = []
  a,b = vecteur_directeur

  for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
      x, y, x2, y2 = [float(i) for i in box.tolist()]

      # Centre de la box
      cx = (x + x2) / 2
      cy = (y + y2) / 2
      ax.plot(cx, cy, 'bo')  # point bleu au centre

      # Ligne (segment) du centre vers le coin supérieur gauche du bounding box
      line = FancyArrowPatch((x_0, y_0), (cx, cy),
                            color='blue',
                            arrowstyle='-')
      ax.add_patch(line)

      vx = cx - x_0
      vy = cy - y_0

      norme = math.sqrt(vx**2 + vy**2)

      vx_norm = vx / norme
      vy_norm = vy / norme

      produit_scalaire = vx_norm*a+vy_norm*b

      if produit_scalaire > 0:
          devant_derriere = "devant"
      else:
          devant_derriere = "derrière"

      # Produit vectoriel 2D → gauche ou droite
      det = a*vy_norm - b*vx_norm
      if det < 0:
          gauche_droite = "à gauche"
      elif det > 0:
          gauche_droite = "à droite"
      else:
          gauche_droite = "aligné"

      Phrases.append(f"La {model.config.id2label[label.item()]} se trouve à {( x, y, x2, y2)} {devant_derriere} {gauche_droite}")

      # Rectangle du bounding box
      rect = patches.Rectangle((x, y), x2-x , y2-y ,
                              linewidth=2, edgecolor='r', facecolor='none')
      ax.add_patch(rect)

      # Texte de label + score
      ax.text(x, y,
              f'{model.config.id2label[label.item()]}: {score.item():.2f}',
              color='white', fontsize=12,
              bbox=dict(facecolor='red', alpha=0.5))
  plt.show()

  for phrase in Phrases:
    print(phrase)

# Fonction pour faire la segmentation sur une image
def predict_mask(model, processor, img):
    inputs = processor(images=img, return_tensors="pt",do_rescale=False).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred = logits.argmax(dim=1)[0].cpu().numpy()
    return pred

# Fonction pour décoder l'output du modèle (ID->RGB)
def decode_segmap_global(mask_tensor,dict_rgb):
    mask_array = mask_tensor
    h,w = mask_array.shape
    rgb_image = np.zeros((h,w,3),dtype=np.uint8)
    for idx, color in dict_rgb.items():
        rgb_image[mask_array==idx] = color
    return rgb_image

def overlay_mask(image, mask_rgb, alpha=0.6):
    # image: (H, W, 3), mask_rgb: (h, w, 3) -> on adapte le masque
    h, w = image.shape[:2]
    mask_resized = cv2.resize(mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

    image = image.astype(np.uint8)
    mask_resized = mask_resized.astype(np.uint8)

    return cv2.addWeighted(image, 1 - alpha, mask_resized, alpha, 0)

def overlay_mask2(image, mask_rgb, alpha=0.6, main_skel=None, show=True):
    # 1) Redimensionner le masque sur l'image
    h, w = image.shape[:2]
    mask_resized = cv2.resize(mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

    image = image.astype(np.uint8)
    mask_resized = mask_resized.astype(np.uint8)

    overlay = cv2.addWeighted(image, 1 - alpha, mask_resized, alpha, 0)

    # 2) Optionnel : tracer le squelette par-dessus
    if show:
        plt.figure(dpi=150)
        plt.imshow(overlay)

        if main_skel is not None:
            # redimensionner le squelette à la même taille
            skel_resized = cv2.resize(
                main_skel.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            plt.contour(skel_resized, levels=[0.5], colors="red", linewidths=2)

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return overlay

def show_overlay_with_traj(image, mask_rgb, main_skel, alpha=0.6):
    H, W = image.shape[:2]

    # 1) resize mask to image
    mask_resized = cv2.resize(mask_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

    # 2) resize skeleton to image (en gardant booléen)
    skel_uint8 = main_skel.astype(np.uint8)
    print("main_skel sum =", main_skel.sum())

    skel_resized = cv2.resize(skel_uint8, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    print("skel resized shape=",skel_resized.shape)

    # 3) overlay image + mask
    overlay = cv2.addWeighted(
        image.astype(np.uint8),
        1 - alpha,
        mask_resized.astype(np.uint8),
        alpha,
        0,
    )

    # 4) afficher + contour de la trajectoire
    plt.figure(dpi=150)
    plt.imshow(overlay)
    plt.contour(skel_resized, levels=[0.5], colors="red", linewidths=2)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return overlay

from skimage.morphology import remove_small_objects, remove_small_holes, dilation, erosion, disk

def nettoyer_masque_trottoir(mask_rgb,
                             min_obj_size=500,
                             min_hole_size=500,
                             smooth_radius=3):
    """
    mask_rgb : (H, W, 3) uint8
    Retourne un mask_rgb 'nettoyé' sans modifier les couleurs
    des autres classes.
    """

    # 1) détecter le trottoir (bleu) -> masque bool 2D
    m = mask_rgb.astype(np.float32) / 255.0
    eps = 0.1
    trottoir = (m[..., 2] > 1 - eps) & (m[..., 0] < eps) & (m[..., 1] < eps)
    # trottoir : (H, W), bool

    # 2) supprimer petits bouts isolés
    trottoir_clean = remove_small_objects(trottoir, min_size=min_obj_size)

    # 3) boucher petits trous
    trottoir_clean = remove_small_holes(trottoir_clean, area_threshold=min_hole_size)

    # 4) lisser les bords avec un SE 2D (footprint 2D appliqué à image 2D)
    selem = disk(smooth_radius)
    trottoir_clean = erosion(trottoir_clean, selem)
    trottoir_clean = dilation(trottoir_clean, selem)

    # 5) appliquer ce masque nettoyé sur le RGB
    mask_rgb_clean = mask_rgb.copy()
    to_clear = trottoir & ~trottoir_clean
    mask_rgb_clean[to_clear] = (0, 0, 0)

    return mask_rgb_clean

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes, dilation, erosion, disk

def lisser_masque_trottoir(mask_rgb,
                           scale_factor=2.0,
                           blur_ksize=7,
                           blur_sigma=2.0,
                           min_obj_size=500,
                           min_hole_size=500,
                           smooth_radius=3):
    """
    Lisse et nettoie le trottoir dans un masque RGB:
      1) détecte le trottoir (bleu),
      2) interpole (upscale), floute, reseuille,
      3) nettoie petits objets / trous,
      4) applique au mask_rgb d'origine sans changer les couleurs des autres classes.

    mask_rgb : (H, W, 3) uint8
    Retour : mask_rgb_lisse
    """

    H, W = mask_rgb.shape[:2]

    # 1) détection du trottoir (bleu)
    m = mask_rgb.astype(np.float32) / 255.0
    eps = 0.1
    trottoir = (m[..., 2] > 1 - eps) & (m[..., 0] < eps) & (m[..., 1] < eps)  # bool (H, W)

    # si rien, on renvoie tel quel
    if trottoir.sum() == 0:
        return mask_rgb

    # 2) interpolation (upscale) + flou + seuil
    #    on travaille sur un float [0,1]
    trottoir_float = trottoir.astype(np.float32)

    # upscale
    new_W = int(W * scale_factor)
    new_H = int(H * scale_factor)
    trottoir_up = cv2.resize(trottoir_float, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    # flou gaussien pour lisser
    if blur_ksize % 2 == 0:
        blur_ksize += 1  # kernel doit être impair
    trottoir_blur = cv2.GaussianBlur(trottoir_up, (blur_ksize, blur_ksize), blur_sigma)

    # reseuil : on garde les zones où la proba est > 0.5
    trottoir_smooth_up = trottoir_blur > 0.5

    # redescendre à la taille d'origine
    trottoir_smooth = cv2.resize(
        trottoir_smooth_up.astype(np.uint8),
        (W, H),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # 3) nettoyage morphologique (optionnel mais utile)
    trottoir_clean = remove_small_objects(trottoir_smooth, min_size=min_obj_size)
    trottoir_clean = remove_small_holes(trottoir_clean, area_threshold=min_hole_size)

    selem = disk(smooth_radius)
    trottoir_clean = erosion(trottoir_clean, selem)
    trottoir_clean = dilation(trottoir_clean, selem)

    # 4) appliquer au masque RGB (on ne modifie que les anciens pixels de trottoir)
    mask_rgb_lisse = mask_rgb.copy()
    # pixels qui étaient trottoir mais ne le sont plus après lissage -> on les met à 0
    to_clear = trottoir & ~trottoir_clean
    mask_rgb_lisse[to_clear] = (0, 0, 0)

    return mask_rgb_lisse


def full_pipeline(image):
  # On calcule le masque.
  pred_finetuned = predict_mask(model_finetuned, processor, image)
  # Ici on peut effectuer un traitement pour améliorer le masque.

  # On transforme le masque en RGB.

  pred_finetuned_rgb = decode_segmap_global(pred_finetuned,dict_korean_color)
  pred_finetuned_rgb= lisser_masque_trottoir(pred_finetuned_rgb)

  #### Test entre le masque réel et le maque calculé.
  main_skel = trajectoire(pred_finetuned_rgb, 5)
  main_skel = main_skel[:-3]

  print("image shape =",image.shape)
  print("mask shape =",pred_finetuned_rgb.shape)
  print("main skel shape =",main_skel.shape)
  print(main_skel)
  #visu = overlay_mask2(image, pred_finetuned_rgb, alpha=0.6, main_skel=main_skel, show=True)

  visu = show_overlay_with_traj(image, pred_finetuned_rgb, main_skel, alpha=0.6)
  plt.imshow(visu); plt.axis("off")

  vecteur_dir = vecteur_directeur(main_skel)
  segmentation_box(image,processor_box,model_box,vecteur_dir)


if __name__ == "__main__":

    #path_image = r"C:\Users\rasou\Desktop\PER\DATA\Bbox_1_new\Bbox_0001\MP_SEL_000068.jpg"
    path_image = r"C:\Users\rasou\Desktop\PER\IMG_1518.jpg"
    image = plt.imread(path_image)
    image = image[:, :, :3]
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #dictionnaire des classes
    dict_korean_color = {0: (255, 128, 255), 1: (0, 0, 255), 2: (230, 170, 255), 3: (0, 0, 0), 4: (255, 155, 155), 5: (255, 255, 0), 6: (0, 255, 0)}

    ######################################################## - Chargment des modèles - ########################################################
    # Charger le modèle de segmentation bonding box
    processor_box = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model_box = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model_box.to(device)

    #lien vers le dossier qui contient le modèle pré-entrainé
    dossier_modele_fine_tune = r"C:\Users\rasou\Desktop\PER\Test_local\Modele_b2_segformer"

    #définition du processor qui traite les images avant de les envoyer au modèle
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")

    #import du modèle pré entrainé
    model_finetuned = SegformerForSemanticSegmentation.from_pretrained(dossier_modele_fine_tune).to(device)
    model_finetuned.eval()

    full_pipeline(image)
