# medial_axis.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
cimport cython

# Importation Python standard pour les bibliothèques externes
import skimage.morphology as morpho
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# On peut aussi faire des alias pour être plus clair
from skimage.morphology import medial_axis, dilation, disk
from libc.math cimport sqrt
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)




def extract_main_skeleton_branch(float[:, :] mask_centred):
    # 1. Calcul du squelette (on garde Scikit-Image ici car c'est déjà optimisé)
    # Note: j'utilise mask_centred comme passé en argument
    skel, _ = medial_axis(np.asarray(mask_centred), return_distance=True)

    # Coordonnées des pixels (y, x)
    cdef int[:] y_coords, x_coords
    y_raw, x_raw = np.where(skel)
    y_coords = y_raw.astype(np.int32)
    x_coords = x_raw.astype(np.int32)

    cdef int N = y_coords.shape[0]
    if N == 0:
        return np.zeros_like(skel, dtype=bool)

    # 2. Création d'une Lookup Table (LUT) pour remplacer le dictionnaire
    # On stocke l'index du point (+1 pour différencier du vide 0)
    cdef int height = skel.shape[0]
    cdef int width = skel.shape[1]
    cdef int[:, :] lut = np.zeros((height, width), dtype=np.int32) - 1

    cdef int i, j, v
    for i in range(N):
        lut[y_coords[i], x_coords[i]] = i

    # 3. Construction du graphe (Voisinage 8)
    # On pré-alloue des tableaux pour éviter les .append()
    cdef int[:] rows = np.zeros(N * 8, dtype=np.int32)
    cdef int[:] cols = np.zeros(N * 8, dtype=np.int32)
    cdef float[:] data = np.ones(N * 8, dtype=np.float32)
    cdef int edge_count = 0

    cdef int py, px, qy, qx, dy, dx, neighbor_idx

    for i in range(N):
        py = y_coords[i]
        px = x_coords[i]

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue

                qy = py + dy
                qx = px + dx

                if 0 <= qy < height and 0 <= qx < width:
                    neighbor_idx = lut[qy, qx]
                    if neighbor_idx != -1:
                        rows[edge_count] = i
                        cols[edge_count] = neighbor_idx
                        edge_count += 1

    # On tronque les tableaux à la taille réelle des connexions trouvées
    graph = csr_matrix((np.asarray(data[:edge_count]),
                        (np.asarray(rows[:edge_count]), np.asarray(cols[:edge_count]))),
                       shape=(N, N))

    # 4. Points extrêmes
    cdef int y_min = np.min(y_raw)
    cdef int y_max = np.max(y_raw)
    top_points = np.where(y_raw == y_min)[0].astype(np.int32)
    bottom_points = np.where(y_raw == y_max)[0].astype(np.int32)

    # 5. Chemin le plus court
    dist_matrix, predecessors = shortest_path(
        graph, directed=False, return_predecessors=True
    )

    # 6. Recherche du meilleur chemin (Plus profond)
    cdef int best_s = -1, best_t = -1
    cdef float best_len = -1.0
    cdef float d

    # Conversion pour accès rapide
    cdef float[:, :] d_mat = dist_matrix.astype(np.float32)
    cdef int[:, :] p_mat = predecessors.astype(np.int32)

    for s in top_points:
        for t in bottom_points:
            d = d_mat[s, t]
            if d > best_len: # d est infini si pas de chemin
                best_len = d
                best_s = s
                best_t = t

    # 7. Reconstruction du masque final
    main_skel = np.zeros((height, width), dtype=bool)
    cdef int cur
    if best_s != -1:
        cur = best_t
        while cur != -9999 and cur != best_s:
            main_skel[y_coords[cur], x_coords[cur]] = True
            cur = p_mat[best_s, cur]
        if cur == best_s:
            main_skel[y_coords[best_s], x_coords[best_s]] = True

    return main_skel


@cython.boundscheck(False)
@cython.wraparound(False)
def trajectoire(unsigned char[:, :, :] mask, int dist_secu):
    cdef int h = mask.shape[0]
    cdef int w = mask.shape[1]
    cdef int i, j

    # 1. Création du masque de couleur (Optimisé en C)
    # On travaille avec des entiers pour la comparaison de couleur
    cdef np.ndarray[np.uint8_t, ndim=2] mask_filtre_np = np.zeros((h, w), dtype=np.uint8)
    cdef unsigned char[:, :] mask_filtre = mask_filtre_np

    for i in range(h):
        for j in range(w):
            # Sélection Bleu (0,0,255) ou Mauve (230,170,255)
            if (mask[i, j, 0] == 244 and mask[i, j, 1] == 35 and mask[i, j, 2] == 232) or \
               (mask[i, j, 0] == 230 and mask[i, j, 1] == 170 and mask[i, j, 2] == 255):
                mask_filtre[i, j] = 1

    # 2. Trouver le centre pour le masque centré
    # On utilise NumPy ici car c'est une opération globale rapide
    rows, cols = np.where(mask_filtre_np)
    if cols.size == 0:
        return np.zeros((h, w), dtype=bool)

    cdef int center_col = (cols.min() + cols.max()) // 2
    cdef int half_width = 7

    # 3. Masque centré
    cdef np.ndarray[np.float32_t, ndim=2] mask_centred_np = np.zeros((h, w), dtype=np.float32)
    cdef float[:, :] mask_centred = mask_centred_np

    cdef int c_start = max(0, center_col - half_width)
    cdef int c_end = min(w, center_col + half_width)

    for i in range(h):
        for j in range(c_start, c_end):
            if mask_filtre[i, j] == 1:
                mask_centred[i, j] = 1.0 # Déjà normalisé (équivalent /255)

    # 4. Calcul du squelette (Appel Scikit-Image)
    # Note: On utilise mask_filtre_np (booléen pour medial_axis)
    skel, distance = medial_axis(mask_filtre_np, return_distance=True)

    # 5. Extraction de la branche principale (via ta fonction Cython)
    # On lui passe le memoryview float32
    main_skel = extract_main_skeleton_branch(mask_centred)

    return main_skel

def check_collisions(float[:, :] main_skel, list obstacles, int dist_secu=10):
    """
    Vérifie si le squelette de trajectoire intersecte une bounding box
    en prenant en compte une distance de sécurité.
    """
    # 1. On déclare TOUTES les variables au début
    cdef int h = main_skel.shape[0]
    cdef int w = main_skel.shape[1]
    cdef int r, c, x1, y1, x2, y2, obj_id
    cdef int safe_x1, safe_y1, safe_x2, safe_y2
    cdef bint collision_found # bint est un booléen efficace en C

    collisions = []

    # 2. Ensuite on exécute le code
    for obs in obstacles:
        obj_id = obs["id"]
        # bbox est au format (x1, y1, x2, y2)
        x1, y1, x2, y2 = obs["bbox"]

        # On calcule les zones de sécurité
        safe_x1 = max(0, x1 - dist_secu)
        safe_y1 = max(0, y1 - dist_secu)
        safe_x2 = min(w - 1, x2 + dist_secu)
        safe_y2 = min(h - 1, y2 + dist_secu)

        collision_found = False

        # On parcourt la zone
        for r in range(safe_y1, safe_y2 + 1):
            for c in range(safe_x1, safe_x2 + 1):
                if main_skel[r, c] > 0:
                    collision_found = True
                    break
            if collision_found:
                break

        if collision_found:
            collisions.append(obj_id)

    return collisions

def generer_alertes_collision(ids_en_conflit, id2label):
    """
    Transforme les IDs de collision en messages d'alerte lisibles.
    """
    if not ids_en_conflit:
        return ["Trajectoire libre"]

    messages = []
    # On utilise un set pour éviter de répéter le même message
    # si plusieurs instances du même objet (ex: 2 poteaux) causent une collision
    uniques_conflits = set(ids_en_conflit)

    for cid in uniques_conflits:
        # Récupération du nom de l'objet (ex: "Le Véhicule")
        nom_objet = id2label.get(cid, "un obstacle inconnu")

        # Formatage du message
        message = f"Attention : Collision possible avec {nom_objet} !"
        messages.append(message)

    return messages







def vecteur_directeur(float[:, :] main_skel):
    cdef int rows = main_skel.shape[0]
    cdef int cols = main_skel.shape[1]

    cdef float x_top = -1, y_top = -1
    cdef float x_bot = -1, y_bot = -1
    cdef int r, c

    # 1. Trouver le point de DESTINATION (le plus haut dans l'image, y minimal)
    # On parcourt du haut vers le bas
    for r in range(rows):
        for c in range(cols):
            if main_skel[r, c] > 0:
                y_top = <float>r
                x_top = <float>c
                break
        if x_top != -1: break

    # 2. Trouver le point de DÉPART (le plus proche de l'utilisateur, y maximal)
    # On parcourt du bas vers le haut
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            if main_skel[r, c] > 0:
                y_bot = <float>r
                x_bot = <float>c
                break
        if x_bot != -1: break

    # 3. Calcul du vecteur : Destination (Top) - Départ (Bot)
    cdef float vx = 0.0
    cdef float vy = 0.0
    cdef float norme = 0.0
    cdef float[:] res = np.zeros(2, dtype=np.float32)

    if x_top != -1 and x_bot != -1:
        # Vecteur qui monte dans l'image
        vx = x_top - x_bot
        vy = y_top - y_bot  # Sera négatif car y_top (ex: 10) < y_bot (ex: 120)

        norme = sqrt(vx*vx + vy*vy)

        if norme > 0:
            res[0] = vx / norme
            res[1] = vy / norme

    return np.asarray(res)