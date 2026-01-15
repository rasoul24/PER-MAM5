import matplotlib.pyplot as plt
import numpy as np
from medial_axis  import compute_main_skeleton

if __name__ == "__main__":

    path_mask = r"C:\Users\rasou\Desktop\PER\DATA\Surface_1\Surface_001\MP_SEL_SUR_000004.jpg"
    mask = plt.imread(path_mask)

     # Sélection du bleu pur (trottoir)
    m = mask.astype(np.float32) / 255.0

    colors = [
        (0,0,255),
        (230,170,255)
        ]

    # masque initial vide
    mask_filtre = np.zeros(m.shape[:2], dtype=bool)

    # ajout des couleurs
    for c in colors:
        mask_filtre |= (mask == c).all(axis=-1)

    mask_filtre = mask_filtre.astype(np.float32) / 255.0



    main_skel = compute_main_skeleton(mask_filtre)
    plt.imshow(main_skel)
    plt.show()
