import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Charger les annotations JSON
with open("annotations_surface.json", "r") as f:
    data = json.load(f)

# ---------------------------
# Choisir l'image à afficher
# ---------------------------
image_name = "MP_SEL_SUR_000002.jpg"



image_path = r"DATA\Surface_1\Surface_001\MASK\MP_SEL_SUR_000002.png"

print("Je cherche :", image_name)
print("Images dans le JSON :", [item["image"] for item in data])


# Trouver l'entrée correspondante dans le JSON
entry = next(item for item in data if item["image"] == image_name)

# Charger l’image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

H, W = img.shape[:2]

x_0 = W // 2
y_0 = H

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img)
ax.axis("off")

# Tracer chaque polygone
for obj in entry["objects"]:
    label = obj["label"]
    points = obj["points"]

    # Créer et tracer le polygone
    polygon = patches.Polygon(points, closed=True,
                              fill=False,
                              edgecolor="yellow",
                              linewidth=2)
    ax.add_patch(polygon)

    x_mean, y_mean = np.mean(points, axis=0)

    line = FancyArrowPatch((x_0, y_0), (x_mean, y_mean),
                           color='blue',
                           arrowstyle='-')
    ax.add_patch(line)


    # Trouver une position pour afficher le label (premier point du polygon)
    x0, y0 = points[0]
    ax.text(x_mean, y_mean - 5, label,
            color="red",
            fontsize=12,
            weight="bold")

plt.show()

        # Classe → ID
class_to_idx = {
            'roadway': 0,
            'sidewalk': 1,
            'alley': 2,
            'background': 3,
            'bike_lane': 4,
            'braille_guide_blocks': 5,
            'caution_zone': 6
        }

        # RGB → ID
color_to_id = {
            (255,128,255): 0,
            (0,0,255): 1,
            (230,170,255): 2,
            (0,0,0): 3,
            (255,155,155): 4,
            (255,255,0): 5,
            (0,255,0): 6
        }
dictionnaire = {
                0:[(255,128,255),(255,0,255)],
                1:[(0,0,255),(217,217,217),(198,89,17),(128,128,128),(255,230,153),(55,86,35),(110,168,70)],
                2:[(230,170,255),(208,88,255),(138,60,200),(88,38,128)],
                3:[(0,0,0)],
                4:[(255,155,155)],
                5:[(255,255,0),(128,96,0)],
                6:[(255,192,0),(255,0,0),(0,255,0),(255,128,0),(105,105,255)]

}