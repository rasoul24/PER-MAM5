import numpy as np
import matplotlib.pyplot as plt
import glob
import xml.etree.ElementTree as ET

path = r"DATA\Polygon_10_new\Polygon_1201\*.png"


# Récupère tous les fichiers .jpg
images_chemin = glob.glob(path)

print(images_chemin)

#####################################################



# Charger le fichier XML
tree = ET.parse(r"DATA\Polygon_10_new\Polygon_1201\P1026_09.xml")

root = tree.getroot()

# Parcourir chaque image
for image in root.findall("image"):
    img_name = image.get("name")
    print("\nImage test :", img_name)

    # Parcourir tous les polygones de cette image
    for poly in image.findall("polygon"):
        label = poly.get("label")
        points_str = poly.get("points")

        # Convertir les points "x,y;x,y;..." en liste de tuples flottants
        points = []
        for p in points_str.split(";"):
            x, y = map(float, p.split(","))
            points.append((x, y))

        print(f"  Label : {label}")
        print(f"  Points : {points}")
