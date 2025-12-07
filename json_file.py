import xml.etree.ElementTree as ET
import json

def xml_to_json(xml_path, json_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Liste des annotations (une entrée par image)
    annotations = []

    for image in root.findall("image"):
        img_name = image.get("name")
        width = int(image.get("width"))
        height = int(image.get("height"))

        objects = []

        for poly in image.findall("polygon"):
            label = poly.get("label")
            points_str = poly.get("points")

            attribute_tag = poly.find("attribute")

            if attribute_tag is not None and attribute_tag.text is not None:
                attr_name = attribute_tag.text.strip()
                attr_value = attribute_tag.text
            else:
                attr_name = None
                attr_value = None

            #attr_name = attribute_tag.get("name")

            # Convertir "x,y;x,y;..." → [[x, y], [x, y], ...]
            points = []
            for p in points_str.split(";"):
                x, y = map(float, p.split(","))
                points.append([x, y])

            objects.append({
                "label":  f"{label} {attr_name}" if attr_name else label,
                "points": points
            })

        annotations.append({
            "image": img_name,
            "width": width,
            "height": height,
            "objects": objects
        })

    # Sauvegarde JSON
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=4)

    print(f"✔️ Sauvegarde terminée : {json_path}")


# -----------------------------
# Appel de la fonction
# -----------------------------

#xml_to_json(r"DATA\Polygon_10_new\Polygon_1201\P1026_09.xml", "annotations.json")

xml_to_json(r"DATA/Surface_1/Surface_001/15_SM0915_01.xml", "annotations_surface.json")
