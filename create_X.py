import json

def describe_objects(json_input_path, json_output_path):
    # Charger le JSON existant
    with open(json_input_path, "r") as f:
        data = json.load(f)

    # Nouveau dictionnaire pour stocker les phrases
    descriptions = {}

    for entry in data:
        image_name = entry["image"]
        descriptions[image_name] = []

        for obj in entry["objects"]:
            label = obj["label"]
            points = obj["points"]
            # Créer la phrase descriptive
            phrase = f"L'objet '{label}' se trouve aux coordonnées {points}."
            descriptions[image_name].append(phrase)

    # Sauvegarder dans un nouveau JSON
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=4, ensure_ascii=False)

    print(f"✔️ Les descriptions ont été sauvegardées dans {json_output_path}")


# -----------------------------
# Exemple d'utilisation
# -----------------------------
describe_objects("annotations.json", "descriptions.json")

describe_objects("annotations_surface.json", "descriptions_surface.json")
