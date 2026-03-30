"""

import pandas as pd
import random

disease_templates = [
    "The {crop} crop shows {symptom} under {condition}.",
    "{crop.capitalize()} plants exhibit {symptom} affecting overall growth.",
    "Field inspection shows {symptom} in {crop} plants after {condition}."
]

healthy_templates = [
    "The {crop} crop is healthy with normal growth and green foliage.",
    "{crop.capitalize()} plants show strong development and no visible issues.",
    "Field observation indicates healthy {crop} growth under current conditions."
]

symptoms = [
    "leaf yellowing and fungal spots",
    "wilting and weak stem development",
    "powdery growth on leaves",
    "reduced flowering and leaf damage"
]

conditions = [
    "high humidity",
    "continuous rainfall",
    "warm climatic conditions",
    "recent irrigation cycles"
]

crops = [
    "rice","maize","chickpea","kidney bean","pigeon pea","moth bean",
    "mung bean","black gram","lentil","pomegranate","banana","mango",
    "grapes","watermelon","muskmelon","apple","orange","papaya",
    "coconut","cotton","jute","coffee"
]

data = []

for _ in range(250):
    crop = random.choice(crops)
    if random.random() > 0.5:
        text = random.choice(disease_templates).format(
            crop=crop,
            symptom=random.choice(symptoms),
            condition=random.choice(conditions)
        )
        label = 1
    else:
        text = random.choice(healthy_templates).format(crop=crop)
        label = 0

    data.append({"text": text, "label": label})

df = pd.DataFrame(data)
df.to_csv("farmer_disease_reports_expanded.csv", index=False)


"""