import pandas as pd
import numpy as np

clss = dict()
clss[10] = "multidisciplinary"
clss[11] = "agricultural and biological sciences"
clss[12] = "arts and humanities"
clss[13] = "biochemistry, genetics and molecular biology"
clss[14] = "business, management and accounting"
clss[15] = "chemical engineering"
clss[16] = "chemistry"
clss[17] = "computer science"
clss[18] = "decision sciences"
clss[19] = "earth and planetary sciences"
clss[20] = "economics, econometrics and finance"
clss[21] = "energy"
clss[22] = "engineering"
clss[23] = "environmental science"
clss[24] = "immunology and microbiology"
clss[25] = "materials science"
clss[26] = "mathematics"
clss[27] = "medicine"
clss[28] = "neuroscience"
clss[29] = "nursing"
clss[30] = "pharmacology, toxicology and pharmaceutics"
clss[31] = "physics and astronomy"
clss[32] = "psychology"
clss[33] = "social sciences"
clss[34] = "veterinary"
clss[35] = "dentistry"
clss[36] = "health professions"

areas = pd.read_csv("../dataset/SJR/areas.txt", sep=";")
areas["Field"] = areas["Field"].apply(lambda x: x.strip().lower())
areas["Subject area"] = areas["Subject area"].apply(
    lambda x: x.strip().lower())
areas["Classification"] = areas["Code"].apply(
    lambda x: clss[int(str(x)[:2])])

areas = areas.append({'Field': 'e-learning',
    'Subject area': "social sciences & humanities",
    "Classification": "social sciences"} , ignore_index=True)
areas = areas.append({'Field': 'nanoscience and nanotechnology',
    'Subject area': "physical sciences",
    "Classification": "materials science"} , ignore_index=True)
areas = areas.append({'Field': 'social work',
    'Subject area': "social sciences & humanities",
    "Classification": "social sciences"} , ignore_index=True)
areas = areas.append({'Field': 'sports science',
    'Subject area': "health sciences",
    "Classification": "health professions"} , ignore_index=True)

areas.to_pickle("../dataset/SJR/areas.pkl")
