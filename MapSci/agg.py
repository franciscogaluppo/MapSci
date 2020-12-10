import re
import unidecode
import pandas as pd
from collections import defaultdict

def __ops(s):
    s = re.sub(r"\s?\(.*\)", "", s)
    s = unidecode.unidecode(s)
    s = s.lower()
    s = re.sub(",", " ", s)
    s = re.sub("\.", " ", s)
    s = re.sub(";", " ", s)
    s = re.sub("\s+", " ", s)
    s = s.strip()
    return s


def __cep(n):
    n = re.sub("\-", "", str(n))[:5]

    if not all(char.isdigit() for char in n) or n == '':
        return "Unknown"

    n = int(n)
    if n < 20000: return "35"
    if n < 29000: return "33"
    if n < 30000: return "32"
    if n < 40000: return "31"
    if n < 49000: return "29"
    if n < 50000: return "28"
    if n < 57000: return "26"
    if n < 58000: return "27"
    if n < 59000: return "25"
    if n < 60000: return "24"  
    if n < 64000: return "23"
    if n < 65000: return "22"
    if n < 66000: return "21"
    if n < 68900: return "15"
    if n < 69000: return "16"
    if n < 69300: return "13"
    if n < 69400: return "14"
    if n < 69500: return "12"
    if n < 70000: return "13"
    if n < 73700: return "53"
    if n < 76800: return "52"
    if n < 77000: return "11"
    if n < 78000: return "17"
    if n < 79000: return "51"
    if n < 80000: return "50"
    if n < 88000: return "41"
    if n < 90000: return "42"
    if n < 100000: return "43"


def get_insts(scientists, arq, sep=";sep;"):
    """
    Reads the institution of each researcher.

    Parameters
    ----------
    arq : str
        File containing the institutions data
    
	sep : str
        Files separator string
    """
    bio = pd.read_csv(arq, sep=sep, engine="python")
    bio.columns = ["id", "inst", "cep"]
    bio = bio[bio["id"] != "None"]
    bio = bio.dropna(subset=['id'])
    bio = bio.fillna("none")
    bio["inst"] = bio["inst"].astype(str).apply(lambda x: __ops(x))
    bio["cep"] = bio["cep"].astype(str).apply(lambda x: __cep(x))

    s = set(scientists)

    inst = bio[["id", "inst"]].set_index("id").to_dict()["inst"]
    inst = {int(k): v for k, v in inst.items() if int(k) in s}

    cep = bio[["id", "cep"]].set_index("id").to_dict()["cep"]
    st = {int(k): v for k, v in cep.items() \
            if int(k) in s and v != "Unknown"}

    return [inst, st]


def aggregate(x, level):
    """
	Creates a new papers object aggragating the papers
	by groups (institutions, states etc.).

	Parameters
	----------
	x : papers object
		Original data as papers object.
	
	level : dict
		Dictionary for mapping from scientist into the 
		its respective grouping.
    """
    new_x = defaultdict(int)
    for sf in x:
        if sf[0] in level:
            new_x[(level[sf[0]], sf[1])] += x[sf]
    return new_x
    
