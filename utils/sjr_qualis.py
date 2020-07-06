import unidecode
import numpy as np
import pandas as pd
import re

qualis = pd.read_csv("../dataset/qualis/qualis_2013-2016.csv",
    encoding="latin-1", sep=",", header=0,
    usecols=["Título","Área de Avaliação","Estrato"])

artigos = pd.read_csv("../dataset/lattes/artigos.csv", sep=";sep;",
        usecols=["id_pesquisador", "ano_publicacao",
        "journal_ou_conferencia", "numero_pesquisadores"],
        header=0, engine='python')
artigos = artigos.dropna()

def ops(s):
    s = re.sub(r"\s?\(.*\)", "", s)
    s = unidecode.unidecode(s)
    s = s.lower()
    s = re.sub("\s+", " ", s)
    s = s.strip()
    
    return s

def splt(s):
    chars = [".", ";", ":", "/", "-"]
    subs = [s]
    
    for c in chars:
        aux = list()
        for s in subs:
            aux += s.split(c)
        subs = aux
        
    return subs

qcatg = qualis["Área de Avaliação"].to_list()
qtitl = qualis["Título"].to_list()
qlist = set([ops(x) for x in qtitl])
lattes_set = [ops(x) for x in artigos["journal_ou_conferencia"].to_list()]

count = len([1 for x in lattes_set if x in qlist])
print(count/len(lattes_set))
print(len(set(qcatg)))
