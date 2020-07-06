import unidecode
import numpy as np
import pandas as pd
import re

sjr = pd.read_csv("../dataset/SJR/scimagojr completo.csv",
    sep=";", header=0, usecols=["Title","Categories"])

artigos = pd.read_csv("../dataset/lattes/artigos.csv", sep=";sep;",
    usecols=["id_pesquisador", "ano_publicacao", "journal_ou_conferencia",
    "numero_pesquisadores"], header=0, engine='python')
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

sjr_catg = sjr["Categories"].to_list()
sjr_titl = sjr["Title"].to_list()
sjr_list = [[ops(x) for x in splt(string)] for string in sjr_titl]
lattes_list =[ops(x) for x in artigos["journal_ou_conferencia"].to_list()]

buffer = dict()
with open("../dataset/lattes_categories.csv", "w") as arq:
    arq.write("pesq;sep;ano;sep;catg;sep;num\n")
    for i in range(len(lattes_list)):
        if i%10000 == 0:
            print(i/len(lattes_list))
        
        title = lattes_list[i]
        lattes = artigos.iloc[i]
        if lattes[0] == "None":
            continue
        if title in buffer:
            arq.write("{};sep;{};sep;{};sep;{}\n".format(lattes[0],
                lattes[1], buffer[title], lattes[3]))
            continue
        
        strings = splt(title) + [title]
        for s in strings:
            for j in range(len(sjr_list)):
                if s in sjr_list[j]:
                    arq.write("{};sep;{};sep;{};sep;{}\n".format(
                        lattes[0], lattes[1], sjr_catg[j], lattes[3]))
                    buffer[title] = sjr_catg[j]
                    break
            else:
                continue
            break
