import re
import os
import unidecode
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class research_space:
    def __init__(self, key):
        """
        Initiate the object. Key just for storing.
        """
        self.key = key
        self.pos = None


    def load(self, year):
        """
        Load cache.
        """
        self.year = year
        self.phi = np.load("__rscache__/{}_phi_{}.npy".format(self.key, self.year))
        self.x = np.load("__rscache__/{}_x_{}.npy".format(self.key, self.year), allow_pickle='TRUE').item()
        self.fields = np.load("__rscache__/{}_fields_{}.npy".format(self.key, self.year), allow_pickle='TRUE').tolist()
        self.scientists = np.load("__rscache__/{}_scientists_{}.npy".format(self.key, self.year), allow_pickle='TRUE').tolist()
        

    def read_articles(self, arq, sep=";sep;"):
        """
        Read arq, file with articles, separeted by sep.
        Assumes 4 columns.
         - pesq: researcher's id
         - ano:  year of publication
         - catg: paper's category
         - num:  number of authors
        """
        art = pd.read_csv(arq, sep=sep, engine="python")
        art = art[(art["ano"] != 'rint') & (art["ano"] != 'onic')]
        self.articles = art


    def set_institution(self, arq, sep=";sep;"):
        """
        Reads the institution of each researcher.
        Assumes 3 columns.
         - id_pesquisador:   researcher's id
         - nome_instituicao: institution's name
         - cep_instituicao:  institution's zip code
        """
        bio = pd.read_csv(arq, sep=sep, engine="python")
        bio = bio[bio["id_pesquisador"] != "None"]
        bio = bio.dropna(subset=['id_pesquisador'])
        bio = bio.fillna("none")
        bio["nome_instituicao"] = bio["nome_instituicao"].astype(str).apply(lambda x: self.__ops(x))
        bio["cep_instituicao"] = bio["cep_instituicao"].astype(str).apply(lambda x: re.sub("\-", "", x))

        s = set(self.scientists)

        inst = bio[["id_pesquisador", "nome_instituicao"]].set_index("id_pesquisador").to_dict()["nome_instituicao"]
        inst = {int(k): v for k, v in inst.items() if int(k) in s}

        id_cep = bio[["id_pesquisador", "cep_instituicao"]].set_index("id_pesquisador").to_dict()["cep_instituicao"]
        est = {int(k): self.__cep(v) for k, v in id_cep.items() if int(k) in s}

        self.inst = inst
        self.est = est


    def __ops(self, s):
        """
        String operations to reduce miss classification.
        """
        s = re.sub(r"\s?\(.*\)", "", s)
        s = unidecode.unidecode(s)
        s = s.lower()
        s = re.sub(",", " ", s)
        s = re.sub("\.", " ", s)
        s = re.sub(";", " ", s)
        s = re.sub("\s+", " ", s)
        s = s.strip()
    
        return s


    def __cep(self, n):
        n = str(n)[:5]
    
        if not all(c.isdigit() for c in n) or n == '':
            return "DESCONHECIDO"
    
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
        if n < 77000: return "14"
        if n < 78000: return "17"
        if n < 79000: return "51"
        if n < 80000: return "50"
        if n < 88000: return "41"
        if n < 90000: return "42"
        if n < 100000: return "43"


    def compute(self, year, threshold=0.1):
        """
        Compute the research space. It takes a lot of time.
        """
        self.year = year
        self.__M(year, threshold)
        indices = dict()
        sums = np.zeros(len(self.m))
    
        for sf in self.p:
            if sf[1] not in indices:
                indices[sf[1]] = of.index(sf[1])
            sums[indices[sf[1]]] += 1

        phi = self.m.copy()
        for i in range(len(self.m)):
            phi[:,i] /= sums[i]
    
        self.phi = phi
        self.__store(year)

    
    def __store(self):
        """
        Store cache.
        """
        if not os.path.isdir("__rscache__"):
            os.mkdir("__rscache__")

        np.save("__rscache__/{}_phi_{}.npy".format(self.key, self.year), self.phi)
        np.save("__rscache__/{}_x_{}.npy".format(self.key, self.year), self.x)
        np.save("__rscache__/{}_fields_{}.npy".format(self.key, self.year), self.fields)
        np.save("__rscache__/{}_scientists_{}.npy".format(self.key, self.year), self.scientists)


    def __catg(self,s):
        """
        Get all the categories from an article
        """
        return [re.sub(r"\s?\(Q[1-9]\)", "", x).strip().lower() \
            for x in s.split(";")]


    def __X(t):
        """
        Create a dict to represent the X matrix
        """
        x = dict()
        for _, row in self.articles.iterrows():
            if int(row["ano"]) < t:
                fs = self.__catg(row["catg"])
                nf = len(fs)
                for field in fs:
                    if row["num"] == 0:
                        continue
                
                    if (row["pesq"], field) in x:
                        x[(row["pesq"], field)] += 1/(nf * row["num"])
                    else:
                        x[(row["pesq"], field)] = 1/(nf * row["num"])
        self.x = x


    def __P(t):
        """
        Create a dict to represent the P matrix
        """
        self.__X(t)
        p = dict()
    
        for sf in self.x:
            if self.x[sf] > threshold:
                p[sf] = 1
        self.p = p


    def __M(t, threshold):
        """
        Create the M matrix
        """
        self.__P(t, threshold)
        s = set()
        f = set()
    
        for sf in self.p:
            s.add(sf[0])
            f.add(sf[1])
        
        of = sorted(list(f))
        os = sorted(list(s))

        indices = {u: v for v, u in enumerate(of)}
        n = len(of)
        m = np.zeros((n,n))
    
        for i in range(n):
            for j in range(i+1, n):
                for k in s:
                    if (k,of[i]) in p and (k,of[j]) in p:
                        m[i,j] += 1
                        m[j,i] += 1
        
        self.m = m
        self.fields = of
        self.scientists = os

    
    def plot(self, values=None, labels=None, pos=None, new=False, threshold=0.212):
        """
        Plot the research space
        """
        plt.rcParams["figure.figsize"] = (15,10)
        G = nx.from_numpy_matrix(self.phi)
        mast = nx.maximum_spanning_tree(G)

        n = len(self.phi)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.phi[i,j] > threshold:
                        mast.add_edge(i,j)

        f = plt.figure(1)
        ax = f.add_subplot(1,1,1)

        if pos == None:
            if self.pos == None or new:
                self.pos = nx.spring_layout(mast)
        else:
            self.pos = pos 
        pos = self.pos
            
        if values != None:  
            cm, scalarMap = self.__colors(values)

        if labels != None:
            for lab in labels:
                ax.plot([0],[0], color = scalarMap.to_rgba(labels[lab]),
                    label=lab, lw=7)
        plt.legend(loc='upper left')

        nx.draw_networkx(mast, pos, cmap=cm, vmin=0,
            vmax= max(values), node_color=values,
            with_labels=False,ax=ax, node_size=self.__size(),
            edge_size=1, edge_color="lightgray")

        plt.axis('off')
        f.set_facecolor('w')
        plt.show()


    def __colors(self, vals):
        if any(isinstance(x, float) for x in vals):
            cm = plt.get_cmap('hot')
        else:    
            cm = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=max(vals))
        scalar_map = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        return [cm, scalar_map]


    def __size(self):
        values = [40 for node in self.fields]
        values[self.fields.index("computer science applications")] = 300
        return values
