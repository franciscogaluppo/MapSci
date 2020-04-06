import re
import os
import unidecode
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from collections import defaultdict


class research_space:
    """
    Stores the research space and perform prediction tasks

    Attributes
    ----------
    key : str
        Data key for storing and loading purposes
    pos : dict
        Positions keyed by node

    Methods
    -------
    load(year)
        Loads locally stored research space using key
    read_articles(arq, sep)
        Reads the articles data
    set_institutions(arq, sep)
        Reads each researcher's institute
    compute(year, threshold)
        Computes the research space, i.e., the phi matrix, for a
        specific year and threshold. It takes a lot of time
    advantages()
        Computes the Revealed Comparative Advantage for researches,
        institutions and states
    set_indicators()
        Computes the indicator matrices for future predictions
    predict(s, level, transition)
        Predicts which areas s will research or improve next
    plot(values, labels, pos, new, threshold)
        Plots the research space.
    """

    def __init__(self, key):
        """
        Initiates the object

        Parameters
        ----------
        key : str
            Used for storing and loading purposes
        """
        self.key = key
        self.pos = None


    def load(self, year):
        """
        Loads locally stored research space using key

        Parameters
        ----------
        year : int
            Specify which research space to load
        """
        self.year = year
        self.phi = np.load("__rscache__/{}_phi_{}.npy".format(self.key, self.year))
        self.x = np.load("__rscache__/{}_x_{}.npy".format(self.key, self.year), allow_pickle='TRUE').item()
        self.fields = np.load("__rscache__/{}_fields_{}.npy".format(self.key, self.year), allow_pickle='TRUE').tolist()
        self.scientists = np.load("__rscache__/{}_scientists_{}.npy".format(self.key, self.year), allow_pickle='TRUE').tolist()
        

    def read_articles(self, arq, sep=";sep;"):
        """
        Reads arq, file with articles, separeted by sep.

        Parameters
        ----------
        arq : str
            File containing the articles data
        sep : str
            Files separator string
        """
        art = pd.read_csv(arq, sep=sep, engine="python")
        art = art[(art["ano"] != 'rint') & (art["ano"] != 'onic')]
        self.articles = art


    def set_institution(self, arq, sep=";sep;"):
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

        Parameters
        ----------
        s : str
            The string object which the operations will act upon

        Returns
        -------
        str
            The resulting string
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
        """
        Giving a cep number (brazilian zip code) returns its state

        Parameters
        ----------
        n : int or str
            The cep number
        
        Returns
        -------
        str
            The state corresponding to the cep code
        """
        n = str(n)[:5]
    
        if not all(c.isdigit() for c in n) or n == '':
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
        if n < 77000: return "14"
        if n < 78000: return "17"
        if n < 79000: return "51"
        if n < 80000: return "50"
        if n < 88000: return "41"
        if n < 90000: return "42"
        if n < 100000: return "43"


    def compute(self, year, threshold=0.1):
        """
        Computes the research space. It takes a lot of time.

        Parameters
        ----------
        year : int
            Specifies the data limit. All papers before the parameter
            are considered
        threshold : float
            Minimum to have published to be considered. Considering
            the sum of fractions of the papers number of authors and
            number of subjects the journal has
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
        Stores locally the computed research space using key
        """
        if not os.path.isdir("__rscache__"):
            os.mkdir("__rscache__")

        np.save("__rscache__/{}_phi_{}.npy".format(self.key, self.year), self.phi)
        np.save("__rscache__/{}_x_{}.npy".format(self.key, self.year), self.x)
        np.save("__rscache__/{}_fields_{}.npy".format(self.key, self.year), self.fields)
        np.save("__rscache__/{}_scientists_{}.npy".format(self.key, self.year), self.scientists)


    def __catg(self, s):
        """
        Get all the categories from an article

        Parameters
        ----------
        s : string
            A string containing all the subjects
        """
        return [re.sub(r"\s?\(Q[1-9]\)", "", x).strip().lower() \
            for x in s.split(";")]


    def __X(self, t):
        """
        Create a dict to represent the X matrix
        
        Parameters
        ----------
        t : int
            
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


    def __P(self, t):
        """
        Create a dict to represent the P matrix
        """
        self.__X(t)
        p = dict()
    
        for sf in self.x:
            if self.x[sf] > threshold:
                p[sf] = 1
        self.p = p


    def __M(self, t, threshold):
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


    def advantages(self):
        """
        Compute revealed comparative advantage for researchers,
        institutions and states.
        """
        rca = defaultdict(int)
        rcai = defaultdict(int)
        rcae = defaultdict(int)

        xi = defaultdict(int)
        xe = defaultdict(int)

        sumf = defaultdict(int)
        sumfi = defaultdict(int)
        sumfe = defaultdict(int)

        sums = defaultdict(int)
        sumsf = 0

        s = set(self.scientists)

        for sf in self.x:
            if sf[0] not in s:
                continue

            ins = self.inst[sf[0]]
            est = self.est[sf[0]]
            
            xi[(ins, sf[1])] += self.x[sf]
            xe[(est, sf[1])] += self.x[sf]
            
            sumf[sf[0]] += self.x[sf]
            sumfi[ins] += self.x[sf]
            sumfe[est] += self.x[sf]
            
            sums[sf[1]] += self.x[sf]
            sumsf += self.x[sf]
    
        for sf in self.x:
            if sf[0] not in s:
                continue
            rca[sf] = (self.x[sf]/sumf[sf[0]])/(sums[sf[1]]/sumsf)

        for sf in xi:
            rcai[sf] = (xi[sf]/sumfi[sf[0]])/(sums[sf[1]]/sumsf)

        for sf in xe:
            rcae[sf] = (xe[sf]/sumfe[sf[0]])/(sums[sf[1]]/sumsf)

        self.rca_scientist = rca
        self.rca_institution = rcai
        self.rca_estate = rcae


    def set_indicators(self):
        """
        Compute the inicator matrices
        """

        Us = [defaultdict(list), defaultdict(list)]
        Ui = [defaultdict(list), defaultdict(list)]
        Ue = [defaultdict(list), defaultdict(list)]

        f = set(self.fields)

        #TODO: redundant information! if x >= 1 then x > 0
        #TODO: searches for the index twice
        for sf in self.rca_scientist:
            if sf[1] not in f:
                continue

            if self.rca_scientist[sf] > 0:
                Us[0][sf[0]].append(self.fields.index(sf[1]))
                if self.rca_scientist[sf] >= 1:
                    Us[1][sf[0]].append(self.fields.index(sf[1]))

        for sf in self.rca_institution:
            if sf[1] not in f:
                continue

            if self.rca_institution[sf] > 0:
                Ui[0][sf[0]].append(self.fields.index(sf[1]))
                if self.rca_institution[sf] >= 1:
                    Ui[1][sf[0]].append(self.fields.index(sf[1]))

        for sf in self.rca_estate:
            if sf[1] not in f:
                continue

            if self.rca_estate[sf] > 0:
                Ue[0][sf[0]].append(self.fields.index(sf[1]))
                if self.rca_estate[sf] >= 1:
                    Ue[1][sf[0]].append(self.fields.index(sf[1]))

        self._U = [Us, Ui, Ue]
        
        n = range(len(self.fields))
        self.norm = [sum(self.phi[i,j] for j in n) for i in n]    


    def predict(self, s, level, transition):
        """
        Predict s's future areas
        """

        if level == 'scientist':
            level = 0
        elif level == 'institution':
            level = 1
        elif level == 'estate':
            level = 2
        else:
            raise ValueError('{} is not a valid level.'.format(level))
        
        # TODO: different transitions
        if transition == 'inactive-active':
            transition = 0
        else:
            raise ValueError('{} is not a valid transition.'.format(transition))

        omega = list()
        for i in range(len(self.fields)):
            if i in self._U[level][0][s]:
                continue

            num = sum(self.phi[i,j] for j in self._U[level][0][s])
            div = np.round(num/self.norm[i], 5)
            if div > 0.0:
                omega.append((div, self.fields[i]))

        return sorted(omega, reverse=True)

    
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
