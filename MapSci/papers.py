import os
import re
import functools
import numpy as np
import pandas as pd
from collections import defaultdict

class papers:
    """
    Given a dataset of publication records, this class preprocess it,
    store it and write it on disk for future research space prediction
    tasks. We assume the dataset  has 4 columns:
        1. ```pesq```: The researcher id;
        2. ```ano```: The year of publication;
        3. ```catg```: A list of research areas, separated by semicolons;
        4. ```num```: The number of authors in this publication.
    
    Attributes
    ----------
    key : str
        Data key for storing and loading purposes.

    Methods
    -------
    compute(arq, sep (optional))
        It reads the dataset arq, computes the X matrix (dict actually)
        and writes it on disk.

    count()

    presence()

    yearly()

    yearly_lazy() 
    """

    def __init__(self, key, macro=None, ):
        """
        Initiates the object

        Parameters
        ----------
        key : str
            Data key for storing and loading purposes.

        macro : dict
            Keys are scientists and values are grouping names
            (institutions, states etc). 
        """

        self.key = key
        self.__macro = macro

        try:
            self.__x = np.load("__rscache__/papers/{}.npy".format(key),
                    allow_pickle='TRUE').item()
            self.__loaded = True
        except:
            self.__loaded = False
            print("File not available: use the compute function.")


    def __set_entries(self, arq, sep):
        art = pd.read_csv(arq, sep=sep, engine="python")
        art = art[(art["ano"] != 'rint') & (art["ano"] != 'onic')]
        art = art[art["num"] > 0]
        art["ano"] = pd.to_numeric(art["ano"])
        self.__entries = art

    
    def compute(self, arq, sep=";sep;"):
        """
        Reads arq, file with articles, separeted by sep and then
        compute the complete X matrix.

        Parameters
        ----------
        arq : str
            File containing the articles data
        sep : str
            Files separator string
        """
        if self.__loaded:
            print("Already on memory.")
            return
        
        self.__set_entries(arq, sep)
        self.__X()
        self.__store()
        self.__loaded = True


    def __catg(self, s):
        catgs = [re.sub(r"\s?\(Q[1-9]\)", "", x).strip().lower() \
            for x in s.split(";")]

        if self.__macro != None:
            catgs_macro = set()
            for i in catgs:
                catgs_macro.add(self.__macro[i])
            return catgs_macro

        return catgs
        

    def __X(self):
        dd_int = functools.partial(defaultdict, int)
        x = defaultdict(dd_int)
        for _, row in self.__entries.iterrows():
            fs = self.__catg(row["catg"])
            nf = len(fs)
            for field in fs:
                x[row["ano"]][(row["pesq"], field)] += 1/(nf * row["num"])
        self.__x = x


    def __store(self):
        if not os.path.isdir("__rscache__/papers"):
            os.mkdir("__rscache__/papers")

        np.save("__rscache__/papers/{}.npy".format(self.key), self.__x)


    def count(self, init, end):
        """
        """
        if not self.__loaded:
            print("File not available: use the compute function.")
            return

		# Should I use a segtree?
        x = defaultdict(int)
        for year in self.__x:
            if year >= init and year <= end:
                for sf in self.__x[year]:
                    x[sf] += self.__x[year][sf]

        return x


    def presence(self, init, end, threshold=0.1, x=None):
        """
        Create a dict to represent the P matrix

        Parameters
        ----------

        Returns
        -------
        """
        if not self.__loaded:
            print("File not available: use the compute function.")
            return

        p = set()
        if x is None:
            x = self.count(init, end) 
        for sf in x:
            if x[sf] > threshold:
                p.add(sf)
        
        return [p, x]


    def yearly(self):
        """
        """
        try:
            return self.__yearly
        except:
            data = defaultdict(int)
            for year in self.__x:
                for sf in self.__x[year]:
                    data[year] += self.__x[year][sf]
            order = sorted(data.keys())
            vals = [list(), list()]
            for i in range(len(order)):
                vals[0].append(order[i])
                vals[1].append(data[order[i]])
            self.__yearly = vals
            return vals


    def yearly_lazy(self):
        try:
            return self.__yearly_lazy
        except:
            publi = defaultdict(int)
            author = defaultdict(set)
            dist = defaultdict(int)
            for _, row in self.__entries.iterrows():
                publi[row["ano"]] += 1
                dist[row["pesq"]] += 1
                author[row["ano"]].add(row["pesq"])
            
            author = {k: len(author[k]) for k in author}
            comp = [publi, author]
            new = list()

            for data in comp:
                order = sorted(data.keys())
                vals = [list(), list()]
                for i in range(len(order)):
                    vals[0].append(order[i])
                    vals[1].append(data[order[i]])
                new.append(vals)

            new += [dist]
            self.__yearly_lazy = new
            return new
        
