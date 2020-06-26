import os
import subprocess
import numpy as np
from collections import defaultdict

import networkx as nx
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from MapSci.curved_edges import curved_edges

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
    compute(model, init, end, threshold)
        Computes the research space, i.e., the phi matrix, for a
        specific year and threshold. It takes a lot of time

    plot(values, labels, pos, new, threshold)
        Plots the research space.
    """

    def __init__(self, key, x):
        """
        Initiates the object

        Parameters
        ----------
        key : str
            Used for storing and loading purposes
        """
        self.__x = x
        self.key = key
        self.pos = None
        self.phi = dict()

    def __store(self, trial):
        """
        """
        if not os.path.isdir("__rscache__/spaces"):
            os.mkdir("__rscache__/spaces")

        np.save("__rscache__/spaces/" + trial +\
                self.key+".npy", self.phi[trial])



    def compute(self, init, end, model="all", threshold=0.1, N=200):
        """
        Computes the research space. It takes a lot of time.

        Parameters
        ----------
        """
        self.__parameters = (init, end, threshold, N)
        self.__p = self.__x.presence(init, end, threshold)[0]
        self.__setup()
        if model == "guevara":
            self.__guevara()
        elif model == "chinazzi":
            self.__chinazzi()
        elif model == "all":
            self.__guevara()
            self.__chinazzi()
        else:
            print("Not a valid model.")
            return


    def __setup(self):
        """
        """
        f = set()
        for sf in self.__p:
            f.add(sf[1])
       
        of = sorted(list(f))
        self.__n = len(of)
        self.__indices = {u: v for v, u in enumerate(of)}
        if not os.path.isdir("__rscache__"):
            os.mkdir("__rscache__")
        

    def __guevara(self):
        """
        """
        trial = "guevara{}".format(self.__parameters[:3])
        print(trial)
        try:
            if trial not in self.phi:
                self.phi[trial] = np.load("__rscache__/spaces/" +\
                    trial + self.key +".npy", allow_pickle='TRUE')
            else:
                return
        except:
            n = self.__n
            m = np.zeros((n,n))
            sums = np.zeros(n)
            s_fields = defaultdict(list)

            for sf in self.__p:
                sums[self.__indices[sf[1]]] += 1
                s_fields[sf[0]].append(self.__indices[sf[1]])

            for s in s_fields:
                l = len(s_fields[s])
                for i in range(l):
                    for j in range(i+1, l):
                        m[s_fields[s][i], s_fields[s][j]] += 1
                        m[s_fields[s][j], s_fields[s][i]] += 1

            for i in range(n):
                m[:,i] /= sums[i]
    
            self.phi[trial] = (m, self.__indices)
            self.__store(trial)


    def __chinazzi(self):
        """
        StarSpace model
        """
        trial = "chinazzi{}".format(self.__parameters)
        print(trial)
        try:
            if trial not in self.phi:
                self.phi[trial] = np.load("__rscache__/spaces/" +\
                    trial + self.key +".npy", allow_pickle='TRUE')
            else:
                return
        except:
        
            B = defaultdict(set)
            for sf in self.__p:
                B[sf[0]].add(sf[1].replace(' ', '|'))

            name = "__rscache__/"+trial+".txt"
            with open(name, 'w') as f:
                for s in B:
                    for fi in B[s]:
                        f.write("__label__{} ".format(fi))
                    f.write("\n")

            N = self.__parameters[-1]
            out = "__rscache__/"+trial+"-out"
            cmd = ["../MapSci/Starspace/starspace", "train", "-trainFile",
                name, "-model", out, "-dim", str(N), "-trainMode", "1"]
            p = subprocess.Popen(cmd)
            p.wait()

            emb = dict()
            model = out+".tsv"
            with open(model, 'r') as f:
                next(f)
                for line in f:
                    vals = line.split("\t")
                    key = vals[0][9:].replace('|',' ')
                    emb[key] = np.array(vals[1:]).astype(np.float)

            norms = {x:np.linalg.norm(emb[x]) for x in emb}

            n = self.__n
            phi = np.zeros((n,n))
            fields = set(self.__indices.keys())
            for f1 in self.__indices:
                fields.remove(f1)
                i = self.__indices[f1]
                for f2 in fields:
                    j = self.__indices[f2]
                    p = np.dot(emb[f1], emb[f2])/(norms[f1]*norms[f2])
                    p = max(0, p)
                    phi[i,j] = p
                    phi[j,i] = p

            self.phi[trial] = (phi, self.__indices)
            self.__store(trial)
            os.remove(name)
            os.remove(out)
            os.remove(model)


    
    def plot(self, trial, values=None, labels=None, pos=None, new=False, threshold=0.212):
        """
        Plot the research space
        """
        plt.rcParams["figure.figsize"] = (10,7)
        phi = self.phi[trial][0]
        G = nx.from_numpy_matrix(phi)
        mast = nx.maximum_spanning_tree(G)

        n = len(phi)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if phi[i,j] > threshold:
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

        curves = curved_edges(mast, pos)
        lc = LineCollection(curves, color='grey', alpha=0.1)
        plt.gca().add_collection(lc)

        nx.draw_networkx_nodes(mast, pos, cmap=cm, vmin=0,
            vmax= max(values), node_color=values,
            with_labels=False,ax=ax, node_size=self.__size())

        plt.axis('off')
        f.set_facecolor('w')
        plt.show()
        return pos


    def __colors(self, vals):
        if any(isinstance(x, float) for x in vals):
            cm = plt.get_cmap('hot')
        else:    
            #cm = plt.get_cmap('jet')
            cm = plt.get_cmap('viridis')
        cNorm = colors.Normalize(vmin=0, vmax=max(vals))
        scalar_map = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        return [cm, scalar_map]


    def __size(self):
        values = [40 for node in self.__indices]
        #values[self.fields.index("computer science applications")] = 300
        return values
