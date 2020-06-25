import os
import subprocess
import numpy as np

import networkx as nx
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from curved_edges import curved_edges

def __store(trial):
    """
    """
    if os.path.isdir("__rscache__/spaces"):
        os.mkdir("__rscache__/spaces")

    np.save("__rscache__/spaces/{}.npy".format(trial), self.phi[trial])


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


    def compute(self, init, end, model="all", threshold=0.1):
        """
        Computes the research space. It takes a lot of time.

        Parameters
        ----------
        """
        self.__parameters = (init, end, threshold)
        self.__p = self.__x.presence(init, end, threshold)
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


    def __guevara(self):
        """
        """
        trial = "guevara{}".format(self.__parameters)
        try:
            if trial not in self.phi:
                self.phi[trial] = np.load(
                    "__rscache__/spaces/" + trial +".npy")
            else:
                return
        except:
            s = set()
            f = set()
    
            for sf in self.__p:
                s.add(sf[0])
                f.add(sf[1])
        
            of = sorted(list(f))
            os = sorted(list(s))
            indices = {u: v for v, u in enumerate(of)}

            n = len(of)
            m = np.zeros((n,n))
            sums = np.zeros(n)
            s_fields = defaultdict(list)

            for sf in self.__p:
                sums[indices[sf[1]]] += 1
                s_fields[sf[0]].add(indices[sf[1]])

            for s in s_fields:
                l = len(s_fields[s])
                for i in range(l):
                    for j in range(i+1, l):
                        m[s_fields[s][i], s_fields[s][j]] += 1
                        m[s_fields[s][j], s_fields[s][i]] += 1

            for i in range(n):
                m[:,i] /= sums[i]
    
            self.phi[trial] = (m, of, os)
            __store(trial)


    def star_space(self, N=200, threshold=0.1):
        """
        StarSpace model
        """
        pesq_catg = {x:set() for x in self.scientists}
        for sf in self.x:
            if self.x[sf] > threshold:
                pesq_catg[sf[0]].add(sf[1].replace(' ', '|'))

        name = "__rscache__/{}_ssinput_{}_{}.txt".format(self.key, self.year, N)
        with open(name, 'w') as f:
            for s in pesq_catg:
                for c in pesq_catg[s]:
                    f.write("__label__{} ".format(c))
                f.write("\n")

        out = "__rscache__/{}_ssoutput_{}_{}".format(self.key, self.year, N)
        cmd = ["../Starspace/starspace", "train", "-trainFile",
                name, "-model", out, "-dim", str(N), "-trainMode", "1"]
        p = subprocess.Popen(cmd)
        p.wait()

        if p.returncode == 0:
            self.load_star_space(N)
        else:
            raise Exception("StarSpace error.")

    
    def load_star_space(self, N=200):
        """
        load StarSpace
        """
        emb = dict()
        model = "__rscache__/{}_ssoutput_{}_{}.tsv".format(self.key, self.year, N)
        with open(model, 'r') as f:
            next(f)
            for line in f:
                vals = line.split("\t")
                emb[vals[0][9:].replace('|',
                    ' ')] = np.array(vals[1:]).astype(np.float)

        norms = {x:np.linalg.norm(emb[x]) for x in emb}

        n = len(self.fields)
        phi = np.zeros((n,n))
        for i in range(n):
            f1 = self.fields[i]
            for j in range(i+1, n):
                f2 = self.fields[j]
                p = max(0, np.dot(emb[f1], emb[f2])/(norms[f1]*norms[f2]))
                phi[i,j] = p
                phi[j,i] = p

        self.phi = phi

    
    def plot(self, values=None, labels=None, pos=None, new=False, threshold=0.212):
        """
        Plot the research space
        """
        plt.rcParams["figure.figsize"] = (10,7)
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
        values = [40 for node in self.fields]
        #values[self.fields.index("computer science applications")] = 300
        return values
