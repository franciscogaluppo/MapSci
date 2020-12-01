import os
import subprocess
from collections import defaultdict

import numpy as np
import networkx as nx

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection

from MapSci.curved_edges import curved_edges


class research_space:
    """
    Stores the research space and perform prediction tasks.
	Assumes there is a compiled Starspace algorithm accessible.
	For backbone extraction purposes, it requires R. Check the README
	for a list of R packages.

    Attributes
    ----------
    key : str
        Data key for storing and loading purposes.

    pos : dict
        Positions keyed by node.

	phi : dict
		Data structure to track all loaded models.

    Methods
    -------
    compute(model, init, end, threshold)
        Computes the research space, i.e., the phi matrix, for a
        specific time interval and threshold, given a model.

	get_backbone(trial, alpha)

    plot(values, labels, pos, new, threshold)
        Plots the research space, removing edges with weights smaller
		than the threshold. It uses colors according to the values.
    """

    def __init__(self, key, x):
        """
        Initiates the object

        Parameters
        ----------
        key : str
            A key string. Used for storing and loading purposes.
			As the proccess may take some time every execution,
			it makes sense to just store previous runs. Before running,
			it checks if it has been saved previously

		x : paper object
			The pre-proccessed data for creating the research spaces.
			See ```papers.py```.
        """
        self.__x = x
        self.key = key
        self.pos = None
        self.phi = dict()


    def compute(self, init, end, model="all", threshold=0.1, N=200):
        """
        Computes the research space, i.e., the phi matrix, for a
        specific time interval and threshold, given a model.

        Parameters
        ----------
		init : int
			The first year of the timeframe.

		end : int
			The last year of the timeframe.

		model : str
			Which model to compute: frequentist, embedding or all
			(which computes both of them).

		threshold : float
			Presences below this threshold will be ignored.

		N : int
			Applies only for the embedding model: number of dimensions
			to be computed.
        """
        self.__parameters = (init, end, threshold, N)
        self.__p = self.__x.presence(init, end, threshold)[0]
        self.__setup()
        if model == "frequentist":
            self.__frequentist()
        elif model == "embedding":
            self.__embedding()
        elif model == "all":
            self.__frequentist()
            self.__embedding()
        else:
            print("Not a valid model.")
            return


    def __store(self, trial):
        if not os.path.isdir("__rscache__/spaces"):
            os.mkdir("__rscache__/spaces")

        np.save("__rscache__/spaces/" + trial +\
                self.key+".npy", self.phi[trial])


    def __setup(self):
        f = set()
        for sf in self.__p:
            f.add(sf[1])
       
        of = sorted(list(f))
        self.__n = len(of)
        self.__indices = {u: v for v, u in enumerate(of)}
        if not os.path.isdir("__rscache__"):
            os.mkdir("__rscache__")
        

    def __frequentist(self):
        trial = "frequentist{}".format(self.__parameters[:3])
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


    def __embedding(self):
        trial = "embedding{}".format(self.__parameters)
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
    

    def get_backbone(self, trial, alpha=0.05):
        """
        """
        if not os.path.isdir("__rscache__/backbone"):
            os.mkdir("__rscache__/backbone")
        
        source = "__rscache__/backbone/"+trial+str(alpha)+self.key+".ncol"

        try:
            return nx.read_weighted_edgelist(source)
        except:
            np.save("__rscache__/phi.npy", self.phi[trial][0])
            cmd = ["Rscript", "../MapSci/backbone.r",
                   "__rscache__/phi.npy", str(alpha), source]
            p = subprocess.Popen(cmd)
            p.wait()
            os.remove("__rscache__/phi.npy")
            return nx.read_weighted_edgelist(source)

    
    def plot(self, trial, values=None, labels=None, pos=None, new=False,
		threshold=0.212, legend=True, save=False, filename="teste.pdf",
		with_labels=False):
        """
        Plot the research space
        """
        plt.rcParams["figure.figsize"] = (15,15)
        phi = self.phi[trial][0]
        names = {v:k for k,v in self.phi[trial][1].items()}
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
            
        if values == None:  
            values = [i for i in range(n)]
        cm, scalarMap = self.__colors(values)

        if labels != None:
            for lab in labels:
                ax.plot([0],[0], color = scalarMap.to_rgba(labels[lab]),
                    label=lab, lw=7)
            if legend:
                plt.legend(loc='upper left')

        curves = curved_edges(mast, pos)
        lc = LineCollection(curves, color='grey', alpha=0.1)
        plt.gca().add_collection(lc)

        nx.draw_networkx_nodes(mast, pos, cmap=cm, vmin=0,
            vmax= max(values), node_color=values,
            ax=ax, node_size=self.__size())

        if with_labels:
            nx.draw_networkx_labels(mast, pos, ax=ax, labels=names)

        plt.axis('off')
        f.set_facecolor('w')
        if save:
            plt.savefig(filename,bbox_inches='tight',pad_inches=0)
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


	# You can edit this if you wish
	# Maybe to you want to highlight some areas
    def __size(self):
        values = [300 for node in self.__indices]
        return values
