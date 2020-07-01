import numpy as np
from collections import defaultdict

class entities:
    """
    """

    def __init__(self, pres, indices):
        """
        """
        self.__indices = indices
        self.__p, self.__x = pres
        self.__advantages()
        self.__indicators()

        s = set()
        for sf in self.__p:
            s.add(sf[0])
        self.set = s


    def __advantages(self):
        """
        Compute revealed comparative advantage for researchers,
        institutions and states.
        """
        rca = defaultdict(int)
        sumf = defaultdict(int)
        sums = defaultdict(int)
        sumsf = 0

        for sf in self.__x:
            # AQUI
            if sf not in self.__p: continue
            sumf[sf[0]] += self.__x[sf]
            sums[sf[1]] += self.__x[sf]
            sumsf += self.__x[sf]
    
        for sf in self.__x:
            # AQUI
            if sf not in self.__p: continue
            rca[sf] = (self.__x[sf]/sumf[sf[0]])/(sums[sf[1]]/sumsf)
        self.rca = rca


    def __indicators(self):
        """
        Compute the indicator matrices
        """

        U = [defaultdict(set), defaultdict(set)]
        f = set(self.__indices.keys())

        for sf in self.rca:
            if sf[1] not in f: continue
            if self.rca[sf] > 0:
                U[0][sf[0]].add(self.__indices[sf[1]])
                if self.rca[sf] >= 1:
                    U[1][sf[0]].add(self.__indices[sf[1]])

        self._U = U


    def predict(self, s, phi, transition):
        """
        Predict s's future areas
        """

        if transition == 'inactive-active':
            transition = 0
        elif transition == 'nascent-developed':
            transition = 1
        elif transition == 'intermediate-developed':
            transition = 2
        else:
            raise ValueError('{} is not a transition.'.format(transition))
        indicator = (transition > 0)

        omega = list()
        n = range(len(self.__indices))
        norm = [sum(phi[i,j] for j in n) for i in n]    
        for f in self.__indices:
            i = self.__indices[f]
            if i in self._U[indicator][s]:
                continue

            if indicator:
                val = self.rca[(s,f)]
                if transition == 1 and (val >= 0.5 or val == 0):
                    continue
                if transition == 2 and (val < 0.5 or val >= 1):
                    continue

            num = sum(phi[i,j] for j in self._U[indicator][s])
            div = np.round(num/norm[i], 5)
            if div > 0.00004:
                omega.append((div, f))

        return sorted(omega, reverse=True)

    
    def info(self):
        """
        """
        try:
            return self.__info
        except:
            sumx = defaultdict(int)
            for sf in self.__p:
                sumx[sf[0]] += rs_2011.x[sf]
            
            fields = [list(), list()]
            X = list()
            for s in self.set:
                fields[0].append(len(self._U[0][s]))
                fields[1].append(len(self._U[1][s]))
                X.append(sumx[s])

            self.__info = (X, fields)
            return self.__info
