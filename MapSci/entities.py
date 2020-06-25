import numpy as np
from collections import defaultdict

class entities:
    """
    """

    def __init__(self, pres, fields):
        """
        """
        self.fields = fields
        self.__p, self.__x = pres
        self.__advantages()
        self.__indicators()


    def __advantages(self):
        """
        Compute revealed comparative advantage for researchers,
        institutions and states.
        """
        rca = defaultdict(int)
        sumf = defaultdict(int)
        sums = defaultdict(int)
        sumsf = 0

        for sf in self.x:
            if sf not in self.__p: continue
            sumf[sf[0]] += self.x[sf]
            sums[sf[1]] += self.x[sf]
            sumsf += self.x[sf]
    
        for sf in self.x:
            if sf not in self.__p: continue
            rca[sf] = (self.x[sf]/sumf[sf[0]])/(sums[sf[1]]/sumsf)
        self.rca = rca


    def __indicators(self):
        """
        Compute the indicator matrices
        """

        U = [defaultdict(set), defaultdict(set)]
        f = set(self.fields)

        for sf in self.rca[lev]:
            if sf[1] not in f: continue
            if self.rca[sf] > 0:
                U[0][sf[0]].add(self.fields.index(sf[1]))
                if self.rca[sf] >= 1:
                    U[1][sf[0]].add(self.fields.index(sf[1]))

        self._U = U


    def predict(self, s, transition):
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
        n = range(len(self.fields))
        norm = [sum(phi[i,j] for j in n) for i in n]    
        for i in n:
            if i in self._U[indicator][s]:
                continue

            if indicator:
                val = self.rca[(s,self.fields[i])]
                if transition == 1 and (val >= 0.5 or val == 0):
                    continue
                if transition == 2 and (val < 0.5 or val >= 1):
                    continue

            num = sum(phi[i,j] for j in self._U[indicator][s])
            div = np.round(num/norm[i], 5)
            if div > 0.00004:
                omega.append((div, self.fields[i]))

        return sorted(omega, reverse=True)
