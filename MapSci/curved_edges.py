import bezier
import networkx as nx
import numpy as np

# This code is not mine. I've just changed some var names.
# Author: Geoff Sims, @beyondbeneath
# Link: https://github.com/beyondbeneath/bezier-curved-edges-networkx
# Thank you!

def curved_edges(G, pos, dist_ratio=0.2, precision=20, polarity='random'):
    edges = np.array(G.edges())
    l = edges.shape[0]

    if polarity == 'random':
        rnd = np.where(np.random.randint(2, size=l)==0, -1, 1)
    else:
        vec = lambda x: np.vectorize(hash)(x)
        val = np.mod(vec(edges[:,0]) + vec(edges[:,1]),2)
        rnd = np.where(val==0,-1,1)
    
    u, inv = np.unique(edges, return_inverse = True)
    coords = np.array([pos[x] for x in u])[inv]
    coords = coords.reshape([edges.shape[0], 2, edges.shape[1]])
    coords_node1 = coords[:,0,:]
    coords_node2 = coords[:,1,:]
    
    should_swap = coords_node1[:,0] > coords_node2[:,0]
    coords_node1[should_swap], coords_node2[should_swap] =\
        coords_node2[should_swap], coords_node1[should_swap]
    
    val = np.sum((coords_node1-coords_node2)**2, axis=1)
    dist = dist_ratio * np.sqrt(val)

    m1 = (coords_node2[:,1]-coords_node1[:,1])/\
        (coords_node2[:,0]-coords_node1[:,0])
    m2 = -1/m1

    t1 = dist/np.sqrt(1+m1**2)
    v1 = np.array([np.ones(l),m1])
    coords_node1_displace = coords_node1 + (v1*t1).T
    coords_node2_displace = coords_node2 - (v1*t1).T

    t2 = dist/np.sqrt(1+m2**2)
    v2 = np.array([np.ones(len(edges)),m2])
    coords_node1_ctrl = coords_node1_displace + (rnd*v2*t2).T
    coords_node2_ctrl = coords_node2_displace + (rnd*v2*t2).T

    node_matrix = np.array([coords_node1, coords_node1_ctrl,
        coords_node2_ctrl, coords_node2])

    curveplots = []
    for i in range(l):
        nodes = node_matrix[:,i,:].T
        space = np.linspace(0,1,precision)
        curve = bezier.Curve(nodes, degree=3)
        curveplots.append(curve.evaluate_multi(space).T)
      
    curves = np.array(curveplots)
    return curves
