import sys
import os
import pandas as pd
from random import randint
import networkx as nx
from reduceGraph import reduce_two_biggest
scratch=os.environ["SCRATCH"]

#loads a connected graph and then reduces it by some amount, while still maintaining connectivity

def prune_graph(g,keep_nodes):
    """ prunes graph to only have nodes in component set
    
    paramters
        g- graph
        keep_nodes -set of nodes to keep
    
    returns
        g -- graph with fewer nodes
    
    """
    
    drop_nodes=[n for n in g.nodes() if n not in keep_nodes]
    g.remove_nodes_from(drop_nodes)
    return g

def snip_leaves(g,limit):
    """ prunes leaves until limit amount leaves have been pruned, or theres none left; whichever comes first
    """
    removed=0
    while removed<limit:
        drop_nodes=[]
        for n in g.nodes():
            in_edges=[e for e in g.in_edges(n)]
            out_edges=[e for e in g.out_edges(n)]
            if len(in_edges)==0 and removed<limit:
                drop_nodes.append(n)
                removed+=1
                continue
            snippable=True
            if len(in_edges)!=len(out_edges):
                snippable=False
            for (tail,head) in in_edges:
                if g.has_edge(head,tail)==False:
                    snippable=False
                    break
            if snippable==True and removed<limit:
                drop_nodes.append(n)
                removed+=1
        if len(drop_nodes)==0:
            print('no more droppable nodes removed {} instead of {}'.format(removed,limit))
            break
        print('removed {}/{} nodes'.format(len(drop_nodes),len(g)))
        g.remove_nodes_from(drop_nodes)
    return g

class Edge:
    def __init__(self,length,capacity,speed):
        self.length=length
        self.capacity=capacity
        self.speed=speed

def shrink_graph(edge_src,demand_src,edge_dest='edges.csv',demand_dest='demand.csv',fraction=0.75,write=True):
    """prunes fraction*(len(graph)) graph leaves using snip_leaves and optionally writes 
    to new files; gets rid of OD demand pairs where both are not in the new graph
    
    """
    g=nx.DiGraph()
    edges={} #(tail,head):Edge
    demand={} #(tail,head): Volume
    with open(edge_src,'r') as src:
        edge_header=next(src).strip()
        for r in src:
            row=r.strip().split(',')
            [tail,head,length,capacity,speed]=row
            edges[(tail,head)]=Edge(length,capacity,speed)
            g.add_edge(tail,head)
    limit=int(fraction* len(g))
    g=snip_leaves(g,limit)
    if write==True:
        total_pairs=0
        written_pairs=0
        with open(demand_src,'r') as src, open(demand_dest,'w') as dest:
            demand_header=next(src).strip()
            dest.write(demand_header)
            for r in src:
                total_pairs+=1
                row=r.strip().split(',')
                [tail,head,volume]=row
                if g.has_node(tail) and g.has_node(head) and nx.has_path(g,tail,head) ==True:
                    written_pairs+=1
                    demand[(tail,head)]=volume
                    dest.write('\n'+r.strip())
        total_edges=0
        written_edges=0
        new_edges=[]
        with open(edge_src,'r') as src,open(edge_dest,'w') as dest:
            edge_header=next(src).strip()
            dest.write(edge_header)
            for r in src:
                total_edges+=1
                row=r.strip().split(',')
                [tail,head,length,capacity,speed]=row
                if g.has_node(tail) and g.has_node(head):
                    new_edges.append(r)
            new_edges.sort()
            for r in new_edges:
                written_edges+=1
                dest.write('\n'+r.strip())
    for (name,written,total) in [('edges',written_edges,total_edges),('demand pairs',written_pairs,total_pairs)]:
        print('{} written : {}/{}'.format(name,written,total))
    return g
                

if __name__=='__main__':
    '''shrink_graph(scratch+'/data/AustinReduced/reducedEdges.csv',
                 scratch+'/data/AustinReduced/reducedDemand.csv', 
                 scratch+'/data/AustinShrunk/edges.csv',
                 scratch+'/data/AustinShrunk/demand.csv',fraction=0.99)'''
    shrink_graph(scratch+'/data/AustinReduced/reducedEdges.csv',
                 scratch+'/data/AustinReduced/reducedDemand.csv', 
                 scratch+'/data/AustinExtraShrunk/edges.csv',
                 scratch+'/data/AustinExtraShrunk/demand.csv',fraction=0.995)