import sys
import pandas as pd
import networkx as nx

#creates a reducedDemand,educedEdges for the biggest connected component and an extraReducedDemand,extraReducedEdges for the 2nd biggest



def reduce_two_biggest(edge_src,demand_src,reduced_list,reduced_edges_list,write=True):
    """
    Given a list of two or more demand files, and two or more edge files, 
        reduce the demand and edge files to only the edges and demand that are in the largest connected
    component.
    
    Args:
      edge_src: the source of the edges.csv file
      demand_src: the source of the demand data
      reduced_list: list of filenames for reduced demand
      reduced_edges_list: list of filenames for the reduced edges
      write: if True, write reduced demand and reduced edges to file. Defaults to True
    
    Returns:
      the graph, the big component and the small component.
    """
    edge_df=pd.read_csv(edge_src)
    demand_df=pd.read_csv(demand_src)
    g=nx.Graph()

    for t,h in zip(edge_df['edge_tail'],edge_df['edge_head']):
        g.add_edge(t,h)

    comps=sorted([c for c in nx.connected_components(g)],key =lambda x: -len(x))

    big_comp=comps[0]
    small_comp=comps[1]

    #reduced_list=['reducedDemand.csv','extraReducedDemand.csv']
    if write==True:
        for reduced,big_comp in zip(reduced_list,comps[:2]):
            with open('demand.csv','r') as src:
                with open(reduced,'w') as dest:
                    dest.write(next(src))
                    for r in src:
                        row=r.strip().split(',')
                        if int(row[0]) in big_comp and int(row[1]) in big_comp:
                            dest.write(r)

        #reduced_edges_list=['reducedEdges.csv','extraReducedEdges.csv']               
        for reduced_edges,big_comp in zip(reduced_edges_list,comps[:2]):
            with open('edges.csv','r') as src:
                with open(reduced_edges,'w') as dest:
                    dest.write(next(src))
                    for r in src:
                        row=r.strip().split(',')
                        if int(row[0]) in big_comp and int(row[1]) in big_comp:
                            dest.write(r)
    return g,big_comp,small_comp
                
if __name__=='__main__':
    edge_src='edges.csv'
    demand_src='demand.csv'
    reduced_list=['reducedDemand.csv','extraReducedDemand.csv']
    reduced_edges_list=['reducedEdges.csv','extraReducedEdges.csv']
    reduce(edge_src,demand_src,reduced_list,reduced_edges_list)