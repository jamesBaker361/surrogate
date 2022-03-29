import pandas as pd
import sys
import os
import networkx as nx


scratch=os.environ["SCRATCH"]

def link_id_to_nodes(network_src):
    """ creates dict to map link IDs to (tail,head)


    Returns
    -------
    d dict
        schema {link_id : (tail,head)}
    """
    with open(network_src,"r") as src:
        header=next(src).strip().split(',')
        link_id_index=header.index("linkId")
        from_index=header.index("fromNodeId")
        to_index=header.index("toNodeId")
        d={} #linkID: (tail,head)
        for r in src:
            row=r.strip().split(',')
            link_id=row[link_id_index]
            d[link_id]=(row[from_index],row[to_index])
        return d

def get_node_paths(path_src,demand_df):
    """
    Parses path_src file to find the actual nodes traversed for each demand pair. path file format:
    
    Parameters
    ----------
    path_src str
        the file path to the path csv; looks like
            # Main file: output/output.csv
            numIteration,odPair,edges
            1,0,1789
            1,1
            1,2,1702,1712
            1,3,1628
    demand_df pd.DataFrame
        df with columns "origin" "destination" "volume"
        
    Returns
    -------
    node_paths list
        node_paths[i]=[h,j,k] where demand pair i starts at node h, then goes to j, then k, etc
    """
    network_src=scratch+"/data/Austin/network.csv"
    id_dict=link_id_to_nodes(network_src)
    node_paths=[]
    iterations=0
    with open(path_src,"r") as src:
        next(src)
        next(src)
        for r in src:
            row=r.strip().split(",")
            iterations=max(iterations,int(row[0]))
    with open(path_src,"r") as src:
        next(src)
        next(src)
        for r in src:
            row=r.strip().split(",")
            if int(row[0])<iterations:
                continue
            pair_index=int(row[1])
            start=demand_df["origin"][pair_index]
            end=demand_df["destination"][pair_index]
            if len(row)==2:
                node_paths.append([start,end])
                continue
            edge_list=[start]
            for p in row[2:]:
                (t,h)=id_dict[p]
                if t not in edge_list:
                    edge_list.append(p)
                if h not in edge_list:
                    edge_list.append(h)
            if end not in edge_list:
                edge_list.append(end)
            node_paths.append(edge_list)
    return node_paths
                
            

            

def compress(big_demand,path_src,little_edges,excess=500):
    """compresses all the paths in path_src to only contain paths [node_i, node_j,.....node_n] with 
        start nodes that can reach the end nodes

    Args:
        big_demand str: file path location of demand file:
            origin,destination,volume
            141285,141322,15
            141285,141285,93
            140976,140979,1
            140860,140861,5
            ....
        path_src str: file path location of traffic paths file
            # Main file: /global/cscratch1/sd/jamesbak/data/AustinReduced/output8/output.csv
            numIteration,odPair,edges
            1,0,279893,295195
            1,1,189626,173400
            1,2,159861,170296
            1,3,110911,72391,72390
        little_edges str: file path location of shrunk edges file
            edge_tail,edge_head,length,capacity,speed
            139084,139183,11,800,15
            139085,139192,12,800,15
            139092,139093,5,800,15
            139092,139200,7,800,15
            139093,139092,5,800,15
        excess (int, optional): how much to add to each demand pair. Defaults to 500.

    Returns:
        pd.DataFrame: new od matrix
                  origin  destination  volume
            0     139500       139509     345
            1     141285       141322      15
            2     139995       139997       1
            3     139390       139403    1125
    """
    big_demand_df=pd.read_csv(big_demand,comment="#")
    lil_edges_df=pd.read_csv(little_edges,comment="#")
    node_paths=get_node_paths(path_src,big_demand_df)
    g=nx.DiGraph()
    for tail,head in zip(lil_edges_df["edge_tail"],lil_edges_df["edge_head"]):
        g.add_edge(int(tail),int(head))
    origins=[]
    destinations=[]
    volumes=[]
    for y in range(len(node_paths)):
        path=node_paths[y]
        path=[int(p) for p in path if int(p) in g]
        if len(path)>1 and nx.has_path(g,path[0],path[-1]):
            origins.append(path[0])
            destinations.append(path[-1])
            volumes.append(big_demand_df["volume"][y]+excess)
    return pd.DataFrame({"origin":origins,"destination":destinations,"volume":volumes})
    
if __name__=="__main__":
    big_demand=scratch+"/data/AustinReduced/demand.csv"
    path_src=scratch+"/data/AustinReduced/output8/paths.csv"
    little_demand=scratch+"/data/AustinShrunk/demand.csv"
    little_edges=scratch+"/data/AustinShrunk/edges.csv"
    big_demand_df=pd.read_csv(big_demand,comment="#")
    little_demand_df=pd.read_csv(little_demand,comment="#")
    df=compress(big_demand,path_src,little_edges)
    df.to_csv(scratch+"/data/AustinShrunk/compressedDemand.csv",index=False)