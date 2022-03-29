import sys
import os
import pandas as pd

#this gets rid of all the demand pairs that are not in a particular set of edges

def trim(edge_file,demand_file,output_file):
    edge_df=pd.read_csv(edge_file)
    valid=set([t for t in edge_df['edge_tail']]).union(set([t for t in edge_df['edge_head']]))
    with open(demand_file,"r") as src, open(output_file,"w+") as dest:
        dest.write(next(src))
        for r in src:
            [tail,head,volume]=r.strip().split(',')
            if int(tail) in valid and int(head) in valid:
                dest.write(r)
                
if __name__ =="__main__":
    trim('data/AustinReduced/reducedEdges.csv','data/Austin/HourlyDemand/6.csv','data/AustinReduced/hour6ReducedDemand.csv')