import sys
import os
import pandas as pd

scratch=os.environ["SCRATCH"]

#loads network, perturbed network and demand and converts to dicts

def path_to_dict(src):
    df=pd.read_csv(src,comment="#")
    ret={}
    for c in df.columns:
        ret[c]=[int(_) for _ in df[c]]
    return ret

demand=path_to_dict(scratch+'/data/AustinShrunk/compressedDemand.csv')
edges=path_to_dict(scratch+'/data/AustinShrunk/edges.csv')
perturbed=path_to_dict(scratch+'/data/AustinShrunk/perturbedEdges.csv')
labels=path_to_dict(scratch+'/data/AustinShrunk/labels.csv')
real_flow=path_to_dict(scratch+'/data/AustinShrunk/realFlow.csv')
fake_flow=path_to_dict(scratch+'/data/AustinShrunk/fakeFlow.csv')

if __name__=="__main__":
    d=path_to_dict(scratch+'/data/AustinShrunk/demand.csv')
    for k,v in d.items():
        print(k, len(v))