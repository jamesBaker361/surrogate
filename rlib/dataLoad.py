import sys
import os
import pandas as pd

scratch=os.environ["SCRATCH"]

#loads network, perturbed network and demand and converts to dicts

def path_to_dict(src):
    df=pd.read_csv(src,comment="#")
    ret={}
    for c in df.columns:
        try:
            ret[c]=[int(_) for _ in df[c]]
        except ValueError:
            ret[c]=[str(_) for _ in df[c]]
    return ret

src_dir='AustinVeryShrunk'

network=path_to_dict(scratch+'/data/Austin/network.csv')
demand=path_to_dict(scratch+'/data/{}/compressedDemand.csv'.format(src_dir))
edges=path_to_dict(scratch+'/data/{}/edges.csv'.format(src_dir))
perturbed=path_to_dict(scratch+'/data/{}/perturbedEdges.csv'.format(src_dir))
labels=path_to_dict(scratch+'/data/{}/labels.csv'.format(src_dir))
real_flow=path_to_dict(scratch+'/data/{}/realFlow.csv'.format(src_dir))
fake_flow=path_to_dict(scratch+'/data/{}/fakeFlow.csv'.format(src_dir))

if __name__=="__main__":
    d=path_to_dict(scratch+'/data/{}/demand.csv')
    for k,v in d.items():
        print(k, len(v))