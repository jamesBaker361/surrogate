
from frankwolfe import flow
import sys
from tqdm import tqdm
from qlearn.dataLoad import *
import numpy as np
#import pandas as pd

import time

def current_milli_time():
    return round(time.time() * 1000)

edges=path_to_dict(scratch+'/data/AustinShrunk/edges.csv')
demand=path_to_dict(scratch+'/data/AustinShrunk/compressedDemand.csv')
perturbed=path_to_dict(scratch+'/data/AustinShrunk/perturbedEdges.csv')
big_cap=perturbed.copy()
small_cap=perturbed.copy()

big_cap["capacity"]=[1000 for c in perturbed["capacity"]]
small_cap["capacity"]=[100 for c in perturbed["capacity"]]

import timeit
names=["edges", "perturbed","big","small"]
nets=[edges,perturbed,big_cap,small_cap]
old_flows=[flow(demand,net,2)["flow"] for net in nets]
histories={name:[] for name in names}

for n in [names,nets,old_flows]:
    n=n

for x in tqdm(range(3,750)):
    new_flows=[]
    for name,net,old_flow in zip(names,nets,old_flows):
        start=current_milli_time()
        new_flow=flow(demand,net,x)["flow"]
        duration=current_milli_time()-start
        diff=[]
        for a,b in zip(old_flow,new_flow):
            diff.append(np.abs(a-b))
        reward=np.linalg.norm(diff)
        new_flows.append(new_flow)
        histories[name].append(reward)
    old_flows=new_flows
    pd.DataFrame(histories).to_csv("histories750.csv",index=False)
    #print("The time taken is ",timeit.timeit(stmt='flow(demand,perturbed,{})'.format(x),globals=globals()))