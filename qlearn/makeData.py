from frankwolfe import flow

import numpy as np
import os
import pandas as pd

from random import randint

scratch=os.environ["SCRATCH"]
from dataLoad import *

runs=250

samples=5000

x=[]
y=[]

for _ in range(samples):
	edges["capacity"]=[randint(200,1050) for c in edges["capacity"]]
	x.append(edges["capacity"])
	y.append(flow(demand,edges,runs)["flow"])

np.savez("samples250.npz",x=x,y=y)