from shrinkBigComponent import *
from perturbEdges import *
from compressDemand import *
from determineFlow import *
import os

if __name__=="__main__":
    new_dir='AustinVeryShrunk'
    fraction=0.995
    directory=scratch+'/data/{}'.format(new_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    shrink_graph(scratch+'/data/AustinReduced/reducedEdges.csv',
                    scratch+'/data/AustinReduced/reducedDemand.csv', 
                    scratch+'/data/{}/edges.csv'.format(new_dir),
                    scratch+'/data/{}/demand.csv'.format(new_dir),
                    fraction=fraction)
    edge_file=scratch+'/data/{}/edges.csv'.format(new_dir)
    perturbed_file=scratch+'/data/{}/perturbedEdges.csv'.format(new_dir)
    label_file=scratch+'/data/{}/labels.csv'.format(new_dir)
    perturb(edge_file,label_file,perturbed_file)

    big_demand=scratch+"/data/AustinReduced/demand.csv"
    path_src=scratch+"/data/AustinReduced/output8/paths.csv"
    little_edges=scratch+"/data/{}/edges.csv".format(new_dir)
    df=compress(big_demand,path_src,little_edges)
    df.to_csv(scratch+"/data/{}/compressedDemand.csv".format(new_dir)
        ,index=False)
    demand=path_to_dict(scratch+'/data/{}/compressedDemand.csv'.format(new_dir))
    edges=path_to_dict(scratch+'/data/{}/edges.csv'.format(new_dir))
    perturbed=path_to_dict(scratch+'/data/{}/perturbedEdges.csv'.format(new_dir))
    iterations=250

    atp=fw.AssignTrafficPython()
    fake_flow=atp.flow(demand,perturbed,iterations)
    pd.DataFrame(fake_flow).to_csv("fakeFlow.csv",index=False)
    real_flow=atp.flow(demand,edges,iterations)
    pd.DataFrame(real_flow).to_csv("realFlow.csv",index=False)