import os

scratch=os.environ["SCRATCH"]
import argparse

parser = argparse.ArgumentParser('')

#this script puts all the flows into one csv

def flowCsv(src_files,dest_file):
    """
    Given a list of source files, this function creates a new csv file that contains the average flow
    for each edge in the network
    
    Args:
      src_files: a list of paths to the files containing the flow data
      dest_file: the name of the file to be created
    """
    flows={}
    for i,path in enumerate(src_files):
        with open(path,'r') as src:
            next(src)
            next(src)
            for l in src:
                line=l.strip().split(',')
                tail=line[1]
                head=line[2]
                flow=float(line[-1])
                if (tail,head) not in flows:
                    flows[(tail,head)]=[]
                flows[(tail,head)].append(flow)
        for k in flows.keys():
            if len(flows[k])<=i:
                flows[k].append(0.0)
    with open(dest_file,'w+') as dest:
        header='tail,head,0,'+','.join([str(z) for z in range(50,1000,50)])
        dest.write(header)
        for (tail,head),v in flows.items():
            dest.write('\n{},{},'.format(tail,head)+','.join([str(f) for f in v]))

if __name__ =='__main__':
    #src_files=['data/jamesburg/output/flow.csv']+['data/jamesburg/output{}/flow.csv'.format(z) for z in range(50,600,50)]
    src_files=['austinshrunk/output/flow.csv']+['austinshrunk/output{}/flow.csv'.format(z) for z in range(50,1000,50)]
    dest_file=scratch+'/data/AustinShrunk/allFlows.csv'
    flowCsv(src_files,dest_file)