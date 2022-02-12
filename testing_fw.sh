for o in sys_opt combined_eq user_eq
do
    sbatch slurm_shifter_fw.sh AssignTraffic -obj $o -n 100 -i data/Austin-sorted-edges.csv -od data/Austin-od.csv -o results/fwoutput_"$o"
done