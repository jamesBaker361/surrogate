for algo in "fw" "msa"
do
	for cost in "gs" "bpr"
	do
		sbatch slurm_shifter.sh python3 /Traffic-Assignment-Frank-Wolfe-2021/assignment.py -a $algo -c $cost -n processed_networks/Austin_net.csv -d processed_networks/Austin_trips.csv -o Austin_"$algo"_"$cost"_flow
	done
done