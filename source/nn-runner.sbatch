#!/bin/bash
#############################################################################
#SBATCH --job-name=neural-net
#SBATCH --output=neural-net.out
#SBATCH --nodelist=node052,node053
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=500M
#SBATCH --ntasks=12

script_name="hparam-search.py"
nsteps=$1
prob_file=$2
save_loc=$3
log_name=$4

#Begin job
echo "--] running job on " `hostname`
echo Begin: `date`
echo "slurm ntasks: $SLURM_NTASKS"

for i in $(seq $nsteps); do
   
   echo "running $i" 
   srun -n 1 -N 1 python $script_name $i $SLURM_CPUS_PER_TASK $prob_file $save_loc $log_name &> py-output.out &
   sleep 2
   
   if (( $i % $SLURM_NTASKS == 0 )); then
      echo "Waiting..."
      wait
   fi

done
wait
echo End: `date`
