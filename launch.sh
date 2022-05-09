#!/bin/bash
#SBATCH --output=pytestjob.%j.%N.out  #saving standard output to file -- %j jobID -- %N nodename
#SBATCH --error=pytestjob.%j.%N.err   #saving standard error to file -- %j jobID -- %N nodename
##SBATCH --mail-type=ALL    #email optitions
##SBATCH --mail-user=Michael.Wojnowicz@tufts.edu

## Reference:  /cluster/tufts/hpc/tools/slurm_scripts
## Example use: sbatch -p preempt —mem=2g launch.sh execution_time.py

python  $1