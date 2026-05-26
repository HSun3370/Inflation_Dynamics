
idarray=($(seq 1 20))

hmc_python_name="SyntheticFullTest.py"



for id in "${idarray[@]}"; do
   
                            count=0
                                        
                            action_name="Test_Full_N300"

                            dataname="${action_name}"

                            mkdir -p ./job-outs/${action_name}/id_${id}/

                            if [ -f ./bash/${action_name}/id_${id}/run.sh ]; then
                                rm ./bash/${action_name}/id_${id}/run.sh
                            fi

                            mkdir -p ./bash/${action_name}/id_${id}/

                            touch ./bash/${action_name}/id_${id}/run.sh

                            tee -a ./bash/${action_name}/id_${id}/run.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=id_${id}
#SBATCH --output=./job-outs/${action_name}/id_${id}/run.out
#SBATCH --error=./job-outs/${action_name}/id_${id}/run.err
#SBATCH --time=1-11:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

module load python/anaconda-2022.05  
source ~/myenv/bin/activate

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)

python3 -u /project/lhansen/Capital_NN_variant/BEGE_GARCH/$hmc_python_name --id ${id} 
echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
    count=$(($count + 1))
    sbatch ./bash/${action_name}/id_${id}/run.sh

done
