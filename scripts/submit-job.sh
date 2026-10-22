#!/bin/bash
#BSUB -n 4
#BSUB -W 01:00

#BSUB -o batch_output/Output_%J_%I.out 
#BSUB -e batch_output/Output_%J_%I.err 

# Activate the enviroment
echo "Starting..."
uv sync
source .venv/bin/activate
echo "Python..."
# View current stack limit, in KiB
ulimit -s
# Higher stack size needed 
ulimit -s 262144

if [[ "$NoNoise" == "1" ]]; then
    python scripts/hpc-run-qse.py -m="$MOL" -e=0.1 -P=$P -N=4 -T=$LSB_JOBINDEX --NoNoise
else
    python scripts/hpc-run-qse.py -m="$MOL" -e=0.1 -P=$P -N=4 -T=$LSB_JOBINDEX -d "ibm_fez"
fi