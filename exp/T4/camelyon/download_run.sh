#!/bin/sh
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=02:00:00
#$ -o results/$JOB_ID.out
#$ -e results/$JOB_ID.err
#$ -N irmv1_vrex.sh

# ======== Module, Virtualenv and Other Dependencies ======
source ../../env/t4_env.sh
echo "PYTHON Environment: $PYTHON_PATH"
export PYTHONPATH=.
export PATH=$PYTHON_PATH:$PATH

export WANDB_API_KEY="7ff2b95cd90e0744a1d5408aaff9de4703d8478d"



# ======== Configuration ========
pushd /home/3/uy02093/workspace/IRM_Extrapolation

# ======== Execution ========
CMD="python -m datasets.download_domainbed --data_dir /gs/bs/tge-24IJ0078/dataset"
echo $CMD
eval $CMD

popd