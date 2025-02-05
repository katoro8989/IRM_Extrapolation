#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=00:20:00
#$ -o results/$JOB_ID.out
#$ -e results/$JOB_ID.err
#$ -N irmv1_vrex.sh

# ======== Module, Virtualenv and Other Dependencies ======
source ./env/t4_env.sh
echo "PYTHON Environment: $PYTHON_PATH"
export PYTHONPATH=.
export PATH=$PYTHON_PATH:$PATH

export WANDB_API_KEY="412e8240b034b61dae066dfd1b3714cdde7e535e"



# ======== Configuration ========
pushd /home/3/uy02093/workspace/ColoredObjects_IRM
PYTHON_ARGS=$@

# ======== Execution ========
CMD="python -m train.py ${PYTHON_ARGS}"
echo $CMD
eval $CMD

popd