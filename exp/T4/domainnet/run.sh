#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:30:00
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
PYTHON_ARGS=$@

# ======== Execution ========
CMD="python -m train ${PYTHON_ARGS}"
echo $CMD
eval $CMD

popd