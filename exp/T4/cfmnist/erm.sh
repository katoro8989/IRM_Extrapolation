DATASET="CFMNIST"
EPOCHS=500
TRAIN_BATCH_SIZE=25000
SEED=2020
ARCH="MLP390"
TRAINER="ERM"
DATA_DIR="/gs/bs/tge-24IJ0078/dataset"
HIDDEN_DIM=390
OPTIM="adam"
TRAINING_ENV="0.1 0.2"
TRAINING_CLASS_ENV=""
TRAINING_COLOR_ENV=""
TEST_ENV=0.9
LABEL_FLIP_P=0.25
WD=0.00110794568
PENALTY_WEIGHT=91257.18613115903
LR=0.0004898536566546834
WARM_START=190
OMEGA_LR=0.002
PRINT_FREQ=100
RESULT_DIR="./results"
GPU="0"
NO_CUDA=""
WANDB_PROJECT_NAME="ERM_CFMNIST"

SEEDS=(2021 2022)
count=0

for seed in "${SEEDS[@]}" ; do
    SHELL_ARGS="--dataset ${DATASET} \
                --epochs ${EPOCHS} \
                --train_batch_size ${TRAIN_BATCH_SIZE} \
                --seed ${seed} \
                --arch ${ARCH} \
                --trainer ${TRAINER} \
                --data_dir ${DATA_DIR} \
                --hidden-dim ${HIDDEN_DIM} \
                --optim ${OPTIM} \
                --save \
                --training_env ${TRAINING_ENV} \
                --test_env ${TEST_ENV} \
                --wd ${WD} \
                --penalty_weight ${PENALTY_WEIGHT} \
                --lr ${LR} \
                --warm_start ${WARM_START} \
                --omega_lr ${OMEGA_LR} \
                --print_freq ${PRINT_FREQ} \
                --wandb_project_name ${WANDB_PROJECT_NAME} \
                "

    CMD="qsub -g tge-24IJ0078 run.sh ${SHELL_ARGS}"
    echo "Exp-$((count + 1)): ${CMD}"
    eval $CMD

    count=$((count += 1))
            
done