DATASET="COCOcolor_LYPD"
ARCH="resnet18"
EPOCHS=42
TRAIN_BATCH_SIZE=250
SEED=2020
TRAINER="ERM"
DATA_DIR="/gs/bs/tge-24IJ0078/dataset"
OPTIM="sgd"
TRAINING_ENV="0.999 0.7"
TRAINING_CLASS_ENV=""
TRAINING_COLOR_ENV=""
TEST_ENV=0.1
LABEL_FLIP_P=0.25
WD=0.00110794568
PENALTY_WEIGHT=10000
LR=0.01
WARM_START=3
OMEGA_LR=0.002
PRINT_FREQ=100
RESULT_DIR="./results"
GPU="0"
NO_CUDA=""
WANDB_PROJECT_NAME="ERM_CObject"

SEEDS=(2020)
count=0

for seed in "${SEEDS[@]}" ; do
    SHELL_ARGS="--dataset ${DATASET} \
                --epochs ${EPOCHS} \
                --train_batch_size ${TRAIN_BATCH_SIZE} \
                --seed ${seed} \
                --trainer ${TRAINER} \
                --data_dir ${DATA_DIR} \
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