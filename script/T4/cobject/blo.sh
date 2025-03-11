DATASET="COCOcolor_LYPD"
EPOCHS=42
TRAIN_BATCH_SIZE=250
SEED=2020
TRAINER="BLO"
DATA_DIR="/gs/bs/tge-24IJ0078/dataset"
OPTIM="sgd"
TRAINING_ENV="0.999 0.7"
TRAINING_CLASS_ENV=""
TRAINING_COLOR_ENV=""
TEST_ENV=0.01
LABEL_FLIP_P=0.25
WD=0.00110794568
PENALTY_WEIGHT=10000
LR=0.01
WARM_START=3
OMEGA_LR=0.01
PRINT_FREQ=100
RESULT_DIR="./results"
GPU="0"
NO_CUDA=""
WANDB_PROJECT_NAME="BLO_CObject"

SEEDS=(2020)
LRS=(0.01)
count=0

for lr in "${LRS[@]}" ; do
    SHELL_ARGS="--dataset ${DATASET} \
                --epochs ${EPOCHS} \
                --train_batch_size ${TRAIN_BATCH_SIZE} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --data_dir ${DATA_DIR} \
                --optim ${OPTIM} \
                --save \
                --training_env ${TRAINING_ENV} \
                --test_env ${TEST_ENV} \
                --wd ${WD} \
                --penalty_weight ${PENALTY_WEIGHT} \
                --lr ${lr} \
                --warm_start ${WARM_START} \
                --omega_lr ${lr} \
                --print_freq ${PRINT_FREQ} \
                --wandb_project_name ${WANDB_PROJECT_NAME} \
                "

    CMD="qsub -g tge-24IJ0078 run.sh ${SHELL_ARGS}"
    echo "Exp-$((count + 1)): ${CMD}"
    eval $CMD

    count=$((count += 1))
            
done