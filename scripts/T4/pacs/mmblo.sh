DATASET="PACS_FROM_DOMAINBED"
EPOCHS=200
TRAIN_BATCH_SIZE=32
SEED=2020
TRAINER="mmBLO"
DATA_DIR="/gs/bs/tge-24IJ0078/dataset"
OPTIM="adam"
TRAINING_ENV="0 1 3"
TRAINING_CLASS_ENV=""
TRAINING_COLOR_ENV=""
TEST_ENV=2
LABEL_FLIP_P=0.25
WD=0.00110794568
PENALTY_WEIGHT=1
LR=5e-4
WARM_START=50
OMEGA_LR=5e-4
PRINT_FREQ=100
RESULT_DIR="./results"
GPU="0"
NO_CUDA=""
WANDB_PROJECT_NAME="mmBLO_PACS"
ALPHA_MM=(-0.1 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9)
NUM_CLASSES=7

SEEDS=(2021 2022)
alpha_mm=-0.3
LRS=(0.1)
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
                --num_classes ${NUM_CLASSES} \
                --alpha_mm ${alpha_mm} \
                "

    CMD="qsub -g tge-24IJ0078 run.sh ${SHELL_ARGS}"
    echo "Exp-$((count + 1)): ${CMD}"
    eval $CMD

    count=$((count += 1))
            
done