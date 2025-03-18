DATASET="VLCS_FROM_DOMAINBED"
ARCH="resnet18"
EPOCHS=200
TRAIN_BATCH_SIZE=32
SEED=2020
TRAINER="vIRM"
DATA_DIR="/gs/bs/tge-24IJ0078/dataset"
OPTIM="adam"
TRAINING_ENV="0 1 3"
TRAINING_CLASS_ENV=""
TRAINING_COLOR_ENV=""
TEST_ENV=2
LABEL_FLIP_P=0.25
WD=0.00110794568
PENALTY_WEIGHT=1
LR=1e-4
WARM_START=0
OMEGA_LR=0.002
PRINT_FREQ=100
RESULT_DIR="./results"
GPU="0"
NO_CUDA=""
WANDB_PROJECT_NAME="vIRMv1_VLCS"
NUM_CLASSES=5

SEEDS=(2020)
VAR_BETAS=(1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1)
count=0

for var_beta in "${VAR_BETAS[@]}" ; do
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
                --lr ${LR} \
                --warm_start ${WARM_START} \
                --omega_lr ${OMEGA_LR} \
                --print_freq ${PRINT_FREQ} \
                --wandb_project_name ${WANDB_PROJECT_NAME} \
                --arch "resnet18" \
                --num_classes ${NUM_CLASSES} \
                --var_beta ${var_beta} \
                "

    CMD="qsub -g tge-24IJ0078 run.sh ${SHELL_ARGS}"
    echo "Exp-$((count + 1)): ${CMD}"
    eval $CMD

    count=$((count += 1))
            
done