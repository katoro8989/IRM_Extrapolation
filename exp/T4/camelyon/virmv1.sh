DATASET="Camelyon_FROM_DOMAINBED"
ARCH="resnet50"
EPOCHS=10
TRAIN_BATCH_SIZE=32
SEED=2020
TRAINER="vIRMv1"
DATA_DIR="/gs/bs/tge-24IJ0078/dataset"
OPTIM="adam"
TRAINING_ENV="1 2 3 4 5"
TRAINING_CLASS_ENV=""
TRAINING_COLOR_ENV=""
TEST_ENV=0
LABEL_FLIP_P=0.25
WD=0.0
PENALTY_WEIGHT=0.001
LR=5e-5
WARM_START=0
OMEGA_LR=0.002
PRINT_FREQ=100
RESULT_DIR="./results"
GPU="0"
NO_CUDA=""
WANDB_PROJECT_NAME="VIRM_CAMELYON"
NUM_CLASSES=2
SEED=2021

LAMBDA=(1e-3 1e-2 1e-1 1e0 1e1)
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
                --arch "resnet50" \
                --num_classes ${NUM_CLASSES} \
                --var_beta ${var_beta} \
                "

    CMD="qsub -g tga-SlavakisLab run.sh ${SHELL_ARGS}"
    echo "Exp-$((count + 1)): ${CMD}"
    eval $CMD

    count=$((count += 1))
            
done