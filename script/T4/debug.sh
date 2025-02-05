SHELL_ARGS="--l2_regularizer_weight=0.001 \
            --lr 0.001 \
            --batch_size=512 \
            --penalty_anneal_iter=80 \
            --opt=sgd \
            --print_every=1 \
            --penalty_weight=0 \
            --steps=1000  \
            --step_gamma=0.1 \
            --dataset=ColoredObject \
            --irm_type=irmv1_vrex \
            --var_beta=0. \
            --min_alpha=-1. \
            --wandb_project_name=ColoredObject_erm_v2 \
            --wandb_entity_name=katoro13 \
            "
            
CMD="qsub -g tge-24IJ0078 run.sh ${SHELL_ARGS}"
echo "Exp-$((count + 1)): ${CMD}"
eval $CMD

count=$((count += 1))
        