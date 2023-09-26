export SIZE=100000
export STEPS=5
export TASKS=1
export RUNLINE="srun --ntasks $TASKS --nodes 1 --gpus-per-node $TASKS --exact ./forces.x $SIZE $STEPS $SIZE"

# To run multiple concurrent batches of sruns, without oversubscribing (but not async)
export BATCH_SIZE=4
export BATCHES=1  # default
if [[ -n $1 ]]; then
    export BATCHES=$1
fi

for j in $(seq $BATCHES); do
    for i in $(seq $BATCH_SIZE); do
        echo $RUNLINE; $RUNLINE &
    done
    wait
done

wait
