datetime=$(date "+%Y.%m.%d-%H.%M.%S")

if [[ -n $1 ]]; then
    export TRIAL=$1_$datetime
else
    export TRIAL=study_$datetime
fi

mkdir $TRIAL

export OUT=$TRIAL/run_outputs
mkdir $OUT

export TIMING=$TRIAL/time.out
export TIMEFORMAT=%R
touch $TIMING

# Run 8 sruns (two per GPU) ---------------------------------------------------------------------------------
echo -n 'Run 2 batches r1:  ' >>$TIMING; { time . ./run_batches.sh 2 >$OUT/f_2b_r1.out 2>&1; } 2>>$TIMING

sleep 10
echo -n 'Run 2 batches r2:  ' >>$TIMING; { time . ./run_batches.sh 2 >$OUT/f_2b_r2.out 2>&1; } 2>>$TIMING

# Run 16 sruns (four per GPU) ---------------------------------------------------------------------------------
sleep 10
echo -n 'Run 4 batches r1:  ' >>$TIMING; { time . ./run_batches.sh 4 >$OUT/f_4b_r1.out 2>&1; } 2>>$TIMING

sleep 10
echo -n 'Run 4 batches r2:  ' >>$TIMING; { time . ./run_batches.sh 4 >$OUT/f_4b_r2.out 2>&1; } 2>>$TIMING
