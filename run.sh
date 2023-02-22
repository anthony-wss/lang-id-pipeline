export CHANNEL_FOLDER=$1
# export VAD_OUTPUT=$2

echo "now predicting "$CHANNEL_FOLDER
python3 run_lid.py \
    -s $CHANNEL_FOLDER \
    -w 4 \
    --batch_size 4 \
    --max_trial 5 \
    --chunk_sec 10 \
    --ground_truth $2
    # -v \
    # --vad_path $VAD_OUTPUT