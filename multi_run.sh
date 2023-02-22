lang_ids=("ar")
# lang_ids=("ar" "bn" "de" "en" "es" "fr" "hi" "id" "ja" "mr" "pt" "ru" "ur" "zh")

export CUDA_VISIBLE_DEVICES=0
export PATH_TO_CMD_DOWNLOAD="../cmd_download/"
# export PATH_TO_CMD_DOWNLOAD="../../nas-60022/cmd_download/"
export VAD_PATH="../../kuanyi/stage1_vad_timestamps/"

export N_WORKER=8
export BATCH_SIZE=64

if [[ ! -d output/ ]]; then
    mkdir output
fi

for lid in "${lang_ids[@]}"; do
    if [[ -d $PATH_TO_CMD_DOWNLOAD/vox100_${lid} ]]; then
        for channel in $(ls $PATH_TO_CMD_DOWNLOAD/vox100_${lid}); do
            if [[ ! -f output/vox100_${lid}/${channel}_output.txt ]]; then
                
                echo "now predicting "vox100_${lid}/${channel}
                python3 run_lid.py \
                    -s $PATH_TO_CMD_DOWNLOAD/vox100_${lid}/${channel} \
                    -w $N_WORKER \
                    --batch_size $BATCH_SIZE \
                    --max_trial 3 \
                    --chunk_sec 10
                
                if [[ $? == "2" ]]; then
                    echo "Error: vox100_${lid}/${channel}"
                fi
            fi
            if [[ ! -f output/vox100_${lid}/${channel}_vad_output.txt ]]; then
                
                echo "now predicting "vox100_${lid}/${channel}" with vad"
                python3 run_lid.py \
                    -s $PATH_TO_CMD_DOWNLOAD/vox100_${lid}/${channel} \
                    -w $N_WORKER \
                    --batch_size $BATCH_SIZE \
                    --max_trial 3 \
                    --chunk_sec 10 \
                    -v \
                    --vad_path $VAD_PATH/vad_time_stamp_${lid}_${channel}.txt
                
                if [[ $? == "2" ]]; then
                    echo "Error: vox100_${lid}/${channel} (vad)"
                fi
            fi
        done
    else
        echo "Source folder $PATH_TO_CMD_DOWNLOAD/vox100_${lid} not found"
    fi
done
