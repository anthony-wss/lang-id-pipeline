for lang_folder in ../cmd_download/stage2-2/*; do

    echo $(basename $lang_folder)

    for channel in $lang_folder/*; do
        if [ ! -f output/$(basename $lang_folder)/$(basename $channel)_output.txt ]; then
            bash run.sh $channel $(basename $lang_folder)
        fi
        # python3 convert_format_sampling.py -s $channel -w 30
    done

done