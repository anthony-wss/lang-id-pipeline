import argparse
import os

from sklearn.metrics import precision_score, recall_score, accuracy_score
import nlp2
from tqdm import tqdm
import json
import ast

from lid_enhancement import AudioLIDEnhancer

import torch
from scripts.multi_thread_loader import MultiThreadLoader
from scripts.log_results import gen_pred_results
import pickle, time, threading
from math import ceil

code2label = json.load(open("code2label.json"))

Voxlingua107_dataset_path = '/storage/PromptGauguin/kuanyi/dev'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str, default=Voxlingua107_dataset_path, help="Source directory")
    parser.add_argument("-w", "--workers", type=int, default=3, help="Number of workers")
    parser.add_argument("-v", "--vad", action='store_true', help="Voice Activity Detection")
    parser.add_argument("--vad_path", type=str, default=None, help="vad timestamps file path")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_sec", type=int, default=30)
    parser.add_argument("--max_trial", type=int, default=10)
    parser.add_argument("--ground_truth", type=str, default='en')
    parser.add_argument("--wav", action='store_true', help="Use .wav file extension")
    args = parser.parse_args()
    config = vars(args)
    source_dir = config['src']
    voice_activity_detection = config['vad']
    vad_timestamp_path = config['vad_path']
    n_workers = config['workers']
    batch_size = config['batch_size']
    chunk_sec = config['chunk_sec']
    max_trial = config['max_trial']
    ground_truth = config["ground_truth"]

    if ground_truth not in code2label.keys():
        print(ground_truth, "is not in our lang list")
        print(f"Language {ground_truth} not found", file=open("error_logging.txt", "a"))
        exit()
    
    if source_dir[-1] == '/':
        source_dir = source_dir[:-1]

    if voice_activity_detection == True:
        with open(vad_timestamp_path, 'r') as f:
            vad_time_stamp_dict = eval(f.readline())

        print("With using VAD")
        vad_path = ast.literal_eval(open(config['vad_path'], "r").readline())
        vad_path = {k.split('/')[-1]: v for k, v in vad_path.items()}
    else:
        vad_time_stamp_dict = dict()
        print("Without using VAD")

    result_jsons = []

    audio_ext = '.ogg' if not config['wav'] else ".wav"

    for i in tqdm(nlp2.get_files_from_dir(source_dir, match=audio_ext)): 
        try:
            result_jsons.append(i)
        except:
            pass

    # Set up multi-thread loader
    loader = MultiThreadLoader(
        n_workers = n_workers, 
        batch_size = batch_size, 
        n_files = len(result_jsons),
        max_trial = max_trial,
        chunk_sec = chunk_sec,
        vad_path = vad_path if voice_activity_detection else None,
        ground_truth = config['ground_truth']
    )
    loader.start(files=result_jsons)
    ready_data_idx = 0
    
    wrong = []
    preds = []
    labels = []
    possible_langs = list(code2label.keys())
    lid_model = AudioLIDEnhancer(device='cuda', enable_enhancement=False, lid_voxlingua_enable=True, lid_silero_enable=False, lid_whisper_enable=True, voice_activity_detection=False)
    skip_audio_list = []

    progress = tqdm(total=ceil(len(result_jsons) / batch_size), desc="PRED", position=1)
    while True:
        X, y, l, r = loader.get_data() # X is list of batched data

        if loader.no_more_data:
            break

        elif X is not None:
            for i in range(len(X)):
                try:
                    pred = lid_model.forward(X[i])
                except RuntimeError:
                    exit(2)
                preds.extend([code2label[code] for code in pred])
                labels.extend(y[i])
                progress.update(1)

            loader.free_data(l, r)

        if not loader.no_more_data:
            time.sleep(1)
    
    if voice_activity_detection:
        output_file_name = f"{source_dir.split('/')[-1].strip()}_vad_output.txt"
    else:
        output_file_name = f"{source_dir.split('/')[-1].strip()}_output.txt"
        # output_file_name = "output.txt"

    gen_pred_results(
        labels=labels,
        preds=preds,
        total_sec=loader.total_sec,
        config=config,
        result_jsons=result_jsons, 
        src_dir=source_dir, 
        loading_time=loader.loading_time,
        file_idx=loader.file_idx,
        num_unvoiced=loader.num_unvoiced,
        unvoiced_idx=loader.unvoiced_idx,
        predicting_time=progress.format_dict['elapsed'],
        output_file=output_file_name,
        output_dir="output/" + config["ground_truth"] + "/"
        # output_dir="output/" + source_dir.split('/')[-2].strip()
    )
