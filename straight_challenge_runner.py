import os
import subprocess
import time

dataset_len = 587
n_processes = 10

indices = list(range(0,dataset_len,587//10))
indices[-1] = dataset_len
pairs = [[indices[x],indices[x+1]] for x in range(len(indices)-1) ]
name = 'noground'
processes = []
log_files = []
for index, pair in enumerate(pairs):
    log_files.append(open(f'save/direct_logs/name_{index}.txt','w'))
    processes.append(subprocess.Popen(args = [f"python3","straight_challenge.py",'--run_name',"direct_run",'--start_index',str(pair[0]),'--end_index',str(pair[1]),'--WANDB_MODE',"dryrun"],stdout=log_files[index],universal_newlines=True))

while True:
    polls = [process.poll() is not None for process in processes]
    if not all(polls):
        time.sleep(60)
    else:
        print("Done running!")
        break
for log_file in log_files:
    log_file.close()