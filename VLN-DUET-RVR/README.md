1. follow scaleVLN to get datasets for R2R and the pretrained LXMERT (connectivity files and img features of rvr are from R2R data) and unzip it in datasets.
2. download RVR data from [here](https://huggingface.co/datasets/OpenGVLab/ScaleVLN/blob/main/rvr_data.zip) and unzip it in datasets. We expects files in this structure:
```
datasets
    ├─ R2R
    ├─ REVERIE
    ├─ pretrained
```
3. bash submit_reverie.sh to run rvr pretraining (make take ~1h to preload the data in training).
4. bash script/run_reverie.sh to run rvr finetuneing.
