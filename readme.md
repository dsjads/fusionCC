## Quick Start
### Requirements
- Python Package:
   - chardet==4.0.0
   - numpy==1.20.3
   - pandas==1.3.4
   - pyclustering==0.10.1.2
   - PyYAML==6.0
   - scikit_learn==1.2.1
   - scipy==1.7.1
   - torch==1.10.2

``pip install -r requirements.txt``

#### RQ1
- Run the scripts in `./cc/cc_baselines`

The output can be found under folder `./results/`

#### RQ2
- Run the scripts in `./cc/survey_pipeline`
- 
The output can be found under folder `./results/`

### Evaluation

#### RQ3 & RQ4
- Run the script: `./cc/triplet_cc_identify/run_fusion.py`

The output can be found under folder `./results/`

Eample output:

- origin_record.txt
  - meaning: ``program-id real_cc_num detected_cc_num intersection_of_them``
  - example: ``Chart-0	4	4	4``
- record.txt
  - meaning: ``program-id recall precision F1``
    - example: ``Chart-0	1.0	1.0	1.0``
- approach_MFR.txt or approach_MAR.txt
  - meaning: ``program-id MFR_or_MAR_value_list``
  - example: ``Chart-0	1	1	1``
- time.txt
  - meaning: ``program-id time_cost``
  - example: ``Chart-0	12.20``

## Project Structure
```
fusionCC
├── CONFIG.py
├── requirements.txt
├── cc
│   ├── CCGroundTruthPipeline.py
│   ├── CCinfo.yaml
│   ├── ReadData.py
│   ├── allinfo.yaml
│   ├── cc_baselines        # baselines
│   ├── cc_evaluation       # evaluation metrics of CCT detection
│   ├── core.py
│   ├── survey_pipeline     # RQ1 and RQ2
│   ├── triplenet_model     # model
│   └── triplet_cc_identify 
│       ├── ...
│       └── run_triplet.py  # entry
├── data            # (hand-made) example data 
├── fl_evaluation   # the suspicious evaluation of AFL
├── read_data       # read data
├── results         # results 
└── utils           
```

