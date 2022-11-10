# NeurIPS 2022 CausalML Challenge

## Requirements

Python 3.7+ is required. To install the requirements:
```setup
pip install -r requirements.txt
```

## How to run the code

### Task 1
1. `cd task_1`
2. Generate features: `python generate_features.py`
3. Get predicted adjacency matrix on private data: `python train_test.py`

### Task 2
1. `cd task_2`
2. Get predicted CATE on private data: `python main.py`

### Task 3
1. `cd task_3`
2. Show meta information: `python know_the_data.py`.
3. Load and filter data: `python data_preparation.py`.
4. Data imputation: `python data_imputation.py`.
5. Get results from heuristics: `python calculate_heuristics.py`.

### Task 4
1. `cd task_4`
2. process data: 
	mkdir task_4_data_processed
	python task_4_data.py
3. imputation: 
	cd submission
	python impute.py (missingpy needs scikit-learn==0.20.1)
4. train and output scores:
	python task_4_train.py

## Remark
- Note that this is just a temporary version of the code and far from being cleaned up.
