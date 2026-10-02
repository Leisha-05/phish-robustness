PY=python

data:
	$(PY) src/data_prep.py
features:
	$(PY) src/features.py
tune:
	$(PY) src/tune.py
train:
	$(PY) src/train.py
perturb:
	$(PY) src/perturb.py
eval:
	$(PY) src/evaluate_robustness.py
explain:
	$(PY) src/explain.py
all: data features tune train perturb eval explain
