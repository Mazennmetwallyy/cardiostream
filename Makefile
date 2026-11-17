PY ?= .venv/bin/python

.PHONY: setup data cv train records test lint dev web docker

setup:            ## create the venv with training + dev dependencies
	python3 -m venv .venv && $(PY) -m pip install -e ".[train,dev]" && cd frontend && npm ci

data:             ## download the MIT-BIH Arrhythmia Database (~90 MB) from PhysioNet
	$(PY) -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw')"

cv:               ## choose class weighting by patient-grouped CV on the training patients
	$(PY) -m cardiostream.cv

train:            ## train on DS1, evaluate once on DS2, export artifacts/model.onnx
	$(PY) -m cardiostream.train

records:          ## package the held-out DS2 recordings for the live demo
	$(PY) -m cardiostream.export_records

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check . && cd frontend && npm run lint

dev:              ## API on :8000 (run `make web` in another terminal for the dashboard)
	$(PY) -m uvicorn cardiostream.api:app --reload --port 8000

web:
	cd frontend && npm run dev

docker:
	docker build -t cardiostream . && docker run --rm -p 8000:8000 cardiostream
