#makefile

install:
	pip install --upgrade pip &&\
		pip install -r requirements/requirements.txt

train:
	python Diabetes_model/train_pipeline.py

test:
	pytest tests

build:
	python -m build

api_run:
	copy dist/*.whl Diabetes_API_app/
	python app/main.py

docker:
	docker build -t demo/Diabetes_model:latest


