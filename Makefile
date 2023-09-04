install:
	python.exe -m pip install --upgrade pip &&\
		pip install -r requirements.txt


format:
	black test.py


lint:
	pylint --disable=R,C --ignore-patterns=test_.*?py *.py mylib/*.py


all: install format lint
