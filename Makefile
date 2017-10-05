clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

test: clean
	python setup.py test --addopts --boxed

installdeps:
	pip install --upgrade pip
	pip install -r dev-requirements.txt
	pip install -e .
