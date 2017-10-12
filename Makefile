TEST_CMD=setup.py test --addopts --boxed
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

test: clean
	python $(TEST_CMD)

coverage:
	py.test --cov=featuretools

installdeps:
	pip install --upgrade pip
	pip install -e .
	pip install -r dev-requirements.txt
