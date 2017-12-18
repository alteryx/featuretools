TEST_CMD=setup.py test --addopts --boxed
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

lint:
	flake8 featuretools && isort --check-only --recursive featuretools

test: lint
	python $(TEST_CMD)

installdeps:
	pip install --upgrade pip
	pip install -e .
	pip install -r dev-requirements.txt
