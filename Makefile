clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

lint:
	flake8 featuretools && isort --check-only --recursive featuretools

lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --exclude="*/migrations/*" --select="E225,E303,E302,E203,E128,E231" featuretools


test: lint
	pytest featuretools/tests

testcoverage: lint
	pytest featuretools/tests --cov=featuretools

installdeps:
	pip install --upgrade pip
	pip install -e .
	pip install -r dev-requirements.txt
