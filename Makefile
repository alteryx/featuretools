FEATURETOOLS_VERSION := $(shell python setup.py --version)

clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

lint:
	flake8 featuretools && isort --check-only --recursive featuretools

lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --exclude="*/migrations/*" --select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306" featuretools
	isort --recursive featuretools


test: lint
	pytest featuretools/tests

testcoverage: lint
	pytest featuretools/tests --cov=featuretools

installdeps:
	pip install --upgrade pip
	pip install -e .
	pip install -r dev-requirements.txt

circleci-install-packaged-featuretools:
	python setup.py sdist
	tar -zxvf "dist/featuretools-${FEATURETOOLS_VERSION}.tar.gz"
	pip install -e "featuretools-${FEATURETOOLS_VERSION}/"
	pip install -r "featuretools-${FEATURETOOLS_VERSION}/test-requirements.txt"

circleci-run-packaged-tests:
	cd "featuretools-$(python setup.py --version)/"
	pytest

circleci-run-test-coverage:
	pip install "$(shell cat dev-requirements.txt | grep codecov)"
	coverage erase
	cd "featuretools-${FEATURETOOLS_VERSION}/"
	coverage erase
	pytest featuretools/tests --cov=featuretools --cov-config=../.coveragerc
	cd ../
	cp "featuretools-${FEATURETOOLS_VERSION}/.coverage" .coverage
	codecov
