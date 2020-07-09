.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

.PHONY: lint
lint:
	flake8 featuretools && isort --check-only --recursive featuretools

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --exclude="*/migrations/*" --select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221,E261,E111,E114" featuretools
	isort --recursive featuretools

.PHONY: test
test: lint
	pytest featuretools/

.PHONY: testcoverage
testcoverage: lint
	pytest featuretools/ --cov=featuretools

.PHONY: installdeps
installdeps:
	pip install --upgrade pip
	pip install -e .
	pip install -r dev-requirements.txt

.PHONY: checkdeps
checkdeps:
	$(eval allow_list='scipy|numpy|pandas|tqdm|pyyaml|cloudpickle|distributed|dask|psutil|click')
	pip freeze | grep -v "FeatureLabs/featuretools.git" | grep -E $(allow_list) > $(OUTPUT_PATH)
