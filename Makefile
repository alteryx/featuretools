.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	isort --check-only featuretools
	python docs/notebook_version_standardizer.py check-execution
	black featuretools -t py39 --check
	flake8 featuretools

.PHONY: lint-fix
lint-fix:
	black -t py39 featuretools
	isort featuretools
	python docs/notebook_version_standardizer.py standardize

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
	$(eval allow_list='scipy|numpy|pandas|tqdm|pyyaml|cloudpickle|distributed|dask|psutil|click|pyspark|koalas|woodwork')
	pip freeze | grep -v "alteryx/featuretools.git" | grep -E $(allow_list) > $(OUTPUT_PATH)

.PHONY: package_featuretools
package_featuretools:
	python setup.py sdist
	$(eval FT_VERSION=$(shell python setup.py --version))
	tar -zxvf "dist/featuretools-${FT_VERSION}.tar.gz"
	mv "featuretools-${FT_VERSION}" unpacked_sdist
