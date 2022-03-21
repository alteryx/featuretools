.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

.PHONY: lint
lint:
	flake8 featuretools && isort --check-only featuretools
	python docs/notebook_cleaner.py check-execution

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --exclude="*/migrations/*" --select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221,E261,E111,E114" featuretools
	isort featuretools
	python docs/notebook_cleaner.py standardize

.PHONY: test
test:
	pytest featuretools/

.PHONY: testcoverage
testcoverage:
	pytest featuretools/ --cov=featuretools

.PHONY: installdeps
installdeps: upgradepip
	pip install -e ".[dev]"

.PHONY: checkdeps
checkdeps:
	$(eval allow_list='holidays|scipy|numpy|pandas|tqdm|cloudpickle|distributed|dask|psutil|click|pyspark|woodwork')
	pip freeze | grep -v "alteryx/featuretools.git" | grep -E $(allow_list) > $(OUTPUT_PATH)

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip

.PHONY: upgradebuild
upgradebuild:
	python -m pip install --upgrade build

.PHONY: package_featuretools
package_featuretools: upgradepip upgradebuild
	python -m build
	$(eval FT_VERSION := $(python -c "from pep517.meta import load; metadata = load('.'); print(metadata.version)"))
	tar -zxvf "dist/featuretools-${FT_VERSION}.tar.gz"
	mv "featuretools-${FT_VERSION}" unpacked_sdist
