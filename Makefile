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
	black featuretools docs/source -t py310 --check
	flake8 featuretools

.PHONY: lint-fix
lint-fix:
	black featuretools docs/source -t py310
	isort featuretools
	python docs/notebook_version_standardizer.py standardize

.PHONY: test
test:
	pytest featuretools/ -n auto

.PHONY: testcoverage
testcoverage:
	pytest featuretools/ --cov=featuretools -n auto

.PHONY: installdeps
installdeps: upgradepip
	pip install -e ".[dev]"
	pre-commit install

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

.PHONY: upgradesetuptools
upgradesetuptools:
	python -m pip install --upgrade setuptools

.PHONY: package
package: upgradepip upgradebuild upgradesetuptools
	python -m build
	$(eval PACKAGE=$(shell python -c "from pep517.meta import load; metadata = load('.'); print(metadata.version)"))
	tar -zxvf "dist/featuretools-${PACKAGE}.tar.gz"
	mv "featuretools-${PACKAGE}" unpacked_sdist
