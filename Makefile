.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	python docs/notebook_version_standardizer.py check-execution
	ruff check . --config=./pyproject.toml
	ruff format . --check --config=./pyproject.toml

.PHONY: lint-fix
lint-fix:
	python docs/notebook_version_standardizer.py standardize
	ruff check . --fix --config=./pyproject.toml
	ruff format . --config=./pyproject.toml

.PHONY: test
test:
	python -m pytest featuretools/ -n auto

.PHONY: testcoverage
testcoverage:
	python -m pytest featuretools/ --cov=featuretools -n auto

.PHONY: installdeps
installdeps: upgradepip
	pip install -e .

.PHONY: installdeps-dev
installdeps-dev: upgradepip
	pip install -e ".[dev]"
	pre-commit install

.PHONY: installdeps-test
installdeps-test: upgradepip
	pip install -e ".[test]"

.PHONY: checkdeps
checkdeps:
	$(eval allow_list='holidays|scipy|numpy|pandas|tqdm|cloudpickle|distributed|dask|psutil|woodwork')
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
	$(eval PACKAGE=$(shell python -c 'import setuptools; setuptools.setup()' --version))
	tar -zxvf "dist/featuretools-${PACKAGE}.tar.gz"
	mv "featuretools-${PACKAGE}" unpacked_sdist
