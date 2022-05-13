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
	black featuretools -t py310 --check
	flake8 featuretools

.PHONY: lint-fix
lint-fix:
	black -t py310 featuretools
	isort featuretools
	python docs/notebook_version_standardizer.py standardize

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

.PHONY: pkgversion
pkgversion:
	$(eval PACKAGE := $(shell grep '__version__\s=' featuretools/version.py | grep -o '[^ ]*$$'))

.PHONY: package_featuretools
package_featuretools: upgradepip upgradebuild pkgversion
	python -m build
	tar -zxvf "dist/featuretools-${PACKAGE}.tar.gz"
	mv "featuretools-${PACKAGE}" unpacked_sdist

.PHONY: install_sdist
install_sdist: pkgversion
	pip install "dist/featuretools-${PACKAGE}.tar.gz"

.PHONY: install_bdist
install_bdist: pkgversion
	pip install "dist/featuretools-${PACKAGE}.whl"
