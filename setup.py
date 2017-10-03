from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext


# Bootstrap numpy install
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


setup(
    name='featuretools',
    version='0.1.9',
    packages=find_packages(),
    description='a framework for automated feature engineering',
    url='http://featuretools.com',
    license='BSD 3-clause',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    classifiers=[
         'Development Status :: 3 - Alpha',
         'Intended Audience :: Developers',
         'Programming Language :: Python :: 2.7'],

    install_requires=['numpy>=1.11.0',
                      'scipy>=0.17.0',
                      'pandas==0.20.1',
                      'tqdm>=4.8.4',
                      "toolz>=0.8.2",
                      "dask[complete]==0.14.3",
                      "pyyaml>=3.12",
                      ],
    setup_requires=['pytest-runner>=2.0,<3dev', 'numpy>=1.11.0'],
    python_requires='>=2.7, <3',
    cmdclass={'build_ext': build_ext},
    test_suite='featuretools/tests',
    tests_require=['pytest>=3.0.1', 'mock==2.0.0'],
    keywords='feature engineering data science machine learning'
)
