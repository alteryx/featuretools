from os import path

from setuptools import find_packages, setup

dirname = path.abspath(path.dirname(__file__))
with open(path.join(dirname, 'README.md')) as f:
    long_description = f.read()

setup(
    name='featuretools',
    version='0.8.0',
    packages=find_packages(),
    description='a framework for automated feature engineering',
    url='http://featuretools.com',
    license='BSD 3-clause',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    classifiers=[
         'Development Status :: 3 - Alpha',
         'Intended Audience :: Developers',
         'Programming Language :: Python :: 2.7',
         'Programming Language :: Python :: 3',
         'Programming Language :: Python :: 3.5',
         'Programming Language :: Python :: 3.6',
         'Programming Language :: Python :: 3.7'
    ],
    install_requires=open('requirements.txt').readlines(),
    setup_requires=open('setup-requirements.txt').readlines(),
    python_requires='>=2.7, <4',
    test_suite='featuretools/tests',
    tests_require=open('test-requirements.txt').readlines(),
    keywords='feature engineering data science machine learning',
    include_package_data=True,
    entry_points={
        'console_scripts': [
          'featuretools = featuretools.__main__:cli'
        ]
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)
