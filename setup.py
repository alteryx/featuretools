from os import path

from setuptools import find_packages, setup

dirname = path.abspath(path.dirname(__file__))
with open(path.join(dirname, 'README.md')) as f:
    long_description = f.read()

extras_require = {
    'update_checker': ['alteryx-open-src-update-checker >= 2.0.0'],
    'koalas': open('koalas-requirements.txt').readlines(),
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name='featuretools',
    version='1.0.0rc1',
    packages=find_packages(),
    description='a framework for automated feature engineering',
    url='http://featuretools.com',
    license='BSD 3-clause',
    author='Feature Labs, Inc.',
    author_email='support@featurelabs.com',
    classifiers=[
         'Development Status :: 3 - Alpha',
         'Intended Audience :: Developers',
         'Programming Language :: Python :: 3',
         'Programming Language :: Python :: 3.7',
         'Programming Language :: Python :: 3.8'
    ],
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7, <4',
    extras_require=extras_require,
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
