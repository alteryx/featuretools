## Release Process
#### Update repo
1. Bump verison number in setup.py, and __init__.py
2. Update changelog.rst


#### Deploy docs
From root
```
cd docs
python source/upload.py --root
```