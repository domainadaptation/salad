
package:
	python3 setup.py sdist bdist_wheel

testupload:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload:
	twine upload dist/*

