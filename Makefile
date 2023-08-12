SOURCE_LIB = ./mab
EXAMPLES = ./examples
BENCHMARKING = ./benchmarking
TEST = ./tests

reformat:
	seed-isort-config
	flake8 $(SOURCE_LIB) --ignore=W503,E501,E203
	flake8 $(EXAMPLES) --ignore=W503,E501,E203
	flake8 $(BENCHMARKING) --ignore=W503,E501,E203
	flake8 $(TEST) --ignore=W503,E501,E203
	isort $(SOURCE_LIB)
	isort $(EXAMPLES)
	isort $(BENCHMARKING)
	isort $(TEST)
	black $(SOURCE_LIB)
	black $(EXAMPLES)
	black $(BENCHMARKING)
	black $(TEST)