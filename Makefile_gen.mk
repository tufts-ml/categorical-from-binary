#  ____________________________________________________
# < DO NOT MODIFY THIS FILE!!!!!! 					   >
# < IT IS DESIGNED TO BE IDENTICAL IN ALL OUR PROJECTS >
# < YOU MAY OVERRIDE ANY OF THE COMMANDS IN `Makefile` >
#  ----------------------------------------------------
#         \    ,-^-.
#          \   !oYo!
#           \ /./=\.\______
#                ##        )\/\
#                 ||-----w||
#                 ||      ||

# Run your unit tests after ensuring dependencies are up-to-date
test: env
	# The '--' tells tox that everything afterwards is a command to run in its
	# environments (or arguments or options for that command)
	tox -- pytest tests/unit

# Run your integration tests after ensuring dependencies are up-to-date
test-integration: env
	tox -- pytest tests/integration

# Start a watcher that re-runs your unit tests when the tests or the code change
# If this doesn't work, make sure `pytest-watch` is in your dev-requirements.txt
watch: env
	# Only run in py3 env because the test watcher doesn't naturally exit, and tox
	# runs commands serially rather than in parallel.
	tox -e env -- ptw

# Check your code for style conformance
lint: $(SOURCES) env
	env/bin/isort --diff -rc src tests
	env/bin/black --diff src tests
	env/bin/flake8 src tests

# This defines 'pylint' as an alias for 'lint', so `make pylint` == `make lint`
pylint: lint

format: env
	env/bin/isort -rc src tests 
	env/bin/black src tests

# Make a compressed tarball of the source files and metadata only (no tests).
# This target will only run if code has changed since the last build.
# Copying the tarball to a remote server is an easy way to move your code around
build: $(TARBALL) # this is another alias definition
$(TARBALL): $(SOURCES) env
	tox -e env -- python setup.py sdist
	@ls dist/* | tail -n 1 | xargs echo "Created source tarball"

docker-build:
	mkdir -p dist
	docker build . -t $(NAME):latest -t $(NAME):$(VERSION)
	docker run --rm --env VERSION="$(VERSION)" $(NAME) > $(TARBALL)

# Install is now handled automatically by tox, so this is simply another alias
install: env

# Install all dependencies from requirements files
env: requirements.txt dev-requirements.txt .pre-commit-config.yaml
# Tox doesn't automatically recreate an environment if requirements change
	tox -vv -e env --notest --recreate
	if [ -d ".git" ]; then tox -vv -e env -- pre-commit install; fi
# Tox's operations may not update the modified time on the env folder, so unless
# we do that ourselves, make will constantly rebuild this target after either of
# the requirements files have been changed.
	@touch env

# Force reinstallation of dependencies
force-env: clean-env env
# (This is also an alias definition, but for running multiple targets in order)

# clean random files left from testing, etc, change these as needed
clean: clean-tarballs
	rm -rf *csv
	rm -rf *h5
	rm -rf src/$(NAME)/*.h5
	rm -rf *pkl
	rm -rf *png
	rm -rf annoy_index_file
	rm -rf X.np.txt
	rm -rf projecion.np.txt
	rm -rf coverage.xml
	rm -rf nosetests.xml
	rm -rf pylint.out
	rm -rf build/
	rm -rf src/$(NAME).egg-info/
	rm -rf pip-wheel-metadata/
	find src tests -name "*.pyc" -type f -delete
	find src tests -name "__pycache__" -type d -delete


clean-env:
	rm -rf env/

clean-build:
	rm -rf dist/

clean-tarball: clean-tarballs
clean-tarballs:
	rm -f dist/*.tar.gz

# Print help
help:
	python help.py

# Mark these targets as 'phony', which means Make considers them always out of
# date so they're always re-run.
.PHONY: clean-build clean-env clean-tarball force-env test test-integration watch update-make help format

# Phony targets are useful in two cases:
#   * when the only point of the target is to clean some filesystem state;
#   * when the target runs tests.
