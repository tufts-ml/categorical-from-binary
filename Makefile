# We use Make to encapsulate many of
# the shell commands for managing our projects. Each project
# has a `Makefile` and a `Makefile_gen.mk`. `Makefile_gen.mk` is a
# "canonical" file that should not be modified from what is
# defined in the template. If make fails to find a
# target in `Makefile`, it will fall back on the default
# commands in `Makefile_gen.mk`. Any given target in
# `Makefile_gen.mk` can be overridden in `Makefile`. In addition,
# you can define arbitrary, project-specific targets here.


##########################################
### Required Variables ###################
##########################################
NAME := categorical_from_binary
VERSION := $(shell cat VERSION)
SOURCES := $(shell find src -name '*.py')
TARBALL := dist/$(NAME)-$(VERSION).tar.gz

##########################################
### Project Specific Targets #############
##########################################

# Check your code for style, run your unit tests, and make a package
all: lint test test-integration clean-build build

## Example: Custom `clean` target to remove pyc files
## Note that this overrides the default `clean` behavior
## defined in `Makefile_ds.mk`.
#
# clean: clean-tarballs
# 	rm -rf *.pyc
#

## Example: Additional target to download artifacts
#
# models/model.pkl:
#	aws s3 sync s3://models/model201944.pkl models/model.pkl


test: env
	# We override the `test` recipe in Makefile_ds.mk so that
	# pytest will print output to screen when `make test` is run
	#
	# The '--' tells tox that everything afterwards is a command to run in its
	# environments (or arguments or options for that command)
	tox -- pytest -sv tests/unit


test-integration: env
	# We override the `test-integration` recipe in Makefile_ds.mk so that
	# pytest will print output to screen when `make test` is run
	tox -- pytest -sv tests/integration

test-all: test test-integration
##########################################
### REQUIRED TARGETS: DO NOT DELETE ######
##########################################

# The code below is required to fall back on the
# canonical Makefile defined in `Makefile_gen.mk`.
export NAME VERSION SOURCES TARBALL
%: force
	@$(MAKE) --makefile Makefile_gen.mk $@
force: ;
.PHONY: Makefile all
