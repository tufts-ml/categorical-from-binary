"""We use `make` to orchestrate common tasks
such as creating virtual environments, running tests, building
packages, etc.

To print this documentation run

$ make help

"""
from __future__ import print_function

import re


TAB_LEN = len(str.expandtabs("\t"))


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def read_makefile(makefile="Makefile_ds.mk"):
    with open(makefile, "r") as f:
        return f.readlines()


def parse_commands(lines):
    commands = {}
    for line_no, line in enumerate(lines):
        m = re.match(r"^([a-z-]+):", line)
        if m and m.group() != "force:":
            commands[m.group()] = get_doc(lines[:line_no])
    return commands


def get_doc(lines):
    doc = []
    for line in reversed(lines):
        if line.startswith("#"):
            doc.append(line)
        else:
            break
    doc = list(reversed(doc))
    return "".join(doc).strip()


def print_commands(commands):
    for k, v in commands.items():
        if len(k) < TAB_LEN and v:
            end = ""
        else:
            end = "\n"
        print(bcolors.OKGREEN + k + bcolors.ENDC, end=end)
        if v:
            print(re.sub("#+", "\t", v))
        print(end="\n")


if __name__ == "__main__":
    print(bcolors.OKBLUE + __doc__, end="Other Commands:\n\n" + bcolors.ENDC)
    default_commands = parse_commands(read_makefile())
    custom_commands = parse_commands(read_makefile("Makefile"))
    custom_commands.update(default_commands)
    print_commands(custom_commands)
