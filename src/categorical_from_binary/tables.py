from typing import Dict

import latextable
import numpy as np
import texttable


def table_from_dict(my_dict: Dict) -> texttable.Texttable:
    # TODO: make alignment choices a function argument with defaults
    np.set_printoptions(precision=3, suppress=True)
    table = texttable.Texttable()
    table.set_cols_align(["p{0.35\linewidth} ", "p{0.6\linewidth} "])
    table.set_cols_valign(["t", "t"])
    table.set_cols_dtype(
        [
            "t",  # text
            "a",  # auto
        ]
    )
    table_rows = [
        list((field.replace("_", " "), value)) for (field, value) in my_dict.items()
    ]
    table.add_rows(table_rows)
    return table


def table_from_namedtuple(my_namedtuple) -> texttable.Texttable:
    return table_from_dict(my_namedtuple._asdict())


def print_namedtuple(my_namedtuple) -> None:
    table = table_from_namedtuple(my_namedtuple)
    print(table.draw() + "\n")


def print_namedtuple_as_latex_table(my_namedtuple, caption) -> None:
    table = table_from_namedtuple(my_namedtuple)
    print(table.draw() + "\n")
    print(latextable.draw_latex(table, caption=caption) + "\n")
