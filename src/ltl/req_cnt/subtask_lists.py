"""
Subtask list types for representing structured relationships between subtasks.
"""


class _LogicList(list):
    """Abstract base for logic-flavoured lists.

    Stores heterogeneous elements (including nested _LogicList instances)
    and renders them with the subclass-defined operator symbol.
    """
    def __init__(self, *elements):
        super().__init__(elements)

    def __repr__(self) -> str:
        l = ", ".join(repr(e) for e in self)
        return f"{type(self).__name__}({l})"

    def __format__(self, format_spec: str) -> str:
        if format_spec == "":
            return repr(self)
        elif format_spec == "pretty":
            l = ",\n".join(f"{e:pretty}" if isinstance(e, _LogicList) else repr(e) for e in self)
            lines = l.split('\n')
            if len(lines) > 1 or len(l) > 60:
                l = "\n".join("    " + e for e in lines)
                return f"{type(self).__name__}(\n{l}\n)"
            else:
                l = "\n".join(lines)
                return f"{type(self).__name__}({l})"
        else:
            raise ValueError("Unrecognized format spec for _LogicList: " + format_spec)


class AndSubtasks(_LogicList):
    """
    A conjunction of subtasks.
    Semantically: A and B and C ...
    """


class OrSubtasks(_LogicList):
    """
    A disjunction of subtasks.
    Semantically: A or B or C ...
    """


class SeqSubtasks(_LogicList):
    """
    A sequence of subtasks.
    Semantically: A then B then C ...
    """


# Quick smoke-test
if __name__ == "__main__":
    empty = AndSubtasks()
    a = AndSubtasks("raining", "umbrella_present")
    o = OrSubtasks("cash", "card", "phone_pay")
    s = SeqSubtasks("wake_up", "shower", "breakfast")

    nested = SeqSubtasks("login", OrSubtasks("2fa_sms", "2fa_app"), "access_granted", a, o, s, SeqSubtasks("single item"))

    for obj in (empty, a, o, s, nested):
        print(repr(obj))
        print(f"{obj:pretty}")