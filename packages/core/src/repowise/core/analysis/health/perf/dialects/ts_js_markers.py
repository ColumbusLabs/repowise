"""The TS/JS anti-pattern marker hooks: list membership, deep-clone, spread-in-reduce,
and client construction inside a loop."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from .base import BasePerfDialect

if TYPE_CHECKING:
    from tree_sitter import Node

# Heavy client/connection classes that should be hoisted, not re-``new``-ed each
# iteration. Distinctive names only (``Client`` / ``Pool`` collide with worker
# pools and unrelated SDKs, so they are deliberately excluded for precision).
_TS_RESOURCE_CTORS: frozenset[str] = frozenset(
    {
        "PrismaClient",
        "MongoClient",
        "Sequelize",
        "DataSource",
        "Redis",
        "IORedis",
    }
)

_FN_LITERAL_KINDS: frozenset[str] = frozenset({"arrow_function", "function", "function_expression"})


# -- TS/JS syntax helpers (shared with ts_js.py) -------------------------------


def node_text(node: Node | None) -> str | None:
    """*node*'s source text, or None when there is no node or no text."""
    if node is None or node.text is None:
        return None
    return node.text.decode("utf-8", "replace")


def identifier_name(node: Node | None) -> str | None:
    """The name *node* spells when it is a bare identifier."""
    return node_text(node) if node is not None and node.type == "identifier" else None


def first_named_child(node: Node) -> Node | None:
    return next((c for c in node.children if c.is_named), None)


# -- marker predicates ---------------------------------------------------------


def _param_names(fn: Node) -> Iterator[str]:
    """Names of an arrow/function's identifier parameters, in order.

    A destructured parameter binds no single name and is skipped.
    """
    params = fn.child_by_field_name("parameters")
    if params is None:
        # ``x => …`` single unparenthesized param.
        candidates = [first_named_child(fn)]
    else:
        candidates = [
            c if c.type == "identifier" else c.child_by_field_name("pattern")
            for c in params.children
        ]
    for candidate in candidates:
        name = identifier_name(candidate)
        if name is not None:
            yield name


def _is_json_deep_clone(call: Node) -> bool:
    """True if the call's first argument is itself a ``JSON.stringify(...)`` call."""
    args = call.child_by_field_name("arguments")
    first = first_named_child(args) if args is not None else None
    if first is None or first.type != "call_expression":
        return False
    return node_text(first.child_by_field_name("function")) == "JSON.stringify"


def _reduce_spreads_accumulator(call: Node) -> bool:
    """A ``reduce`` whose callback spreads its OWN accumulator param into a fresh
    array / object literal."""
    args = call.child_by_field_name("arguments")
    cb = first_named_child(args) if args is not None else None
    if cb is None or cb.type not in _FN_LITERAL_KINDS:
        return False
    acc = next(_param_names(cb), None)
    body = cb.child_by_field_name("body")
    return acc is not None and body is not None and _spreads_name_in_collection(body, acc)


def _spreads_name_in_collection(body: Node, name: str) -> bool:
    """True if *body* spreads ``name`` into an array / object literal.

    A nested arrow/function that re-binds ``name`` as a parameter (an inner
    ``reduce`` with its own ``acc``) is skipped: its spreads target that binding.
    """
    stack: list[Node] = [body]
    while stack:
        n = stack.pop()
        if n != body and n.type in _FN_LITERAL_KINDS and name in _param_names(n):
            continue
        if _is_collection_spread_of(n, name):
            return True
        stack.extend(n.children)
    return False


def _is_collection_spread_of(node: Node, name: str) -> bool:
    """``[...name]`` / ``{...name}``."""
    parent = node.parent
    if node.type != "spread_element" or parent is None or parent.type not in ("array", "object"):
        return False
    return identifier_name(first_named_child(node)) == name


# A literal table up to this size is a fixed set; past it a linear probe can
# still cost enough to be worth a ``Set``. Ceiling: the size is a judgement, not
# a measurement.
_FIXED_TABLE_MAX = 64
_SCALAR_KINDS: frozenset[str] = frozenset({"string", "number", "true", "false", "null"})
_GROWING_METHODS: frozenset[str] = frozenset({"push", "unshift", "splice"})


def _is_fixed_table(declarator: Node, array: Node) -> bool:
    """``const xs = ['a', 1, `b`]``: a non-empty ``const`` literal of at most
    ``_FIXED_TABLE_MAX`` scalars. An empty literal is a container filled later."""
    decl = declarator.parent
    if decl is None or decl.type != "lexical_declaration" or decl.children[0].type != "const":
        return False
    items = [c for c in array.children if c.is_named and c.type != "comment"]
    return 0 < len(items) <= _FIXED_TABLE_MAX and all(_is_scalar(item) for item in items)


def _is_scalar(item: Node) -> bool:
    if item.type == "template_string":
        return not any(c.type == "template_substitution" for c in item.children)
    return item.type in _SCALAR_KINDS


def _grown_array(call: Node) -> list[str]:
    """``xs`` when *call* is ``xs.push(...)`` / ``unshift`` / ``splice``."""
    fn = call.child_by_field_name("function")
    if fn is None or fn.type != "member_expression":
        return []
    name = identifier_name(fn.child_by_field_name("object"))
    method = node_text(fn.child_by_field_name("property"))
    return [name] if name is not None and method in _GROWING_METHODS else []


class TsJsMarkerHooks(BasePerfDialect):
    def loop_call_marker(
        self, root: str, method: str, node: Node, list_names: frozenset[str]
    ) -> str | None:
        # ``arr.includes(x)`` where ``arr`` is a known array -> O(n) membership.
        if method == "includes" and root in list_names:
            return "membership_test_against_list_in_loop"
        # Only the ``JSON.parse(JSON.stringify(x))`` deep clone: a bare parse or
        # stringify per iteration converts a distinct payload each pass, which is
        # necessary work.
        if root == "JSON" and method == "parse" and _is_json_deep_clone(node):
            return "json_parse_in_loop"
        return None

    def bare_call_marker(self, root: str, method: str, node: Node) -> str | None:
        # ``arr.reduce((acc, x) => [...acc, x], [])`` rebuilds the accumulator every
        # step (O(n^2)). The ``.reduce`` is the loop, so this fires at any depth.
        if method == "reduce" and _reduce_spreads_accumulator(node):
            return "array_spread_in_reduce"
        return None

    def loop_stmt_marker(self, node: Node, list_names: frozenset[str]) -> str | None:
        # ``new PrismaClient(...)`` etc. is a ``new_expression`` (not a
        # ``call_expression``), so it arrives here rather than via the call path.
        if node.type != "new_expression":
            return None
        # Rightmost identifier of a ``new X()`` / ``new pkg.X()`` constructor.
        ctor = node_text(node.child_by_field_name("constructor"))
        if ctor is not None and ctor.split(".")[-1] in _TS_RESOURCE_CTORS:
            return "resource_construction_in_loop"
        return None

    def list_bound_names(self, root: Node) -> frozenset[str]:
        """Names bound to an array literal (``const arr = [...]``), less fixed tables.

        A ``const`` bound to a short literal of scalars that the file never grows
        (``const booleanAttributes = ['async', 'checked', ...]``) has a fixed
        length, so a probe per iteration is not the O(n·m) the marker means.
        """
        names: set[str] = set()
        fixed: set[str] = set()
        grown: set[str] = set()
        for n in self._walk(root):
            if n.type == "call_expression":
                grown.update(_grown_array(n))
                continue
            if n.type != "variable_declarator":
                continue
            name = identifier_name(n.child_by_field_name("name"))
            value = n.child_by_field_name("value")
            if name is not None and value is not None and value.type == "array":
                (fixed if _is_fixed_table(n, value) else names).add(name)
        return frozenset(names | (fixed & grown))
