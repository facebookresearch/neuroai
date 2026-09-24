# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Safe, DataFrame-free predicates for querying event objects."""

import ast
import dataclasses
import operator
import typing as tp

from neuralset.events import etypes

_ALLOWED_COMPARE_OPS: dict[type[ast.cmpop], tp.Callable[[tp.Any, tp.Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}


def _unknown_field_error(name: str, available: tp.Iterable[str]) -> ValueError:
    return ValueError(
        f"unknown field {name!r}; queryable fields: {', '.join(sorted(available))}"
    )


def _reject(node: ast.AST, reason: str) -> tp.NoReturn:
    raise ValueError(f"Unsupported event query syntax ({type(node).__name__}): {reason}")


def _validate_string_literal(node: ast.AST) -> None:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, str):
            _reject(
                node, f"unsupported literal {node.value!r}; only strings are supported"
            )
        return
    _reject(node, "only string literals are supported")


def _validate_string_literal_container(node: ast.AST) -> None:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        _reject(node, "'in' comparisons require a list, tuple, or set of string literals")
    for element in node.elts:
        _validate_string_literal(element)


def _validate(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate(node.body)
        return
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            _reject(node, "only 'and' and 'or' boolean operators are supported")
        for value in node.values:
            _validate(value)
        return
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            _validate(node.operand)
            return
        _reject(node, "only 'not' is supported")
    if isinstance(node, ast.Compare):
        if not isinstance(node.left, ast.Name):
            _reject(node, "comparisons must be of the form field == 'literal'")
        if "__" in node.left.id:
            raise ValueError(
                f"Unsupported event query identifier {node.left.id!r}: '__' is not allowed"
            )
        if len(node.ops) != 1 or len(node.comparators) != 1:
            _reject(node, "chained comparisons are not supported")
        op = node.ops[0]
        if type(op) not in _ALLOWED_COMPARE_OPS:
            _reject(node, f"comparison operator {type(op).__name__} is not supported")
        comparator = node.comparators[0]
        if isinstance(op, (ast.In, ast.NotIn)):
            _validate_string_literal_container(comparator)
        else:
            _validate_string_literal(comparator)
        return
    if isinstance(node, ast.Name):
        _reject(
            node, "bare field names are not supported; compare them to string literals"
        )
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)
    ):
        _reject(node, "use 'and'/'or' instead of '&'/'|'/'^'")
    _reject(node, "this construct is not supported")


def _string_literal_value(node: ast.AST) -> str:
    return tp.cast(str, tp.cast(ast.Constant, node).value)


def _string_literal_container_value(node: ast.AST) -> tp.Any:
    if isinstance(node, ast.List):
        return [_string_literal_value(element) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_string_literal_value(element) for element in node.elts)
    if isinstance(node, ast.Set):
        return {_string_literal_value(element) for element in node.elts}
    raise TypeError(f"Node {node!r} is not a string literal container")


def _available_fields(event: etypes.Event) -> list[str]:
    fields = set(type(event).model_fields) | set(event.extra)
    return sorted(fields)


def _resolve_name(event: etypes.Event, name: str) -> tp.Any:
    try:
        return event._get_field_or_extra(name)
    except ValueError as exc:
        raise _unknown_field_error(name, _available_fields(event)) from exc


@dataclasses.dataclass(frozen=True)
class _EventQueryPredicate:
    expr: str
    tree: ast.Expression

    def __call__(self, event: etypes.Event) -> bool:
        return bool(self._interpret(self.tree.body, event))

    def _interpret(self, node: ast.AST, event: etypes.Event) -> tp.Any:
        if isinstance(node, ast.BoolOp):
            # ``all``/``any`` short-circuit, so a name in an unreached branch is
            # never resolved (matching Python/pandas boolean semantics).
            if isinstance(node.op, ast.And):
                return all(self._interpret(value, event) for value in node.values)
            return any(self._interpret(value, event) for value in node.values)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not self._interpret(node.operand, event)
        if isinstance(node, ast.Compare):
            left_name = tp.cast(ast.Name, node.left).id
            left = _resolve_name(event, left_name)
            comparator = node.comparators[0]
            op = node.ops[0]
            right = (
                _string_literal_container_value(comparator)
                if isinstance(op, (ast.In, ast.NotIn))
                else _string_literal_value(comparator)
            )
            return _ALLOWED_COMPARE_OPS[type(op)](left, right)
        raise TypeError(f"Unsupported validated node: {node!r}")


def compile_event_query(
    expr: str, fields: tp.Collection[str] | None = None
) -> tp.Callable[[etypes.Event], bool]:
    """Compile an event-query expression into a safe object predicate.

    The expression is parsed and validated once. Evaluation walks the validated
    AST directly against an :class:`Event`; it never executes Python code.
    Supported queries are boolean formulas over simple string comparisons:
    ``field == "literal"``, ``field != "literal"``, ``field in ["literal"]``,
    and ``field not in ["literal"]`` combined with ``and``/``or``/``not``.
    Boolean operators short-circuit, so a name is only resolved when its branch
    is actually evaluated. Referencing a field that is genuinely absent on an
    evaluated event raises a clear ``ValueError`` (fail-fast on typos).

    Parameters
    ----------
    expr:
        The query expression (a deliberate subset of the pandas query dialect).
    fields:
        Optional collection of field names the query is allowed to reference.
        When provided, every referenced name must be in ``fields``, otherwise a
        ``ValueError`` is raised at compile time. This catches typos early —
        including names in branches that would short-circuit at evaluation — at
        the cost of rejecting per-event ``extra`` keys not declared on the event
        class. When ``None`` (default), names are only checked lazily at
        evaluation time against each event's declared fields and ``extra``.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Event query {expr!r} is not valid: {exc}") from exc
    _validate(tree)
    if fields is not None:
        allowed = set(fields)
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        unknown = sorted(referenced - allowed)
        if unknown:
            raise _unknown_field_error(unknown[0], allowed)
    return _EventQueryPredicate(expr=expr, tree=tree)
