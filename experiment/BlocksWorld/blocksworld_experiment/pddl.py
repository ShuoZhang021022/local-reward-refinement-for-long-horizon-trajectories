"""Read the untyped Blocksworld-4ops problems emitted by PlanBench's generator."""

from __future__ import annotations

from pathlib import Path
import re

from .domain import Atom, Problem, State


def _parse_sexpression(source: str) -> list:
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", source).lower())
    stack: list[list] = []
    root: list | None = None
    for token in tokens:
        if token == "(":
            node: list = []
            if stack:
                stack[-1].append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("Unbalanced PDDL parentheses")
            root = stack.pop()
        else:
            if not stack:
                raise ValueError("PDDL token outside a list")
            stack[-1].append(token)
    if stack or root is None or root[0] != "define":
        raise ValueError("Expected one complete PDDL define expression")
    return root


def _atom(node: list) -> Atom:
    if not isinstance(node, list) or not node or not all(isinstance(x, str) for x in node):
        raise ValueError(f"Expected a positive ground atom: {node!r}")
    predicate, *arguments = node
    arity = {"handempty": 0, "clear": 1, "ontable": 1,
             "holding": 1, "on": 2}
    if predicate not in arity or len(arguments) != arity[predicate]:
        raise ValueError(f"Unknown predicate or wrong arity: {node!r}")
    return Atom(predicate, tuple(arguments))


def parse_problem(source: str, *, required_blocks: int | None = None) -> Problem:
    tree = _parse_sexpression(source)
    sections = {}
    for section in tree[1:]:
        if isinstance(section, list) and section and isinstance(section[0], str):
            sections[section[0]] = section[1:]
    if sections.get(":domain") != ["blocksworld-4ops"]:
        raise ValueError("Expected the PlanBench Blocksworld-4ops domain")
    objects = sections.get(":objects")
    if not objects or not all(isinstance(x, str) and x != "-" for x in objects):
        raise ValueError("Expected an untyped nonempty object list")
    blocks = tuple(sorted(objects))
    if len(blocks) != len(set(blocks)):
        raise ValueError("Duplicate object name")
    if required_blocks is not None and len(blocks) != required_blocks:
        raise ValueError(f"Expected {required_blocks} blocks, found {len(blocks)}")
    raw_init = sections.get(":init")
    raw_goal = sections.get(":goal")
    if raw_init is None or raw_goal is None or len(raw_goal) != 1:
        raise ValueError("Missing or malformed init/goal")
    init = frozenset(_atom(node) for node in raw_init)
    goal_node = raw_goal[0]
    if not isinstance(goal_node, list):
        raise ValueError("Malformed goal")
    raw_atoms = goal_node[1:] if goal_node and goal_node[0] == "and" else [goal_node]
    goal = tuple(sorted({_atom(node) for node in raw_atoms}))

    if Atom("handempty") not in init or any(atom.predicate == "holding" for atom in init):
        raise ValueError("PlanBench generator initial states must have an empty hand")
    bottoms = [atom.arguments[0] for atom in init if atom.predicate == "ontable"]
    if len(bottoms) != len(set(bottoms)):
        raise ValueError("Repeated ontable fact")
    above: dict[str, str] = {}
    supported: set[str] = set()
    for atom in init:
        if atom.predicate == "on":
            upper, lower = atom.arguments
            if lower in above or upper in supported:
                raise ValueError("Invalid branching or repeated support")
            above[lower] = upper
            supported.add(upper)
    stacks = []
    for bottom in sorted(bottoms):
        stack = [bottom]
        while stack[-1] in above:
            stack.append(above[stack[-1]])
            if len(stack) > len(blocks):
                raise ValueError("Cycle in initial state")
        stacks.append(tuple(stack))
    state = State(tuple(stacks))
    if state.blocks != blocks or state.atoms != init:
        raise ValueError("Initial PDDL facts do not describe a complete physical state")
    return Problem(blocks, state, goal)


def load_problem(path: Path, *, required_blocks: int | None = None) -> Problem:
    return parse_problem(path.read_text(encoding="utf-8"),
                         required_blocks=required_blocks)
