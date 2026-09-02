"""Prototype: build the narrative deterministically from graph.edges.
NOT wired into the backend - it exists to measure the completeness ceiling."""
import re
from collections import defaultdict

def render(chunk):
    g = chunk.get("graph") or {}
    lab = {n["id"]: (n.get("label") or "").strip() for n in (g.get("nodes") or [])}
    edges = [(e["from"], e["to"], (e.get("condition") or "").strip())
             for e in (g.get("edges") or []) if lab.get(e["from"]) and lab.get(e["to"])]
    if not edges:
        return None
    out_e, indeg = defaultdict(list), defaultdict(int)
    for a, b, c in edges:
        out_e[a].append((b, c)); indeg[b] += 1
    starts = [n for n in lab if indeg[n] == 0 and out_e[n]]
    if not starts:
        starts = [n for n in lab if re.search(r'creation|start', lab[n], re.I)] or [edges[0][0]]
    start = starts[0]

    depth, order, seen, queue = {start: 0}, [start], {start}, [start]
    while queue:
        n = queue.pop(0)
        for b, _ in out_e[n]:
            if b not in seen:
                seen.add(b); depth[b] = depth[n] + 1
                order.append(b); queue.append(b)
    ends = [n for n in lab if not out_e[n]]
    pos = {n: k for k, n in enumerate(order)}

    # main chain: follow while exactly one forward edge
    main, node, chain = [], start, set()
    while True:
        fwd = [(b, c) for b, c in out_e[node]
               if depth.get(b, 99) > depth.get(node, 0) and b not in ends]
        if len(fwd) != 1 or fwd[0][0] in main:
            break
        nxt = fwd[0][0]
        chain.add((node, nxt))
        main.append(nxt); node = nxt

    used = set(chain)                       # ONLY the chain edges, not every main-main edge
    lines = [f"The workflow begins with {lab[start]}."]
    i = 0

    def emit_node(n, prefix):
        """One numbered step for node n, with its unused outgoing routes."""
        nonlocal i
        outs = [(b, c) for b, c in out_e[n] if (n, b) not in used]
        i += 1
        if len(outs) > 1:
            lines.append(f"{i}. {prefix}A decision is made:")
            for b, c in outs:
                back = pos.get(b, 99) < pos.get(n, 0)
                verb = "it returns to" if back else "it moves to"
                lines.append(f"   - If {c or 'approved'}, {verb} {lab[b]}.")
                used.add((n, b))
        elif len(outs) == 1:
            b, c = outs[0]
            back = pos.get(b, 99) < pos.get(n, 0)
            verb = "returns to" if back else "moves to"
            lines.append(f"{i}. {prefix}"
                         + (f"If {c}, the process {verb} {lab[b]}." if c
                            else f"The process {verb} {lab[b]}."))
            used.add((n, b))
        else:
            i -= 1

    # start node's own extra routes (branch at creation)
    for b, c in out_e[start]:
        if (start, b) not in used:
            emit_node(start, "")
            break

    for n in main:
        i += 1
        lines.append(f"{i}. The process moves to {lab[n]}.")
        for b, c in out_e[n]:
            if (n, b) in used:
                continue
            back = pos.get(b, 99) < pos.get(n, 0)
            if back:
                lines.append(f"   - If {c or 'sent back'}, it returns to {lab[b]}.")
                used.add((n, b))
        rest = [(b, c) for b, c in out_e[n] if (n, b) not in used]
        if len(rest) > 1:
            i += 1
            lines.append(f"{i}. A decision is made:")
            for b, c in rest:
                lines.append(f"   - If {c or 'approved'}, it moves to {lab[b]}.")
                used.add((n, b))
        elif len(rest) == 1:
            b, c = rest[0]
            lines.append(f"   - If {c or 'approved'}, it moves to {lab[b]}.")
            used.add((n, b))

    for n in order:
        if n == start or n in main or not out_e[n]:
            continue
        emit_node(n, f"From {lab[n]}, ")

    if ends:
        i += 1
        closers = sorted({lab[a] for a, b, c in edges if b in ends})
        lines.append(f"{i}. The workflow ends once closed by {', '.join(closers)}.")
    lines += ["", f"Source: {chunk['filename']}"]
    return "\n".join(lines)
