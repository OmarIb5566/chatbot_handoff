"""Score a workflow answer against the diagram it cites. Read-only, no LLM.

Model of an answer: a walk with an `anchor` (the step we are currently at).
- lead-in / numbered step with labels  -> edge(anchor, first label), anchor = last label
- numbered step with no label ("A decision is made:") -> anchor unchanged
- sub-bullet under a step              -> edge(anchor, first label in the bullet)
- any line with >=2 labels             -> edges between consecutive labels
"""
import re

def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()

def graph_of(chunk):
    g = chunk.get("graph") or {}
    lab = {n["id"]: (n.get("label") or "").strip() for n in (g.get("nodes") or [])}
    edges = [(lab.get(e["from"], ""), lab.get(e["to"], ""), (e.get("condition") or "").strip())
             for e in (g.get("edges") or [])]
    return lab, [(a, b, c) for a, b, c in edges if a and b]

def labels_in(line, labels):
    n = norm(line)
    spans = []
    for L in sorted(labels, key=len, reverse=True):
        nl = norm(L)
        if not nl:
            continue
        for m in re.finditer(r'(?<![a-z0-9])' + re.escape(nl) + r'(?![a-z0-9])', n):
            if not any(s <= m.start() < e for _, s, e in spans):
                spans.append((L, m.start(), m.end()))
    return [L for L, _, _ in sorted(spans, key=lambda x: x[1])]

def implied_routes(answer, labels):
    implied, anchor = set(), None
    for raw in answer.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        ls = labels_in(line, labels)
        is_step = bool(re.match(r'^\s*\d+[.)]', line))
        is_bullet = bool(re.match(r'^\s*[-*•]', line))
        is_lead = bool(re.search(r'\bbegins with\b', line, re.I))
        if re.match(r'^\s*(returns and loops|send-backs)', line, re.I):
            anchor = None          # a trailing list of edges, not a continuation of the walk
            continue
        for a, b in zip(ls, ls[1:]):
            implied.add((a, b))
        if is_lead and ls:
            anchor = ls[-1]
        elif is_step:
            if ls:
                if anchor:
                    implied.add((anchor, ls[0]))
                anchor = ls[-1]
        elif is_bullet:
            if ls and anchor:
                implied.add((anchor, ls[0]))
        elif ls:
            if anchor:
                implied.add((anchor, ls[0]))
            anchor = ls[-1]
    return implied

def anchors_for_decisions(answer, labels):
    """Which node each 'A decision is made' line sits on."""
    out, anchor = [], None
    for raw in answer.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        ls = labels_in(line, labels)
        if re.search(r'a decision is made', line, re.I):
            out.append(ls[-1] if ls else anchor)
        if re.match(r'^\s*\d+[.)]', line) or re.search(r'\bbegins with\b', line, re.I):
            if ls:
                anchor = ls[-1]
    return out

def score(answer, chunk):
    lab, edges = graph_of(chunk)
    labels = set(lab.values())
    gold = {(a, b) for a, b, _ in edges}
    implied = implied_routes(answer, labels)
    implied = {(a, b) for a, b in implied if a != b}
    covered = gold & implied
    outdeg = {}
    for a, b, _ in edges:
        outdeg[a] = outdeg.get(a, 0) + 1
    bad_dec = [n for n in anchors_for_decisions(answer, labels)
               if n is not None and outdeg.get(n, 0) < 2]
    body = norm(chunk.get("text"))
    quoted = re.findall(r'"([^"]{3,80})"', answer)
    bad_cond = [q for q in quoted if norm(q) and norm(q) not in body]
    return {
        "gold": len(gold), "covered": len(covered),
        "pct": round(100 * len(covered) / len(gold), 1) if gold else None,
        "missing": sorted(gold - covered),
        "asserted_not_in_diagram": sorted(implied - gold),
        "unsupported_quoted_conditions": bad_cond,
        "decisions_on_non_branching_step": bad_dec,
        "pdf_mentions": len(re.findall(r'\.pdf', answer)),
    }


def branch_faults(answer, chunk):
    """Sub-bullets under 'A decision is made:' must use that node's OWN routes.

    Two faults are possible and they are different in kind:
      borrowed_condition - the condition text is real, but it is written on a
                           different node's route. This is the failure that
                           produced a fabricated threshold at Finance Review.
      wrong_target       - the bullet routes to a node this step has no edge to.
    """
    lab, edges = graph_of(chunk)
    labels = set(lab.values())
    own_conds, succ, all_conds = {}, {}, set()
    for a, b, c in edges:
        succ.setdefault(a, set()).add(b)
        if c:
            own_conds.setdefault(a, set()).add(norm(c))
            all_conds.add(norm(c))
    borrowed, wrong = [], []
    anchor, in_decision = None, False
    for raw in answer.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        ls = labels_in(line, labels)
        if re.match(r'^\s*\d+[.)]', line) or re.search(r'\bbegins with\b', line, re.I):
            in_decision = bool(re.search(r'a decision is made', line, re.I))
            if ls:
                anchor = ls[-1]
            continue
        if in_decision and re.match(r'^\s*[-*\u2022]', line) and anchor:
            tgt = ls[0] if ls else None
            if tgt and tgt not in succ.get(anchor, set()):
                wrong.append((anchor, tgt))
            cands = re.findall(r'"([^"]{3,80})"', line)
            if not cands:
                m = re.match(r'\s*[-*\u2022]\s*If\s+(.+?)\s*,', line)
                cands = [m.group(1)] if m else []
            cands = [q for q in cands
                     if norm(q) not in ("approved", "sent back", "further action is required")]
            for q in cands:
                nq = norm(q)
                if not nq:
                    continue
                mine = any(nq in c or c in nq for c in own_conds.get(anchor, set()))
                elsewhere = any(nq in c or c in nq for c in all_conds)
                if elsewhere and not mine:
                    borrowed.append((anchor, q.strip()))
                    break
    return {"borrowed_conditions": borrowed, "wrong_branch_targets": wrong}
