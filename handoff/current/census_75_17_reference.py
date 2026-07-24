#!/usr/bin/env python3
"""
census_final.py -- hardened, extraction-strategy-agnostic adjudicator (75.17).
This is the REFERENCE SPEC for the committed sweep tool.

Design: MANY extractors (A structure-aware, B broad-regex, C whitespace+quoted,
D maximal-recall) feed ONE rigorous ADJUDICATOR. The adjudicator, not the
extractor, decides genuine-vs-FP. If the genuine set is identical whether fed by
{A,B} or {A,B,C,D}, the census is robust to extraction strategy => coverage dry.
"""
import json, re, os, subprocess
from pathlib import Path
ROOT=Path("/Users/ford/.openclaw/workspace/pyfinagent")
d=json.load(open(ROOT/".claude/masterplan.json"))
STEPS=[s for ph in d["phases"] for s in (ph.get("steps") or [])]
GIT=set(subprocess.run(["git","-C",str(ROOT),"ls-files"],capture_output=True,text=True).stdout.split())
BASE={p.split("/")[-1] for p in GIT}
EXT=r"(?:py|tsx|ts|jsx|js|md|json|sh|ya?ml|tsv|txt|plist|sql|html|css|cfg|ini|toml|lock|csv|pkl|joblib|proto|parquet)"
ANNOTATED={"4.17.9","4.14.24","4.14.4"}

def cmds(v):
    if v is None:return[]
    if isinstance(v,str):return[v]
    if isinstance(v,list):return[x for x in v if isinstance(x,str)]
    if isinstance(v,dict):
        c=v.get("command")
        if isinstance(c,str):return[c]
        if isinstance(c,list):return[x for x in c if isinstance(x,str)]
    return[]

# ---------- EXTRACTORS ----------
A_CTX=[re.compile(r"""open\(\s*['"]([^'"]+)['"]"""),
       re.compile(r"\btest\s+(?:!\s+)?-[a-zA-Z]+\s+([A-Za-z0-9_./+-]+\.\w+)"),
       re.compile(r"!\s*test\s+-[a-zA-Z]+\s+([A-Za-z0-9_./+-]+\.\w+)"),
       re.compile(r"\bpython3?\s+(?:-m\s+\S+\s+)?([A-Za-z0-9_./+-]+\.py)\b"),
       re.compile(r"\bpytest\s+([A-Za-z0-9_./+-]+\.py)\b"),
       re.compile(r"\bsource\s+([A-Za-z0-9_./+-]+)"),
       re.compile(r"\bcat\s+([A-Za-z0-9_./+-]+\.\w+)"),
       re.compile(r"\bbash\s+([A-Za-z0-9_./+-]+\.sh)\b")]
RE_EXT=re.compile(r"(?<![A-Za-z0-9_./+-])([A-Za-z0-9_][A-Za-z0-9_./+-]*\.%s)(?![A-Za-z0-9])"%EXT)
RE_DIR=re.compile(r"(?<![A-Za-z0-9_./+-])((?:backend|frontend|scripts|docs|tests|\.claude)/[A-Za-z0-9_./+-]+)")
def extA(c):
    o=set()
    for r in A_CTX:
        for m in r.finditer(c):o.add(m.group(1))
    return o
def extB(c):
    o=set()
    for r in(RE_EXT,RE_DIR):
        for m in r.finditer(c):o.add(m.group(1).rstrip("./"))
    return o
def extC(c):
    o=set()
    for m in re.finditer(r"""['"]([^'"]+)['"]""",c):o.add(m.group(1))
    for t in c.split():o.add(t)
    return {t for t in o if ("/" in t and re.search(r"\.%s(?![A-Za-z0-9])"%EXT,t)) or re.match(r"^[A-Za-z0-9_./+-]+\.%s$"%EXT,t)}
def extD(c):
    o=set()
    for m in re.finditer(r"[A-Za-z0-9_][A-Za-z0-9_./+-]*\.%s(?![A-Za-z0-9])"%EXT,c):o.add(m.group(0))
    for m in re.finditer(r"(?:python3?|pytest|source|bash)\s+([A-Za-z0-9_./+-]+)",c):o.add(m.group(1))
    return o

ABS=("/Users","/Library","/tmp","/var","/etc","/opt","/private","/Applications","/System","/bin","/usr","/sbin")
def clean(t):return t.strip().strip("'\"").rstrip(".,;:)").lstrip("(")

def fp_reason(tok,cmd):
    """Return an FP-class string if this token is NOT a genuine command-breaking absent path, else None."""
    t=clean(tok)
    if not t: return "empty"
    # well-formedness gate: a real path token is clean path chars only (allow leading dot, globs)
    if not re.match(r"^[.A-Za-z0-9_][A-Za-z0-9_./+*?\[\]-]*$", t): return "malformed-token"
    # boundary gate: token must appear as a WHOLE path in the command (not a mis-split of a neighbor)
    if not re.search(r"(?<![A-Za-z0-9_./+-])"+re.escape(t)+r"(?![A-Za-z0-9./+-])", cmd): return "mis-split"
    if "://" in t or t.startswith(("http","localhost","127.0.0.1")): return "url"
    if re.search(r"(localhost|127\.0\.0\.1)[:/0-9A-Za-z._?=&%-]*"+re.escape(t),cmd): return "url"
    if t.startswith("/") and not t.startswith(ABS): return "url-route"
    if t.startswith("/"): return None if not os.path.exists(t) and False else "abs-host-path"
    if t.startswith(("tmp/","handoff/","frontend/handoff/")): return "runtime/transient"
    # glob-prefix: token is a truncated glob (real command has token* / token? / token[..])
    e=re.escape(t)
    if re.search(e+r"[*?\[]", cmd):
        try:
            if list(ROOT.glob(t+"*")) or list((ROOT/"frontend").glob(t+"*")): return "glob-prefix-matches"
        except Exception: return "malformed-glob"
    # negative assertions (shell + python inline; full-path-aware so a bare basename still matches
    # an assert on its full path, and tolerated-missing `if os.path.exists(x) else ''` branches)
    for pat in (r"!\s*test\s+-[a-zA-Z]+\s+"+e, r"test\s+!\s+-[a-zA-Z]+\s+"+e, r"\[\s*!\s+-[a-zA-Z]+\s+"+e,
                r"test\s+-[a-zA-Z]+\s+"+e+r"\s*\|\|",
                r"assert\s+not\s+(?:any\()?os\.path\.exists\([^)]*"+e,
                r"not\s+os\.path\.exists\([^)]*"+e):
        if re.search(pat,cmd): return "absence-asserted"
    # tolerated-missing: command has an exists-guarded read with an empty-string else branch
    if re.search(r"os\.path\.exists\([^)]*\)\s+else\s+['\"]{2}", cmd) and re.search(e, cmd):
        return "absence-tolerated(else-empty)"
    # grep search-PATTERN (path appears inside a grep pattern, not as a file arg)
    if re.search(r"grep\b[^|;&]*?['\"][^'\"]*"+e, cmd) and not re.search(r"grep\b[^|;&]*\s"+e+r"(\s|$|\|)", cmd):
        # appears only inside a quoted grep pattern
        if re.search(r"grep[^|;&]*['\"][^'\"]*"+e+r"[^'\"]*['\"]", cmd): return "grep-search-pattern"
    # shell variable expansion ($f.md etc.)
    if re.search(r"\$\w*"+e, cmd) or re.search(r"\$\{?\w+\}?"+re.escape("/"+t.split("/")[-1]), cmd): return "shell-var"
    if "$" in tok: return "shell-var"
    # wildcard glob that matches
    if any(c in t for c in "*?["):
        try:
            return "glob-matches" if (list(ROOT.glob(t)) or list((ROOT/"frontend").glob(t))) else None
        except Exception: return "malformed-glob"
    # existence (repo-root, frontend, frontend/src, backend)
    if (ROOT/t).exists() or (ROOT/"frontend"/t).exists() or (ROOT/"frontend/src"/t).exists() or (ROOT/"backend"/t).exists():
        return "exists-on-disk"
    if "/" not in t and t in BASE: return "basename-exists-elsewhere"
    if "/" not in t and not re.search(r"\.%s$"%EXT,t): return "bare-word-no-ext"
    # inline python -c fragment (token has an unbalanced paren-ish neighbor or is a %s template)
    if "%s" in tok or re.search(r"%s\."+re.escape(t.split('.')[-1]),cmd): return "printf-template"
    return None  # GENUINE absent path

def gitclass(p):
    a=subprocess.run(["git","-C",str(ROOT),"log","--all","--diff-filter=A","--oneline","--",p],capture_output=True,text=True).stdout.strip()
    dl=subprocess.run(["git","-C",str(ROOT),"log","--all","--diff-filter=D","--oneline","--",p],capture_output=True,text=True).stdout.strip()
    if not a: return "never-existed",""
    if dl: return "retired", dl.split("\n")[0][:12]
    return "in-history-absent(runtime?)",""

def adjudicate(extractors):
    genuine={}   # sid -> [(path,class,retcommit)]
    for st in STEPS:
        if st.get("status")!="done": continue
        sid=st["id"]; v=st.get("verification")
        for cmd in cmds(v):
            cand=set()
            for e in extractors: cand|=e(cmd)
            for tok in cand:
                t=clean(tok)
                if not t: continue
                fp=fp_reason(tok,cmd)
                if fp: continue
                gc,rc=gitclass(t)
                genuine.setdefault(sid,[]).append((t,gc,rc))
    return genuine

AB=adjudicate([extA,extB])
ABCD=adjudicate([extA,extB,extC,extD])
def flat(g): return {(sid,p) for sid,rows in g.items() for (p,_,_) in rows}
print("GENUINE via {A,B}      :",len(AB),"steps,",len(flat(AB)),"paths")
print("GENUINE via {A,B,C,D}  :",len(ABCD),"steps,",len(flat(ABCD)),"paths")
print("Round 3 (add C) new genuine:", sorted({s for s,_ in flat(adjudicate([extA,extB,extC]))} - set(AB)) or "NONE -> DRY")
print("Round 4 (add D) new genuine:", sorted({s for s,_ in flat(ABCD)} - {s for s,_ in flat(adjudicate([extA,extB,extC]))}) or "NONE -> DRY")
print()
print("=== FINAL GENUINE SET (adjudicated, strategy-agnostic) ===")
for sid in sorted(ABCD, key=lambda x:[int(n) if n.isdigit() else 0 for n in re.split(r'[.]',re.sub(r'^phase-','',x))]):
    ann=" [ALREADY-ANNOTATED]" if sid in ANNOTATED else ""
    for (p,gc,rc) in ABCD[sid]:
        print(f"  {sid:10s} {gc:18s} {('ret='+rc) if rc else '':16s} {p}{ann}")
need = {sid for sid in ABCD if sid not in ANNOTATED}
print()
print("Steps needing a NEW superseded_record:", len(need), sorted(need,key=lambda x:[int(n) if n.isdigit() else 0 for n in re.split(r'[.]',x)]))
