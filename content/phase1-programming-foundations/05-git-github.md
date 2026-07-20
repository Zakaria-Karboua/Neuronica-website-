# Phase 1 · Lesson 5 — Git & GitHub

> Prerequisite: Lesson 6 (Linux & Bash) is helpful but not required; can be read in either order.

---

## 1. Introduction

### What is Git?
Git is a distributed version control system (DVCS) created by Linus Torvalds in 2005 to manage the Linux kernel's source code after a licensing dispute with the previous tool (BitKeeper). GitHub (2008) is a hosting platform built around Git, adding collaboration features (pull requests, issues, Actions/CI).

### Why does it exist?
Before DVCS, teams used centralized version control (CVS, SVN) requiring constant network access to a single server, with weak branching support. Git decentralizes the *entire history* to every clone, making branching/merging cheap (a branch is just a movable pointer) and enabling offline work — critical as software teams globalized.

### Historical background
Torvalds built Git in about 10 days in April 2005, prioritizing speed and integrity (every object is content-addressed by SHA hash) over ease of use — a tradeoff still felt in Git's notoriously unfriendly CLI. GitHub Actions (2019) and, more recently, tight AI-tool integrations (Copilot code review, PR summarization) have made GitHub itself an AI-native platform in 2026, not just a code host.

### Real-world motivation
Every ML project — from a solo Kaggle notebook to a multi-team foundation-model training repo — lives in Git. You cannot collaborate, reproduce experiments, or roll back a broken model-training script without it.

---

## 2. Theory

### The Git object model (this is the single most important mental model)
Git stores four object types, all content-addressed by SHA-1/SHA-256 hash:
- **blob**: raw file content (no filename — just bytes).
- **tree**: a directory listing (maps names → blob/tree hashes), like a filesystem snapshot.
- **commit**: a pointer to one tree + parent commit(s) + metadata (author, message, timestamp).
- **tag**: a named, optionally signed pointer to a commit (releases).

A **branch** is nothing more than a mutable pointer (a file containing a commit hash) that advances automatically on each new commit. `HEAD` is a pointer to "the branch you're currently on" (or directly to a commit, in "detached HEAD" state).

### Key terminology
| Term | Meaning |
|---|---|
| Working directory | Your actual files on disk |
| Staging area (index) | A snapshot-in-progress you build with `git add` before committing |
| Repository (`.git/`) | The full object database + refs |
| Remote | A named reference to another repository (usually `origin`) |
| Fast-forward merge | Moving a branch pointer forward with no divergent history to reconcile |
| Three-way merge | Combining divergent histories using a common ancestor commit |
| Rebase | Rewriting commits to sit on top of a new base, producing linear history |

---

## 3. Mathematical Foundations

### Content addressing (hashing)
Every Git object's identity is `SHA-1(object_type + " " + length + "\0" + content)` (Git is transitioning toward SHA-256 for new repos as of recent versions, for collision resistance). This gives:
- **Deduplication for free**: identical file content anywhere in history is stored once.
- **Integrity verification**: any corruption changes the hash, immediately detectable — Git history is effectively a Merkle DAG (directed acyclic graph), the same structure underlying blockchains.

$$
\text{commit}_n \to \text{tree}_n \to \{\text{blob or tree}, ...\}
$$
$$
\text{commit}_n.\text{parent} = \text{commit}_{n-1}
$$

### The commit graph as a DAG
History is a directed acyclic graph where merge commits have 2+ parents. Common-ancestor finding for merges is a graph problem: Git computes the **merge base** via a variant of lowest common ancestor (LCA) search over this DAG — directly reusing graph-traversal ideas from Lesson 4.

---

## 4. Algorithm — Three-Way Merge (conceptual)

```
GIVEN: base commit (common ancestor), branch A (ours), branch B (theirs)
FOR each file:
  IF only A changed it since base: take A's version
  IF only B changed it since base: take B's version
  IF both changed it identically: take that version
  IF both changed it differently: CONFLICT -> insert conflict markers, human resolves
```
This is why Git can auto-merge most changes: it only needs human input when *both* branches touched the *same lines* differently — a diff/patch reconciliation algorithm, not magic.

---

## 5. Practical Implementation (Command Reference as "Code")

```bash
# --- initial setup ---
git init                                   # create a new repo
git clone https://github.com/user/repo.git # copy a remote repo + full history

# --- daily workflow ---
git status                                 # what's changed
git add file.py                            # stage a specific file
git add -p                                 # stage interactively, hunk by hunk (essential skill)
git commit -m "Add mortality model training script"
git log --oneline --graph --all            # visualize the commit DAG

# --- branching ---
git switch -c feature/xgboost-tuning       # create + switch to new branch (modern syntax)
git switch main                            # switch back
git merge feature/xgboost-tuning           # merge into current branch
git rebase main                            # replay current branch's commits on top of main

# --- collaboration ---
git remote add origin https://github.com/user/repo.git
git push -u origin main                    # push + set upstream tracking
git fetch origin                           # download remote changes WITHOUT merging
git pull --rebase                          # fetch + rebase onto remote (cleaner history than default merge)

# --- undoing mistakes (the part everyone actually needs) ---
git restore file.py                        # discard uncommitted changes to a file
git restore --staged file.py               # unstage (keep changes in working dir)
git reset --soft HEAD~1                    # undo last commit, keep changes staged
git revert <commit-hash>                   # create a NEW commit that undoes an old one (safe for shared history)
git reflog                                 # your safety net: shows EVERY HEAD movement, even after "losing" commits
```

**Line-by-line rationale for the two most misunderstood commands:**
- `git reset --hard` rewrites history locally and discards changes — never use on shared/pushed commits.
- `git revert` is the *safe* undo for anything already pushed/shared, because it adds new history instead of rewriting old history that others may have already based work on.

---

## 6. Build From Scratch (Conceptual Mini-Implementation)

To demystify Git's object model, here's a toy content-addressable store in Python mirroring its core idea:

```python
import hashlib, json, os

STORE_DIR = ".toygit/objects"

def hash_object(content: bytes, obj_type: str = "blob") -> str:
    header = f"{obj_type} {len(content)}\0".encode()
    full = header + content
    sha = hashlib.sha1(full).hexdigest()
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(os.path.join(STORE_DIR, sha), "wb") as f:
        f.write(full)
    return sha

def read_object(sha: str) -> bytes:
    with open(os.path.join(STORE_DIR, sha), "rb") as f:
        return f.read().split(b"\0", 1)[1]

# A "commit" is just a JSON blob referencing a tree hash + parent
def make_commit(tree_sha: str, parent_sha: str | None, message: str) -> str:
    commit_data = json.dumps({"tree": tree_sha, "parent": parent_sha, "message": message}).encode()
    return hash_object(commit_data, "commit")
```
This is deliberately simplified (no packfiles, no delta compression, no actual tree recursion) but captures the essential truth: **a commit is a hash pointing to other hashes, forming an immutable, verifiable DAG.**

---

## 7. Library/Tool Comparison

| Toy implementation | Real Git |
|---|---|
| SHA-1 of raw content | SHA-1 (legacy) / SHA-256 (modern), plus zlib compression of objects |
| One file per object | Packfiles (delta-compressed, batched objects) for efficiency at scale |
| No branches | Full ref system (`refs/heads/*`, `refs/remotes/*`, `refs/tags/*`) |
| No merge logic | Full three-way merge, rename detection, conflict markers |

---

## 8. Visual Explanations

```
Commit DAG:
  A ── B ── C  (main)
        \
         D ── E  (feature/xgboost-tuning)

After `git switch main && git merge feature/xgboost-tuning`:
  A ── B ── C ─────── M  (main)  <- merge commit, 2 parents (C and E)
        \            /
         D ──────── E
```

**Staging area as a "photo you're composing before taking the picture":**
```
Working Dir  --git add-->  Staging Area  --git commit-->  Repository (.git objects)
   (messy)                  (curated)                       (permanent snapshot)
```

---

## 9. Practical Examples

**Simple:** initialize a repo, make 3 commits, view history with `git log --oneline`.
**Medium:** simulate a merge conflict (edit the same line on two branches), resolve it manually, commit the resolution.
**Real-world:** set up a `.gitignore` for a Python ML project excluding `__pycache__/`, `*.pyc`, `.venv/`, large model checkpoint files (`*.pt`, `*.ckpt`) — and instead track large binaries via **Git LFS**.

```
# .gitignore for ML projects
__pycache__/
*.pyc
.venv/
*.ckpt
*.pt
wandb/
mlruns/
.env
```

---

## 10. Real Industry Use Cases

- **Hugging Face Hub**: uses Git (+ Git LFS) as the underlying storage for every model/dataset repository — `git clone` works directly on model repos.
- **Every AI lab's monorepo**: training code, eval harnesses, and infra-as-code live in Git, with GitHub Actions running CI on every PR (linting, unit tests, sometimes small-scale training smoke tests).
- **DVC (Data Version Control)**: builds directly on Git's object model to version large datasets/model weights without bloating the Git repo itself.
- **MLflow/Weights & Biases**: often log the exact Git commit hash alongside each experiment run — reproducibility requires knowing precisely which code produced which model.

---

## 11. Common Mistakes

- Committing large binary files (model checkpoints, datasets) directly into Git — bloats repo size permanently (even after deletion, history retains them) — use Git LFS or DVC instead.
- `git push --force` onto a shared branch — silently destroys teammates' work; use `--force-with-lease` at minimum, and only on your own feature branches.
- Committing secrets (API keys, `.env` files) — once pushed, consider them compromised even if later removed (history persists; requires history rewriting + credential rotation).
- Writing vague commit messages ("fix stuff") — makes `git bisect` and code archaeology painful later.

---

## 12. Best Practices (2026)

- Conventional Commits format (`feat:`, `fix:`, `refactor:`) — now near-universal, enables automated changelog generation and semantic versioning, and plays well with AI PR-summarization tools.
- Small, atomic commits with clear messages — each commit should represent one logical change and ideally still pass tests on its own.
- Protected `main` branch + required PR review + CI checks before merge — standard on every serious repo.
- Use `git switch`/`git restore` (modern, clearer commands introduced in Git 2.23) over the overloaded legacy `git checkout` in new scripts/docs.
- Sign commits (`git commit -S`) increasingly expected for supply-chain security in 2026, especially for published model/dataset repositories.

---

## 13. Exercises

**Easy:** Create a repo, make 5 commits, and use `git log --oneline --graph` to visualize it.
**Medium:** Create two branches that both modify the same file's same line; merge them and resolve the conflict by hand.
**Hard:** Use `git rebase -i` to squash 4 messy commits into 1 clean commit with a proper message.
**Mathematical:** Given a repo with $n$ commits, what is the worst-case number of comparisons for `git bisect` to find the commit that introduced a bug? (binary search → $O(\log n)$, tying directly back to Lesson 4.)
**Coding:** Extend the toy content-addressable store in Section 6 to support a simple two-file "tree" object and a `checkout` function that reconstructs files from a commit hash.

---

## 14. Mini Project

Set up a real **ML project repository** from scratch: initialize Git, create a proper `.gitignore`, structure `src/`, `tests/`, `data/` (data itself untracked, tracked via DVC or LFS pointer), write a `README.md`, make a feature branch for "add baseline model," open a (self-reviewed) pull request on GitHub, and configure a minimal GitHub Actions workflow that runs `pytest` on every push.

---

## 15. Interview Preparation

- Explain the difference between `git merge` and `git rebase`, and when you'd choose each.
- What is a detached HEAD state, and how do you get out of it safely?
- How would you recover a commit that you accidentally `reset --hard` away? (`git reflog`)
- System design: how would you structure Git workflow + branch protection rules for a team of 10 engineers shipping a production ML API weekly?

---

## 16. Summary

Git's entire model reduces to one idea: **an immutable, content-addressed, hash-linked DAG of snapshots**, with branches as cheap mutable pointers into that DAG. Everything else — merging, rebasing, conflict resolution — is graph algorithms operating on that structure. Understanding this model (not just memorizing commands) is what lets you recover from any Git mistake instead of panicking, and is a direct prerequisite for reproducible ML experimentation.

---

## 17. References

- Pro Git (free, official book): https://git-scm.com/book
- GitHub Docs — Actions, branch protection, PR workflows
- Scott Chacon — *Pro Git*, chapter on Git internals (plumbing vs porcelain)
- Git LFS and DVC official documentation
