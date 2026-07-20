# Phase 1 · Lesson 6 — Linux & Bash

> Prerequisite: Lessons 1–5

---

## 1. Introduction

### What is this topic?
Linux is the dominant operating system kernel for servers, cloud VMs, containers, and virtually all GPU training clusters. Bash (Bourne Again SHell) is the standard command-line shell/scripting language used to interact with it. Nearly every AI training run, model deployment, or data pipeline eventually executes on a Linux box — often one you access only via SSH, with no GUI.

### Why does it exist?
Unix (Linux's philosophical ancestor, 1969) was designed around small, composable tools that "do one thing well" and communicate via plain-text streams — a philosophy directly mirrored in modern ML tooling (piping data between CLI tools, chaining shell commands in training scripts, containerized microservices communicating over simple protocols).

### Historical background
Linux (Torvalds, 1991) combined with GNU tools created a free, Unix-like OS that now runs the overwhelming majority of cloud infrastructure, including every major AI lab's training clusters (Kubernetes nodes, Slurm HPC clusters) and virtually all Docker containers (Phase 8).

### Real-world motivation
You will SSH into remote GPU boxes, debug why a training job died overnight, inspect logs at 2am, manage file permissions, monitor GPU memory (`nvidia-smi`), and write shell scripts to orchestrate multi-step pipelines. None of this is optional for an AI engineer.

---

## 2. Theory

### Filesystem hierarchy
- `/` root; `/home` user directories; `/etc` config files; `/var/log` logs; `/tmp` scratch space; `/usr/bin`, `/usr/local/bin` executables.
- **Everything is a file** in Unix philosophy — devices, processes (`/proc`), even kernel parameters are exposed as files.

### Processes
- Every running program is a process with a PID; processes have a parent-child tree (`init`/`systemd` at the root, PID 1).
- **Signals**: how processes are told to stop/reload — `SIGINT` (Ctrl+C, "please stop"), `SIGKILL` (force kill, cannot be caught/ignored), `SIGTERM` (graceful stop request).

### Permissions model
Every file has an owner, a group, and permission bits (`rwx` for owner/group/other), represented as `-rwxr-xr--` or octal `754`. This matters enormously when deploying models: a service running as a restricted user should not be able to read other users' secrets.

### Standard streams and piping
Every process has `stdin` (0), `stdout` (1), `stderr` (2). The pipe (`|`) redirects one process's `stdout` into another's `stdin`, enabling composable data pipelines — conceptually identical to chaining generator functions in Python (Lesson 2).

---

## 3. Mathematical Foundations

Linux/Bash is less mathematically dense than earlier lessons, but two areas matter directly:

### Process scheduling as an optimization problem
The Linux Completely Fair Scheduler (CFS) approximates giving each process an equal share of CPU time over a scheduling period, tracked via a "virtual runtime" $vruntime$; the scheduler always picks the runnable process with the lowest $vruntime$, implemented internally with a red-black tree ($O(\log n)$ insert/lookup — directly reusing balanced-tree ideas from Lesson 4).

### Regular expressions (used constantly in `grep`/`sed`/`awk`)
A regex defines a formal language recognized by a finite automaton. Practically: `.` matches any char, `*` = zero-or-more (Kleene star), `+` = one-or-more, `^`/`$` anchor start/end, `[...]` character class. Complexity: naive backtracking regex engines can be exponential $O(2^n)$ on pathological patterns (catastrophic backtracking) — a real production incident cause (regex-based log filters hanging a pipeline).

---

## 4. Algorithm — How a Shell Pipeline Executes

```
COMMAND: cat access.log | grep "ERROR" | wc -l

1. Shell forks 3 child processes: cat, grep, wc
2. Connects cat's stdout -> grep's stdin via an OS pipe (kernel buffer)
3. Connects grep's stdout -> wc's stdin via another pipe
4. All 3 processes run CONCURRENTLY (not sequentially!) —
   cat can be writing line 500 while grep is still filtering line 100
5. Shell waits for all processes to exit, reports wc's output to the terminal
```
This concurrent, buffered streaming model is why Unix pipelines handle huge files without loading them fully into memory — directly analogous to Python generators (Lesson 2) chained together.

---

## 5. Practical Implementation

```bash
#!/usr/bin/env bash
set -euo pipefail   # STANDARD SAFETY HEADER for every production script:
                     #  -e: exit on any command failure
                     #  -u: error on undefined variables
                     #  -o pipefail: a pipeline fails if ANY stage fails, not just the last

MODEL_DIR="/data/models/mortality_xgb"
LOG_FILE="/var/log/train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training run..." | tee -a "$LOG_FILE"

# Check GPU availability before starting an expensive job
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: no GPU detected, falling back to CPU" >> "$LOG_FILE"
fi

# Run training, capturing both stdout and stderr to the log file
python train.py \
    --data "$MODEL_DIR/train.csv" \
    --epochs 50 \
    --lr 0.001 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}   # exit code of `python`, not `tee`
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "Training FAILED with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    exit "$EXIT_CODE"
fi

echo "Training completed successfully." | tee -a "$LOG_FILE"
```

**Essential daily-driver commands:**
```bash
grep -rn "TODO" src/                 # recursive search with line numbers
find . -name "*.pt" -mtime +7        # find checkpoint files older than 7 days
du -sh */ | sort -rh                 # disk usage per folder, sorted largest-first
top / htop                           # live process/CPU/memory monitor
nvidia-smi -l 1                      # live GPU utilization, refreshed every second
df -h                                # disk space per mount
chmod 600 secrets.env                # owner read/write only — no group/other access
ssh -i key.pem user@remote-gpu-box   # connect to a remote training server
scp model.pt user@remote:/data/      # copy a file to a remote server
rsync -avz --progress ./data/ remote:/data/   # efficient incremental sync (deltas only)
tail -f train.log                    # follow a growing log file live
awk -F',' '{sum += $3} END {print sum}' claims.csv   # sum column 3 of a CSV
```

---

## 6. Build From Scratch

**A minimal shell pipeline emulator in Python (to demystify what Bash does with `|`):**
```python
import subprocess

def run_pipeline(commands: list[list[str]]) -> str:
    """Emulates `cmd1 | cmd2 | cmd3` using subprocess pipes."""
    processes = []
    prev_stdout = None
    for cmd in commands:
        p = subprocess.Popen(
            cmd, stdin=prev_stdout, stdout=subprocess.PIPE
        )
        if prev_stdout is not None:
            prev_stdout.close()   # allow upstream process to receive SIGPIPE if downstream exits
        prev_stdout = p.stdout
        processes.append(p)
    output, _ = processes[-1].communicate()
    return output.decode()

# Emulates: cat access.log | grep ERROR | wc -l
result = run_pipeline([
    ["cat", "access.log"],
    ["grep", "ERROR"],
    ["wc", "-l"],
])
```
This mirrors, at a simplified level, exactly what Bash's parser does when it sees `|`: fork processes, wire stdout→stdin via OS pipes, run concurrently.

---

## 7. Library/Tool Comparison

| Concept | Manual/scratch | Production tool |
|---|---|---|
| Log filtering | hand-written Python loop | `grep`/`ripgrep` (Rust, far faster on large files) |
| Remote file sync | custom SCP loop | `rsync` (delta-transfer algorithm, resumable) |
| Process supervision | custom retry loop | `systemd`, `supervisord`, or container orchestrators (Phase 8) |
| Log aggregation | `tail -f` on each box manually | centralized logging (ELK/Loki stack, Phase 8 Monitoring) |

---

## 8. Visual Explanations

```
Pipe as a kernel buffer between processes:

 cat access.log  ──stdout──▶ [kernel pipe buffer] ──stdin──▶ grep "ERROR"
                                                     ──stdout──▶ [kernel pipe buffer] ──stdin──▶ wc -l

Both cat and grep run concurrently; if the buffer fills, the writer blocks until the reader consumes (backpressure) —
the same backpressure concept reappears in async streaming APIs (Phase 8, LLM token streaming).
```

**Permission bits:**
```
-rwxr-xr--
│└┬┘└┬┘└┬┘
│ │  │  └─ other: r-- (read only)
│ │  └──── group: r-x (read + execute)
│ └─────── owner: rwx (read + write + execute)
└───────── file type (- = regular file, d = directory, l = symlink)
```

---

## 9. Practical Examples

**Simple:** count how many `.py` files exist recursively: `find . -name "*.py" | wc -l`.
**Medium:** find the 5 largest files under a directory: `find . -type f -exec du -h {} \; | sort -rh | head -5`.
**Real-world:** a cron job that runs a data-cleaning script nightly and emails you only if it fails:
```bash
0 2 * * * /usr/bin/python3 /opt/pipelines/clean_claims.py || echo "FAILED" | mail -s "Pipeline Alert" you@example.com
```

---

## 10. Real Industry Use Cases

- **Every AI training cluster** (SLURM, Kubernetes on bare-metal GPU nodes): jobs are submitted, monitored, and debugged through Linux shells and logs, regardless of how high-level the ML framework is.
- **Netflix, Google, Meta internal tooling**: enormous internal fleets of Linux servers managed via shell scripting + configuration management (Ansible, etc.), the substrate underneath every "one-click deploy" button.
- **Docker/Kubernetes (Phase 8)**: containers *are* Linux processes with namespace/cgroup isolation — you cannot deeply understand containers without understanding the Linux primitives (cgroups, namespaces) they wrap.
- **NVIDIA GPU clusters**: `nvidia-smi`, CUDA driver management, and GPU process isolation are all Linux-level concepts every ML engineer eventually debugs directly.

---

## 11. Common Mistakes

- Not using `set -euo pipefail` in production scripts — a failed step silently continues, corrupting downstream results.
- Parsing `ls` output in scripts (fragile, breaks on filenames with spaces/newlines) — use `find` with `-print0`/`xargs -0` instead.
- Running long training jobs directly in a foreground SSH session — disconnect and the job dies; use `tmux`/`screen`/`nohup` or a proper job scheduler.
- Editing files as `root` unnecessarily — increases blast radius of any mistake; use `sudo` only for the specific command that needs it.

---

## 12. Best Practices (2026)

- Use `tmux` (or a managed job orchestrator like SLURM/Kubernetes Jobs) rather than raw `nohup &` for anything long-running — session persistence + easy re-attachment.
- Prefer modern CLI tool replacements when available in your environment: `ripgrep` (`rg`) over `grep` for large codebases, `fd` over `find`, `bat` over `cat` — dramatically faster on large repos, common in 2026 dev environments.
- Write idempotent scripts (safe to re-run) — critical for training/data pipelines that might be retried after a crash.
- Store secrets in environment variables or a secrets manager, never hard-coded in scripts committed to Git (ties back to Lesson 5).

---

## 13. Exercises

**Easy:** Write a one-liner to count unique IP addresses in a log file.
**Medium:** Write a Bash script that monitors free disk space every 60 seconds and sends an alert if it drops below 10%.
**Hard:** Write a script that watches a directory for new `.csv` files (using `inotifywait` or polling) and automatically triggers a Python cleaning script on each new file.
**Mathematical:** Given CFS's red-black-tree-based scheduler picks the lowest-`vruntime` process each tick in $O(\log n)$, argue why this scales better than a naive $O(n)$ linear scan scheduler on a machine with hundreds of processes.
**Coding:** Extend the Section 6 Python pipeline emulator to support redirecting the final output to a file (like `>` in Bash).

---

## 14. Mini Project

Build a **remote training job launcher**: a Bash script that (1) rsyncs your local code to a remote GPU box, (2) launches training inside a `tmux` session so it survives disconnection, (3) tails the log back to your terminal, (4) on completion, rsyncs the resulting model checkpoint back to your local machine, and (5) sends a desktop/email notification on success or failure. This mirrors real workflows used before (and often alongside) more sophisticated orchestration platforms.

---

## 15. Interview Preparation

- Explain the difference between a process and a thread at the OS level.
- What does `chmod 755` mean numerically and in `rwx` terms?
- How does a Unix pipe achieve concurrency between two commands?
- System design: you SSH into a box and a training job is unresponsive — walk through your debugging steps (`top`/`htop`, `nvidia-smi`, checking logs, checking `dmesg` for OOM kills).

---

## 16. Summary

Linux and Bash are the operating substrate for essentially all real AI infrastructure: processes, signals, permissions, and pipes are the primitives that Docker, Kubernetes, and every training cluster are built from. Learning to comfortably navigate a remote Linux box via SSH, write safe idempotent shell scripts, and reason about processes/permissions is not "extra" — it's the difference between being able to debug a broken 3am training run yourself versus being stuck waiting for someone else to.

---

## 17. References

- *The Linux Command Line* — William Shotts (free online book)
- *Unix and Linux System Administration Handbook*
- `man bash`, `man find`, `man grep` — the original, still-authoritative documentation
- ShellCheck (https://www.shellcheck.net/) — essential static analysis tool for Bash scripts
