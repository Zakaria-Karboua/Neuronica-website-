# Phase 8 · Lesson 2 — Docker

> Prerequisite: Phase 1 Lesson 6 (Linux & Bash), Lesson 1 (FastAPI, this phase)

---

## 1. Introduction

### What is Docker?
A platform for building, packaging, and running applications inside **containers** — lightweight, isolated units bundling an application with all its dependencies (libraries, runtime, system tools) so it runs identically regardless of the underlying host machine's configuration. Docker (2013) popularized containerization, building on existing Linux kernel primitives (namespaces and cgroups, Phase 1 Lesson 6) into an accessible, standardized developer tool.

### Why does it exist?
"It works on my machine" is a genuine, historically enormous source of deployment friction — an application that works in development can fail in production due to differing Python versions, missing system libraries, or conflicting dependency versions. Docker eliminates this class of problem by packaging the *entire* runtime environment (not just the application code) into a single, portable, reproducible unit.

### Historical background
Containerization concepts predate Docker (chroot jails, Solaris Zones, LXC), but Docker's 2013 release made these Linux kernel primitives dramatically more accessible via a simple CLI and a standardized image format — triggering the now-ubiquitous container ecosystem, directly enabling Kubernetes (Lesson 3) and the entire modern cloud-native deployment paradigm.

### Real-world motivation
Every model/API you build across this curriculum eventually needs to run reliably on infrastructure you don't fully control (a cloud VM, a Kubernetes cluster) — Docker is the standard mechanism for packaging your FastAPI application (Lesson 1) plus its exact Python version, dependencies, and system libraries into one deployable unit.

---

## 2. Theory

### Images vs. containers
A Docker **image** is a read-only template (a snapshot of a filesystem plus metadata) built from a `Dockerfile`; a **container** is a running (or stopped) instance of an image — directly analogous to Phase 1 Lesson 3's class-vs-instance distinction (an image is the "class," a container is an "object" instantiated from it).

### Layers and the build cache
A Docker image is built as a stack of **layers**, each corresponding to one instruction in the `Dockerfile` — Docker caches each layer, and if a `Dockerfile` instruction and its inputs haven't changed since the last build, Docker reuses the cached layer rather than rebuilding it, dramatically speeding up iterative builds. This directly motivates the practice of ordering `Dockerfile` instructions from least-frequently-changing (installing system dependencies) to most-frequently-changing (copying your actual application code) — putting expensive, rarely-changing steps early so their cache remains valid across most rebuilds.

### Namespaces and cgroups (the Linux kernel mechanisms underneath, Phase 1 Lesson 6 revisited)
**Namespaces** provide isolation (a container's processes see their own isolated view of the filesystem, network, process IDs) — the container *believes* it has the whole machine to itself. **cgroups** (control groups) limit and account for resource usage (CPU, memory) per container — together, these two kernel features are what actually implement container isolation; Docker is, at its core, a convenient interface over these primitives, not a fundamentally new virtualization technology.

### Containers vs. virtual machines
A VM virtualizes an entire machine (including its own kernel), while a container shares the host machine's kernel, isolating only at the process/namespace level — containers are consequently much lighter-weight (faster startup, smaller footprint) than VMs, at the cost of weaker isolation guarantees (all containers on a host share one kernel, a genuine security consideration, Lesson 9).

---

## 3. Mathematical Foundations

Docker is primarily a systems/infrastructure topic, but a few quantitative considerations matter directly:

### Image size and layer caching efficiency
Total image size is the sum of all layers' sizes (with some deduplication for shared base-layer content across images) — minimizing image size (using slim/minimal base images, multi-stage builds, Section 4) directly reduces deployment time (image pull/transfer time) and storage costs at scale, a genuinely quantifiable production concern when deploying across many servers/containers.

### Resource allocation and the multi-tenancy problem
When running $n$ containers on a host with total resources $R$, cgroups let you allocate/limit each container's share (e.g., container $i$ gets at most $r_i$ of $R$) — a direct resource-allocation problem: setting limits too low causes application slowdowns/crashes under load; too high (or unset) risks one container starving others (or the host itself) of resources — directly connecting to Lesson 3's Kubernetes resource requests/limits, which formalize this allocation problem further at cluster scale.

### Startup time and cold-start latency
Container startup time (typically seconds, versus a VM's tens of seconds to minutes) directly affects auto-scaling responsiveness (Lesson 3) — how quickly new capacity can come online in response to increased load — a genuinely important latency consideration for any application with variable traffic, including most LLM-serving APIs (Lesson 1) experiencing bursty usage patterns.

---

## 4. Algorithm — The Docker Build and Run Lifecycle (fully specified)

```
BUILD TIME:
1. docker build reads the Dockerfile, executing each instruction IN ORDER
2. FOR each instruction:
     CHECK the build cache: has this exact instruction + its inputs been built before?
     IF cached: REUSE the existing layer (fast)
     IF not cached: EXECUTE the instruction, creating a NEW layer
     (once one layer is rebuilt, ALL subsequent layers must also rebuild -- cache invalidation cascades forward)
3. RESULT: a tagged image, ready to be run or pushed to a registry (Docker Hub, or a cloud provider's registry)

RUN TIME:
1. docker run creates a NEW container from the specified image
2. Linux namespaces isolate the container's view of processes/network/filesystem
3. cgroups enforce any specified resource limits (--memory, --cpus)
4. the container's specified ENTRYPOINT/CMD process starts (e.g., launching Uvicorn for a FastAPI app)
5. the container runs until its main process exits (or is stopped/killed)
```

---

## 5. Python Implementation (Dockerfile + supporting files)

```dockerfile
# Dockerfile — a production-shaped, multi-stage build for the Phase 8 Lesson 1 FastAPI app

# --- STAGE 1: build dependencies (this stage's heavy build tools are DISCARDED in the final image) ---
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- STAGE 2: the actual runtime image (small, no build tools included) ---
FROM python:3.12-slim

WORKDIR /app

# Copy ONLY the installed packages from the builder stage -- not the build tools that installed them
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Layer ordering: dependencies (rarely change) BEFORE application code (changes often) -- Section 2's caching logic
COPY ./app ./app

# Run as a non-root user (Lesson 9's security territory, applied here at the container level)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```text
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.0
```

```bash
# Build and run commands
docker build -t actuarial-api:1.0.0 .
docker run -d -p 8000:8000 --memory=512m --cpus=1 --name actuarial-api actuarial-api:1.0.0
docker logs -f actuarial-api        # follow logs (Phase 1 Lesson 6's tail -f, containerized)
docker exec -it actuarial-api bash   # shell into a running container for debugging
```

**Line-by-line rationale for the multi-stage build:** Stage 1 installs dependencies (which may need compilers/build tools not needed at runtime); Stage 2 copies *only* the resulting installed packages into a fresh, minimal image — the final image never contains the build tools themselves, directly minimizing final image size (Section 3's concern) without sacrificing build capability.

---

## 6. Build From Scratch

**A minimal illustration of layer caching's practical effect (measuring rebuild time with vs. without a code-only change):**
```bash
#!/usr/bin/env bash
# demonstrate_layer_caching.sh

echo "First build (no cache):"
time docker build --no-cache -t demo:1 .

echo "Second build (fully cached, no changes):"
time docker build -t demo:2 .    # should be NEAR-INSTANT -- every layer cache-hits

echo "Modifying only application code (not requirements.txt)..."
echo "# comment" >> app/main.py

echo "Third build (dependency layers cached, only code layer rebuilds):"
time docker build -t demo:3 .    # should be FAST -- only the COPY ./app layer and after rebuild
```
Running this concretely demonstrates Section 2's layer-caching claim: the first build pays the full cost of installing dependencies; the second (no changes) is nearly instant; the third (only application code changed) skips the expensive dependency-installation layers entirely, rebuilding only the cheap "copy application code" layer and anything after it — directly validating why `Dockerfile` instruction ordering (dependencies before code) matters in practice, not just in theory.

---

## 7. Library/Tool Comparison

| Manual/from-scratch approach | Docker |
|---|---|
| Manually installing dependencies on every target machine | `Dockerfile` — a single, versioned, reproducible specification of the entire environment |
| Manual process isolation via `chroot`/raw namespaces | Docker's CLI/daemon — a convenient, standardized interface over the same underlying kernel primitives |
| Ad hoc deployment scripts copying files to servers | Container registries (Docker Hub, cloud-provider registries) + `docker pull`/`docker run` — standardized image distribution |
| Manual multi-stage dependency management | Docker's native multi-stage build support (Section 5) |

---

## 8. Visual Explanations

**Docker image layers (each Dockerfile instruction = one layer, cached independently):**
```
Dockerfile:                          Resulting layers (bottom to top):
FROM python:3.12-slim         ──▶    [Layer 0: base Python image]
RUN pip install ...           ──▶    [Layer 1: installed dependencies]  <- cache-hits if requirements.txt unchanged
COPY ./app ./app              ──▶    [Layer 2: application code]        <- rebuilds whenever code changes
CMD ["uvicorn", ...]          ──▶    [Layer 3: metadata, no filesystem change]
   (changing Layer 2 does NOT invalidate Layer 1's cache -- but changing Layer 1 WOULD invalidate Layer 2)
```

**Container vs. VM isolation (Section 2):**
```
VIRTUAL MACHINES:                        CONTAINERS:
┌─────────┬─────────┬─────────┐         ┌─────────┬─────────┬─────────┐
│  App A  │  App B  │  App C  │         │  App A  │  App B  │  App C  │
├─────────┼─────────┼─────────┤         ├─────────┴─────────┴─────────┤
│ Guest   │ Guest   │ Guest   │         │   Namespaces/cgroups (per-  │
│ OS      │ OS      │ OS      │         │   container isolation)      │
├─────────┴─────────┴─────────┤         ├─────────────────────────────┤
│         Hypervisor           │         │      Shared Host Kernel     │
├───────────────────────────────┤         ├─────────────────────────────┤
│         Host OS/Hardware      │         │      Host OS/Hardware       │
└───────────────────────────────┘         └─────────────────────────────┘
  (heavier -- each VM has its OWN kernel)   (lighter -- kernel is SHARED)
```

---

## 9. Practical Examples

**Simple:** write a `Dockerfile` for a minimal "hello world" Python script and build/run it, verifying it produces the expected output identically regardless of the host machine's own Python installation.
**Medium:** containerize the Phase 8 Lesson 1 FastAPI application (Section 5), run it with a memory limit, and verify the `/health` endpoint responds correctly from outside the container.
**Real-world:** containerize a Phase 6/7 model-serving application requiring a genuinely large dependency set (PyTorch, Transformers), using a multi-stage build to minimize final image size, and measure/report the size difference between a naive single-stage build and your optimized multi-stage version.

---

## 10. Real Industry Use Cases

- **Every modern cloud-native deployment**: containerization is the near-universal standard for packaging and deploying applications, ML models included, across virtually every tech company.
- **Kubernetes** (Lesson 3): operates entirely on containers as its fundamental deployment unit — you cannot use Kubernetes without first understanding Docker's containerization model.
- **CI/CD pipelines** (Lesson 4): commonly build a Docker image as a core pipeline step, running tests inside a container matching the production environment exactly, eliminating environment-mismatch bugs.
- **ML model reproducibility**: packaging a specific model version alongside its exact dependency versions (a particular PyTorch/CUDA/Transformers version combination) in a container is a standard practice for ensuring a model behaves identically across development, staging, and production.

---

## 11. Common Mistakes

- Using a large, general-purpose base image (e.g., a full `ubuntu` image) when a slim/minimal base image would suffice — unnecessarily bloats image size and increases attack surface (Lesson 9's security concern).
- Ordering `Dockerfile` instructions poorly (copying application code before installing dependencies) — invalidates the dependency-installation cache layer on every single code change, needlessly slowing every rebuild.
- Running containers as the root user by default — a real security risk (Lesson 9) if the container is ever compromised; always create and use a non-root user for the actual application process.
- Not setting resource limits (`--memory`, `--cpus`) — a runaway or misbehaving container can consume all of a host's resources, starving other containers/processes.

---

## 12. Best Practices (2026)

- Use multi-stage builds to keep final image size minimal, separating build-time dependencies from runtime dependencies.
- Order `Dockerfile` instructions from least- to most-frequently-changing to maximize build-cache effectiveness.
- Always run application processes as a non-root user inside the container.
- Include a `HEALTHCHECK` instruction (or equivalent orchestration-level health check, Lesson 3) so container orchestrators can detect and respond to an unhealthy container automatically.

---

## 13. Exercises

**Easy:** Write a `Dockerfile` for a simple Python script with one dependency, build it, and run it, confirming correct output.
**Medium:** Implement the layer-caching demonstration (Section 6) and measure/report the build-time difference across the three scenarios described.
**Hard:** Convert a single-stage `Dockerfile` for a dependency-heavy ML application (e.g., including PyTorch) into a multi-stage build, and measure the resulting final image size reduction.
**Mathematical:** Given a host with total memory $R$ and $n$ containers each requesting $r_i$ memory, determine the maximum $n$ that can run simultaneously without exceeding $R$, and discuss how you'd handle a container that occasionally spikes above its requested allocation.
**Coding:** Add a proper `HEALTHCHECK` instruction to the Section 5 Dockerfile and verify (via `docker inspect`) that Docker correctly reports the container's health status as it starts up and, if you deliberately break the `/health` endpoint, as it becomes unhealthy.

---

## 14. Mini Project

**Fully containerize your Phase 8 Lesson 1 actuarial API application**: write a production-shaped, multi-stage `Dockerfile` with proper layer ordering, a non-root user, a `HEALTHCHECK`, and appropriately minimal final image size; build and run it locally with sensible resource limits; write a short `docker-compose.yml` if your application has supporting services (a database, a vector store from Phase 7); and document the build/run/debug commands needed for another engineer to reproduce your setup — directly preparing this containerized application for Lesson 3's Kubernetes deployment.

---

## 15. Interview Preparation

- Explain the difference between a Docker image and a Docker container.
- How does Docker's layer caching work, and how should you order Dockerfile instructions to take advantage of it?
- What's the difference between containers and virtual machines, and what are the tradeoffs?
- Why is it a security best practice to run containers as a non-root user?

---

## 16. Summary

Docker packages an application with its complete runtime environment into portable, reproducible containers — built from layered, cached `Dockerfile` instructions, and isolated via Linux namespaces/cgroups (Phase 1 Lesson 6's kernel primitives, made accessible through a standard CLI). Multi-stage builds keep production images minimal by separating build-time from runtime dependencies, and careful layer ordering keeps iterative builds fast. Every application built across this curriculum — FastAPI services (Lesson 1), ML/LLM pipelines (Phases 4-7) — is packaged this way before being deployed at real scale, making Docker the direct, essential foundation for Lesson 3's Kubernetes orchestration and Lesson 4's CI/CD pipelines.

---

## 17. References

- Official Docker documentation (docs.docker.com)
- Docker's own "Best practices for writing Dockerfiles" guide
- Kane, S. & Matthias, K. — *Docker: Up & Running*
- Linux namespaces and cgroups documentation (`man namespaces`, `man cgroups`)
