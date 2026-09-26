# Phase 8 · Lesson 3 — Kubernetes

> Prerequisite: Docker (Lesson 2)

---

## 1. Introduction

### What is Kubernetes?
An open-source container orchestration platform (originally developed at Google, based on their internal "Borg" system, released 2014) for automating the deployment, scaling, networking, and management of containerized applications (Lesson 2) across a cluster of machines — the standard way production ML/LLM services (and virtually all containerized applications) are actually run at real scale in 2026.

### Why does it exist?
Running a single container manually (Lesson 2) doesn't address real production needs: what happens when a container crashes (it should be automatically restarted), when traffic increases (more copies should automatically start), when you want zero-downtime deployments (rolling updates without dropping requests), or when you need to run across many physical machines rather than one? Kubernetes exists specifically to automate all of this, treating a cluster of machines as one unified, programmable deployment target.

### Historical background
Kubernetes (2014) built on Google's decade-plus of internal experience running containerized workloads at massive scale (Borg), open-sourced and donated to the Cloud Native Computing Foundation, rapidly becoming the de facto standard container orchestrator — displacing or absorbing most competing systems (Docker Swarm, Mesos) by the early 2020s, and remaining dominant through 2026.

### Real-world motivation
Once you've containerized your FastAPI/ML application (Lessons 1-2), Kubernetes is very likely how it actually gets deployed and kept running reliably in any organization beyond a small startup — auto-restarting failed instances, auto-scaling under load, and enabling zero-downtime deployments.

---

## 2. Theory

### Core Kubernetes objects
- **Pod**: the smallest deployable unit — one or more tightly-coupled containers sharing network/storage, typically one container per pod for simple services (your FastAPI app).
- **Deployment**: manages a set of identical Pod replicas, handling rolling updates and self-healing (automatically replacing failed Pods) — the standard way to run a stateless service like an API.
- **Service**: provides a stable network endpoint (a fixed IP/DNS name) in front of a changing set of Pods (which come and go as they're replaced/scaled) — solving the problem that individual Pods are ephemeral and shouldn't be addressed directly.
- **Ingress**: manages external HTTP(S) access into the cluster, routing based on hostname/path to the appropriate Service — the typical entry point for user-facing traffic.
- **ConfigMap/Secret**: externalize configuration and sensitive credentials from container images, allowing the same image to be configured differently across environments (directly extending Phase 1 Lesson 8's separation-of-concerns principle).

### Self-healing
Kubernetes continuously monitors actual state against **desired state** (declared in configuration, e.g., "I want 3 replicas of this Pod running") — if a Pod crashes or a node fails, Kubernetes automatically creates replacement Pods to restore the desired state, without human intervention — a direct, practical application of control-theory-style feedback loops (observe actual state, compare to desired state, take corrective action, repeat).

### Horizontal Pod Autoscaling (HPA)
Automatically adjusts the number of running Pod replicas based on observed metrics (commonly CPU/memory utilization, or custom metrics like request queue depth) — scaling up under load and back down when load subsides, directly addressing the variable-traffic concern raised in Lesson 2's container-startup-time discussion.

### Rolling updates
When deploying a new application version, Kubernetes can gradually replace old Pods with new ones (rather than stopping everything and starting the new version, which would cause downtime) — starting a few new-version Pods, verifying they're healthy (via readiness probes), then terminating an equivalent number of old-version Pods, repeating until the rollout completes — enabling zero-downtime deployments.

---

## 3. Mathematical Foundations

### Autoscaling as a control system
HPA computes the desired replica count via:

$$
\text{desired\_replicas} = \text{current\_replicas} \times \frac{\text{current\_metric\_value}}{\text{target\_metric\_value}}
$$

e.g., if current CPU utilization is 80% and the target is 50%, with 4 current replicas: $4 \times (80/50) = 6.4 \to 7$ replicas — a direct, proportional-control feedback formula (a simplified relative of the PID controllers used in classical control theory), continuously re-evaluated and adjusted as actual metrics change.

### Scheduling as a bin-packing/constraint-satisfaction problem
Kubernetes' scheduler must decide which node each new Pod runs on, given each Pod's resource requests (CPU/memory) and each node's available capacity — formally a variant of the **bin-packing problem** (NP-hard in general, Phase 1 Lesson 4's complexity theory territory), solved via practical heuristics (filtering nodes that can't fit the Pod, then scoring/ranking remaining candidates by various criteria) rather than an exact optimal solution, exactly the same greedy-heuristic tradeoff seen throughout this curriculum (Phase 1 Lesson 4's algorithms, Phase 6 Lesson 1's BPE) when exact optimization is computationally infeasible at scale.

### Rolling update safety margins
A rolling update's `maxUnavailable` and `maxSurge` parameters (what fraction of Pods can be simultaneously unavailable/added beyond the desired count during the rollout) directly trade off rollout speed against risk — an aggressive `maxUnavailable` speeds up rollouts but risks serving reduced capacity (or triggering autoscaling thrashing) during the transition; a conservative setting is slower but safer, a genuine, quantifiable engineering tradeoff configured per-application based on its criticality and traffic patterns.

---

## 4. Algorithm — Kubernetes' Reconciliation Loop (fully specified, the core operating principle)

```
CONTINUOUSLY (this is a persistent, ongoing loop, not a one-time procedure):
1. OBSERVE the actual current state of the cluster (which Pods are running, their health, resource usage)
2. COMPARE against the DESIRED state (declared in your Deployment/Service/etc. configuration, e.g., YAML)
3. IF actual state != desired state:
     COMPUTE the necessary actions to reconcile them (start new Pods, terminate excess ones,
     reroute traffic, etc.)
     EXECUTE those actions
4. REPEAT (this loop runs continuously, for EVERY managed resource type, indefinitely)

THIS SAME PATTERN APPLIES TO:
  - Deployments maintaining N replicas (self-healing, Section 2)
  - HPA maintaining a target metric value (autoscaling, Section 3)
  - Rolling updates gradually shifting from old to new Pod versions
  ALL of Kubernetes' behavior is fundamentally this one reconciliation pattern, applied to different
  resource types and desired-state specifications.
```

---

## 5. Python Implementation (Kubernetes manifests + supporting scripts)

```yaml
# deployment.yaml — deploying the Phase 8 Lesson 2 containerized FastAPI application
apiVersion: apps/v1
kind: Deployment
metadata:
  name: actuarial-api
spec:
  replicas: 3                        # DESIRED STATE: always keep 3 healthy replicas running
  selector:
    matchLabels:
      app: actuarial-api
  strategy:
    rollingUpdate:
      maxUnavailable: 1               # Section 3's rolling-update safety margin
      maxSurge: 1
  template:
    metadata:
      labels:
        app: actuarial-api
    spec:
      containers:
        - name: actuarial-api
          image: actuarial-api:1.0.0
          ports:
            - containerPort: 8000
          resources:                  # Section 3's bin-packing inputs -- what the scheduler uses
            requests:
              cpu: "250m"              # 0.25 CPU cores guaranteed
              memory: "256Mi"
            limits:
              cpu: "500m"               # hard ceiling -- Lesson 2's cgroups enforcement, now at cluster scale
              memory: "512Mi"
          livenessProbe:               # "is this container alive?" -- restart if this fails repeatedly
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          readinessProbe:              # "is this container ready for traffic?" -- distinct from liveness
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: actuarial-api-service
spec:
  selector:
    app: actuarial-api                # routes traffic to ANY healthy Pod matching this label
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP                     # internal-only; an Ingress or LoadBalancer type exposes it externally
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: actuarial-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: actuarial-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 50      # Section 3's target_metric_value
```

```bash
kubectl apply -f deployment.yaml            # DECLARE the desired state -- Kubernetes reconciles automatically
kubectl get pods -l app=actuarial-api        # observe ACTUAL current state
kubectl rollout status deployment/actuarial-api    # monitor a rolling update in progress
kubectl scale deployment/actuarial-api --replicas=5 # manually override desired replica count
kubectl logs -f deployment/actuarial-api             # follow logs across the deployment's Pods
```

---

## 6. Build From Scratch

**A minimal reconciliation-loop simulator (to make Section 4's core algorithm concrete):**
```python
import time
import random

class SimplePod:
    def __init__(self, pod_id: int):
        self.pod_id = pod_id
        self.healthy = True

    def simulate_random_failure(self, failure_prob: float = 0.1):
        if random.random() < failure_prob:
            self.healthy = False

class SimpleReconciler:
    """Illustrates Kubernetes' core reconciliation loop (Section 4) at a toy scale."""
    def __init__(self, desired_replicas: int):
        self.desired_replicas = desired_replicas
        self.pods: list[SimplePod] = [SimplePod(i) for i in range(desired_replicas)]
        self.next_pod_id = desired_replicas

    def observe_and_reconcile(self) -> None:
        for pod in self.pods:
            pod.simulate_random_failure()

        healthy_pods = [p for p in self.pods if p.healthy]
        unhealthy_count = len(self.pods) - len(healthy_pods)
        if unhealthy_count > 0:
            print(f"  Detected {unhealthy_count} unhealthy pod(s), removing and replacing...")

        self.pods = healthy_pods
        while len(self.pods) < self.desired_replicas:
            new_pod = SimplePod(self.next_pod_id)
            self.next_pod_id += 1
            self.pods.append(new_pod)
            print(f"  Started replacement pod {new_pod.pod_id}")

reconciler = SimpleReconciler(desired_replicas=3)
for tick in range(5):
    print(f"Reconciliation tick {tick}:")
    reconciler.observe_and_reconcile()
    print(f"  Current healthy pod count: {len(reconciler.pods)} (desired: {reconciler.desired_replicas})")
    time.sleep(0.1)
```
This directly demonstrates Section 4's reconciliation pattern: at every "tick," the system observes actual state (some pods may have randomly failed), compares against desired state (always 3 replicas), and takes corrective action (removing unhealthy pods, starting replacements) — precisely the mechanism underlying Kubernetes' self-healing behavior, at a vastly simplified scale.

---

## 7. Library/Tool Comparison

| From scratch/manual | Kubernetes |
|---|---|
| `SimpleReconciler` (toy) | Kubernetes' actual controller-manager — production-grade, handles thousands of resources across real multi-node clusters, integrates with real container runtimes |
| Manual container restart scripts | Kubernetes' Deployment controller — automatic, continuous, no manual intervention needed |
| Manual load balancer configuration | Kubernetes Services — automatic endpoint management as Pods come and go |
| Manual scaling scripts (e.g., cron-triggered) | Horizontal Pod Autoscaler — continuous, metric-driven, real-time scaling |

---

## 8. Visual Explanations

**Kubernetes' layered architecture (Pod -> Deployment -> Service):**
```
                    Service (stable endpoint, e.g., actuarial-api-service)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          Pod (replica 1) Pod (replica 2) Pod (replica 3)   <- managed by a Deployment
          [container]      [container]      [container]        (self-healing, rolling updates)
   (Service routes traffic to WHICHEVER pods are currently healthy -- pods are ephemeral, Service is stable)
```

**Rolling update (gradual replacement, zero downtime):**
```
Start:     [v1][v1][v1]                    (3 old-version pods, 100% traffic on v1)
Step 1:    [v1][v1][v1][v2]                (maxSurge: 1 extra pod started)
Step 2:    [v1][v1][v2]                    (one v1 terminated once v2 is READY -- maxUnavailable respected)
Step 3:    [v1][v2][v2]
Step 4:    [v2][v2][v2]                    (rollout complete -- traffic NEVER dropped to zero capacity)
```

---

## 9. Practical Examples

**Simple:** deploy the Section 5 manifests to a local Kubernetes cluster (Minikube or Kind) and verify 3 replicas start and respond to requests via the Service.
**Medium:** deliberately kill one Pod (`kubectl delete pod <name>`) and observe Kubernetes automatically start a replacement, restoring the desired replica count.
**Real-world:** deploy your Phase 8 Lesson 2 containerized actuarial API to a local cluster with the full manifest set (Deployment, Service, HPA), simulate load (using a simple load-testing script) and observe the HPA automatically scale up replica count in response.

---

## 10. Real Industry Use Cases

- **Virtually every company running containerized production workloads at meaningful scale**: Kubernetes (or a managed variant — EKS, GKE, AKS, Lesson 5's cloud territory) is the standard orchestration platform.
- **LLM inference serving at scale**: production LLM APIs (whether self-hosted open-weight models or internal services wrapping provider APIs) commonly run on Kubernetes, using HPA to handle variable request volume and GPU-aware scheduling for models requiring specialized hardware.
- **MLOps platforms** (Lesson 7): many MLOps tools (Kubeflow, and others) are built directly on top of Kubernetes, using it as the underlying orchestration layer for training jobs, model serving, and pipeline execution.
- **Zero-downtime deployment practices**: rolling updates are the standard mechanism enabling frequent, safe production deployments at companies shipping code multiple times per day.

---

## 11. Common Mistakes

- Not setting resource requests/limits (Section 5) — leads to poor scheduling decisions (Section 3's bin-packing) and potential resource contention/starvation across Pods on the same node.
- Confusing liveness and readiness probes — a liveness-probe failure triggers a container *restart*; a readiness-probe failure just temporarily removes the Pod from receiving traffic (without restarting it) — using the wrong one for the wrong situation causes either unnecessary restarts or continued traffic to a genuinely broken Pod.
- Setting `maxUnavailable`/`maxSurge` too aggressively for a critical, high-traffic service — risking capacity shortfalls during rollouts.
- Manually managing individual Pods directly rather than through a Deployment — loses self-healing and rolling-update capabilities entirely; Pods should almost always be managed indirectly via a higher-level controller.

---

## 12. Best Practices (2026)

- Always specify resource requests and limits for every container, informed by actual observed usage (Lesson 8's monitoring) rather than guesswork.
- Implement distinct, meaningful liveness and readiness probes (not just reusing the same `/health` endpoint identically for both, if your application's "alive" and "ready for traffic" conditions genuinely differ).
- Use HPA for any service with variable traffic patterns, tuned against a metric (CPU, or better, a custom application-specific metric like request queue depth) that genuinely correlates with the service's actual capacity needs.
- Use managed Kubernetes services (EKS/GKE/AKS, Lesson 5) rather than self-managing cluster infrastructure, unless you have a specific, well-justified reason to run your own control plane.

---

## 13. Exercises

**Easy:** Deploy the Section 5 manifests to a local cluster (Minikube/Kind) and verify all 3 Pods reach a "Running" and "Ready" state.
**Medium:** Deliberately break the `/health` endpoint in your application and observe how Kubernetes' liveness probe responds (repeated restarts) versus what happens if only the readiness probe fails.
**Hard:** Perform a rolling update (change the image tag and reapply the Deployment) and monitor the rollout in real time (`kubectl rollout status`), verifying zero requests are dropped throughout (using a continuous load-testing script hitting the Service during the rollout).
**Mathematical:** Using Section 3's HPA formula, compute the resulting replica count for a service currently at 6 replicas experiencing 90% CPU utilization with a target of 60%, and discuss what would happen on the NEXT reconciliation tick if that scale-up successfully reduces utilization to exactly the target.
**Coding:** Implement and extend the Section 6 `SimpleReconciler` to also simulate a basic HPA-style scaling rule (adjusting `desired_replicas` based on a simulated load metric) and observe the reconciler respond to changing simulated load over several ticks.

---

## 14. Mini Project

Deploy your **complete containerized actuarial API (from Lesson 2) to a local Kubernetes cluster** with a full production-shaped manifest set: a Deployment with appropriate resource requests/limits and distinct liveness/readiness probes, a Service exposing it, an HPA configured against CPU utilization, and a `ConfigMap` externalizing any environment-specific configuration; perform a simulated rolling update between two versions while running a continuous load-testing script against the service, verifying zero dropped requests; and write a short runbook documenting how to deploy, scale, update, and roll back this application.

---

## 15. Interview Preparation

- Explain the relationship between Pods, Deployments, and Services in Kubernetes.
- What is Kubernetes' reconciliation loop, and how does it enable self-healing?
- What's the difference between a liveness probe and a readiness probe?
- How does Horizontal Pod Autoscaling work, and what are the tradeoffs in choosing which metric to scale on?

---

## 16. Summary

Kubernetes automates the deployment, scaling, and self-healing of containerized applications (Lesson 2) across a cluster, operating on one core principle applied uniformly across every resource type: continuously observe actual state, compare against declared desired state, and take corrective action to reconcile any difference — the mechanism underlying self-healing (replacing failed Pods), autoscaling (HPA's proportional-control formula), and zero-downtime rolling updates. Understanding this reconciliation pattern — rather than memorizing each feature independently — is what makes Kubernetes' behavior predictable and debuggable, directly setting up Lesson 4's CI/CD pipelines (which typically build, test, and deploy exactly these kinds of manifests) and Lesson 5's cloud infrastructure (where managed Kubernetes services are the standard production deployment target).

---

## 17. References

- Official Kubernetes documentation (kubernetes.io/docs)
- Burns, Beda, Hightower — *Kubernetes: Up and Running*
- Google's original Borg paper — "Large-scale cluster management at Google with Borg" (2015, the direct intellectual predecessor to Kubernetes)
- CNCF (Cloud Native Computing Foundation) — Kubernetes project resources and certification materials
