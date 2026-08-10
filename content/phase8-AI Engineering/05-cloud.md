# Phase 8 · Lesson 5 — Cloud (AWS, Azure, GCP)

> Prerequisite: Docker, Kubernetes (Lessons 2–3)

---

## 1. Introduction

### What is "the cloud" in this context?
On-demand, pay-as-you-go computing infrastructure (compute, storage, networking, and increasingly, managed AI/ML-specific services) provided by major providers — Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform (GCP) — the substrate on which essentially all of this curriculum's applications ultimately run in production, rather than on physical, self-owned/self-managed hardware.

### Why does it exist?
Building and maintaining physical data centers requires enormous upfront capital investment and ongoing operational expertise (power, cooling, hardware replacement, networking) that isn't a core competency for most organizations building ML/AI applications. Cloud providers exist to let organizations rent exactly the compute/storage/services they need, scaling up or down on demand, paying only for what's actually used — directly enabling the elastic, auto-scaling architectures covered in Lesson 3.

### Historical background
AWS (2006) pioneered large-scale public cloud computing; Azure and GCP followed, each developing distinct strengths — AWS's breadth and market-leading maturity, Azure's enterprise/Microsoft-ecosystem integration, GCP's data/ML tooling heritage (stemming from Google's own internal ML infrastructure, including significant contributions to Kubernetes itself, Lesson 3). By 2026, all three offer comprehensive, largely comparable core infrastructure plus increasingly important, provider-specific managed AI/ML services (foundation model APIs, managed vector databases, MLOps platforms).

### Real-world motivation
Every containerized application (Lesson 2) and Kubernetes cluster (Lesson 3) built across this curriculum needs to actually run *somewhere* in production — for the overwhelming majority of organizations, that "somewhere" is a major cloud provider's managed infrastructure, not self-hosted hardware.

---

## 2. Theory

### The core service categories, consistent across all three providers (naming differs)
| Category | AWS | Azure | GCP |
|---|---|---|---|
| Virtual machines | EC2 | Virtual Machines | Compute Engine |
| Managed Kubernetes | EKS | AKS | GKE |
| Object storage | S3 | Blob Storage | Cloud Storage |
| Managed relational DB | RDS | Azure SQL/Database | Cloud SQL |
| Serverless functions | Lambda | Functions | Cloud Functions |
| Managed ML platform | SageMaker | Azure ML | Vertex AI |

### IaaS, PaaS, and serverless — a spectrum of managed responsibility
- **IaaS (Infrastructure as a Service)**: rent raw virtual machines (EC2/VMs/Compute Engine) — you manage the OS, runtime, and application; the provider manages only the physical hardware/virtualization.
- **PaaS (Platform as a Service)**: rent a managed runtime environment (managed Kubernetes, managed databases) — the provider handles more of the operational burden (patching, scaling infrastructure), you focus on your application/configuration.
- **Serverless**: you provide only application code (a function); the provider manages everything else, including automatically scaling to zero when idle — the most managed, least operationally-burdensome end of the spectrum, at the cost of some flexibility/control and, for certain workloads, cold-start latency.

### Managed Kubernetes (directly extending Lesson 3)
EKS/AKS/GKE handle the Kubernetes **control plane** (the components managing cluster state, scheduling decisions) as a managed service — you don't need to install/patch/scale this yourself, focusing instead on defining your workloads (Deployments, Services, Lesson 3) and (for some configurations) managing worker nodes, or using fully-managed node provisioning (further reducing operational burden) where even node management is automated.

### Managed AI/ML platforms
SageMaker, Azure ML, and Vertex AI each provide integrated environments for the full ML lifecycle (Phase 4-6's model development, this phase's MLOps/deployment concerns) — managed training infrastructure (including GPU provisioning), model registries, and managed inference endpoints — reducing the DevOps/infrastructure burden of running Phase 4-7's models in production, at the cost of some vendor lock-in and often higher cost than self-managing equivalent raw infrastructure.

---

## 3. Mathematical Foundations

### Cost modeling — on-demand vs. reserved vs. spot pricing
Cloud compute pricing typically offers: **on-demand** (pay full price, no commitment, maximum flexibility), **reserved/committed-use** (commit to usage over 1-3 years for a substantial discount, often 30-60%+), and **spot/preemptible** instances (unused capacity at steep discounts, 60-90%+ off, but the provider can reclaim the instance with little notice) — a genuine cost-optimization decision (Lesson 10, this phase) requiring matching workload characteristics (can it tolerate interruption? is usage predictable long-term?) to the appropriate pricing model:
$$
\text{Expected cost} = p_{\text{interrupted}} \times \text{cost}_{\text{restart}} + (1 - p_{\text{interrupted}}) \times \text{cost}_{\text{spot}}
$$
For interruption-tolerant batch workloads (e.g., large-scale model training checkpointed regularly, Phase 6 Lesson 5), spot instances often provide the best expected cost even accounting for occasional restart overhead.

### Multi-region latency and availability tradeoffs
Deploying across multiple geographic regions reduces latency for geographically-distributed users (physical distance directly bounds network latency, per the speed of light — a genuine hard physical constraint, not just an engineering one) and improves availability (a regional outage doesn't take down the entire service) — at the cost of increased architectural complexity (data replication/consistency across regions, a genuinely hard distributed-systems problem) and cost (running redundant infrastructure).

### Availability and the "nines" of uptime
Cloud SLAs (Service Level Agreements) are commonly expressed in "nines" of availability — 99.9% ("three nines") allows roughly 8.7 hours of downtime per year; 99.99% ("four nines") allows roughly 52 minutes per year. Achieving higher availability requires increasingly redundant, complex architecture (multiple availability zones, multiple regions, automated failover) — a direct cost-versus-reliability tradeoff that should be matched to the actual business criticality of the specific application, not maximized indiscriminately (Phase 4 Lesson 3's cost-of-errors reasoning, applied here to infrastructure reliability investment).

---

## 4. Algorithm — Choosing Compute Pricing Model for a Workload (a decision procedure)

```
GIVEN a workload with known characteristics:
IS the workload's usage predictable and long-term (running most of the year, known in advance)?
   YES -> use RESERVED/COMMITTED-USE pricing (substantial discount for the commitment)
IS the workload interruption-tolerant (checkpointed, can restart if interrupted) AND not time-critical?
   YES -> use SPOT/PREEMPTIBLE instances (steep discount, accept occasional interruption risk)
IS the workload highly variable/unpredictable, or short-lived/experimental?
   YES -> use ON-DEMAND pricing (maximum flexibility, no commitment, standard price)
IS the workload's request pattern bursty with LONG idle periods between bursts?
   CONSIDER serverless/scale-to-zero options (pay only for actual invocation time, no idle cost)
```

---

## 5. Python Implementation (infrastructure-as-code example)

```python
"""cloud_core.py — illustrating provider-agnostic infrastructure reasoning + a Terraform-style example"""

# While actual cloud provisioning uses provider SDKs/Terraform (not pure Python), this illustrates
# the DECISION LOGIC (Section 4) an engineer applies when choosing infrastructure configuration:

def recommend_compute_pricing_model(workload: dict) -> str:
    """workload: {'predictable_long_term': bool, 'interruption_tolerant': bool,
                   'bursty_with_idle_periods': bool}"""
    if workload.get("predictable_long_term"):
        return "reserved/committed-use (1-3 year commitment for 30-60%+ discount)"
    if workload.get("interruption_tolerant"):
        return "spot/preemptible instances (60-90%+ discount, accept interruption risk)"
    if workload.get("bursty_with_idle_periods"):
        return "serverless/scale-to-zero (pay only for actual invocation time)"
    return "on-demand (maximum flexibility, standard pricing)"


training_job = {"predictable_long_term": False, "interruption_tolerant": True, "bursty_with_idle_periods": False}
inference_api = {"predictable_long_term": True, "interruption_tolerant": False, "bursty_with_idle_periods": False}
occasional_batch_job = {"predictable_long_term": False, "interruption_tolerant": True, "bursty_with_idle_periods": True}

for name, workload in [("Model training", training_job), ("Production inference API", inference_api),
                        ("Occasional batch job", occasional_batch_job)]:
    print(f"{name}: {recommend_compute_pricing_model(workload)}")
```

```hcl
# main.tf — Terraform (infrastructure-as-code) example: provisioning a managed Kubernetes cluster on AWS (EKS)
# Directly extending Lesson 3's Kubernetes concepts to CLOUD-PROVISIONED infrastructure

resource "aws_eks_cluster" "actuarial_cluster" {
  name     = "actuarial-api-cluster"
  role_arn = aws_iam_role.eks_role.arn

  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }
}

resource "aws_eks_node_group" "actuarial_nodes" {
  cluster_name    = aws_eks_cluster.actuarial_cluster.name
  node_group_name = "actuarial-workers"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.private[*].id

  scaling_config {
    desired_size = 3     # matches Lesson 3's Deployment replica count reasoning, now at the NODE level
    max_size     = 10
    min_size     = 2
  }

  capacity_type = "SPOT"  # Section 4's decision: worker nodes for a fault-tolerant, auto-healing
                            # Kubernetes workload are a good candidate for spot pricing
}
```

---

## 6. Build From Scratch

**A minimal multi-region latency estimator (making Section 3's physical-distance latency constraint concrete):**
```python
import math

SPEED_OF_LIGHT_FIBER_KM_PER_MS = 200   # light in fiber optic cable travels at ~2/3 c; ~200km/ms is a common estimate

def estimate_min_latency_ms(distance_km: float) -> float:
    """A LOWER BOUND on round-trip latency -- real latency is always higher due to routing,
    processing, and non-straight-line cable paths, but this establishes the HARD PHYSICAL FLOOR."""
    one_way_ms = distance_km / SPEED_OF_LIGHT_FIBER_KM_PER_MS
    return 2 * one_way_ms   # round trip

# Example: estimating latency floor for users in different regions accessing a single-region deployment
distances = {
    "Same region (local)": 50,
    "Cross-country (e.g., US East to West)": 4000,
    "Cross-continent (e.g., US to Europe)": 8000,
    "Cross-continent (e.g., US to Asia)": 12000,
}
for region, distance in distances.items():
    print(f"{region}: minimum round-trip latency ~ {estimate_min_latency_ms(distance):.1f} ms (physical floor)")
```
This concretely demonstrates why a single-region deployment imposes a hard, physics-bound latency floor for geographically distant users — no amount of software optimization can beat the speed of light, directly justifying multi-region deployment (Section 3) for latency-sensitive, globally-distributed applications, while also showing why it's *unnecessary* overhead for applications with a genuinely regional/local user base.

---

## 7. Library/Tool Comparison

| From scratch/manual | Production tooling |
|---|---|
| `recommend_compute_pricing_model` (decision logic) | Cloud provider cost calculators and dedicated cost-optimization tools (AWS Cost Explorer, and similar) that analyze actual usage patterns |
| Manual cloud resource provisioning (clicking through a web console) | Terraform, Pulumi, or provider-native IaC tools (AWS CloudFormation, Azure Bicep) — version-controlled, reproducible infrastructure provisioning |
| `estimate_min_latency_ms` (simplified) | Real-world latency monitoring tools and CDN providers' actual measured latency data across regions |

---

## 8. Visual Explanations

**The IaaS/PaaS/Serverless management-responsibility spectrum:**
```
IaaS (EC2/VM/Compute Engine)      PaaS (managed K8s, managed DB)      Serverless (Lambda/Functions)
YOU manage: OS, runtime,          Provider manages: control plane,   Provider manages: EVERYTHING
  patching, scaling                 patching; YOU manage: workloads    except your function's code
[more control, more burden] ─────────────────────────────────────▶ [less control, less burden]
```

**Multi-region latency floor (Section 6):**
```
Single-region deployment:                    Multi-region deployment:
  US user ──▶ [US datacenter]  (fast)           US user ──▶ [US datacenter]     (fast)
  Asia user ──▶ [US datacenter] (SLOW,           Asia user ──▶ [Asia datacenter] (fast, LOCAL)
    ~150ms+ physical floor)                        (requires data replication/sync between regions)
```

---

## 9. Practical Examples

**Simple:** provision a small virtual machine on a free-tier cloud account and deploy a simple containerized application (Lesson 2) to it manually.
**Medium:** set up a managed Kubernetes cluster (EKS/AKS/GKE) via a cloud console or Terraform, and deploy the Phase 8 Lesson 3 manifests to it.
**Real-world:** estimate and compare the monthly cost of running your actuarial API on-demand versus with a committed-use/reserved pricing plan, for an assumed steady, predictable traffic level, quantifying the savings.

---

## 10. Real Industry Use Cases

- **Virtually every production ML/LLM application at scale**: runs on AWS, Azure, or GCP (or a combination) rather than self-hosted infrastructure, for the overwhelming majority of organizations.
- **Managed AI/ML platforms** (SageMaker, Azure ML, Vertex AI): widely used specifically to reduce the DevOps burden of training and serving models at scale, integrating directly with each provider's broader infrastructure ecosystem.
- **Spot/preemptible instances for ML training**: a common, substantial cost-saving practice (Lesson 10) for large-scale model training jobs (Phase 6 Lesson 5's pretraining, at smaller organizational scale) that can tolerate occasional interruption with proper checkpointing.
- **Multi-cloud and hybrid-cloud strategies**: some organizations deliberately use multiple providers (for redundancy, cost negotiation leverage, or to access specific provider-unique services) — a genuine, if operationally complex, architectural choice.

---

## 11. Common Mistakes

- Defaulting to on-demand pricing for genuinely predictable, long-running workloads — leaving substantial, easily-captured cost savings (reserved pricing) unused.
- Using spot/preemptible instances for workloads that aren't genuinely interruption-tolerant (e.g., a stateful service without proper checkpointing) — risking data loss or service disruption when the instance is reclaimed.
- Deploying to a single region for a genuinely global user base without considering the hard physical latency floor (Section 6) — resulting in poor experience for distant users that no software optimization can fix.
- Manually provisioning infrastructure via a web console rather than infrastructure-as-code (Terraform, etc.) — loses reproducibility, version control, and review-ability of infrastructure changes, directly undermining Phase 1 Lesson 5's version-control discipline at the infrastructure level.

---

## 12. Best Practices (2026)

- Use infrastructure-as-code (Terraform, or a provider-native equivalent) for all cloud resource provisioning, treating infrastructure definitions with the same version-control and review discipline as application code.
- Match compute pricing models to actual workload characteristics (Section 4's decision procedure) rather than defaulting to on-demand pricing universally.
- Use managed Kubernetes (EKS/AKS/GKE) rather than self-managing cluster control planes, unless there's a specific, well-justified reason not to.
- Consider multi-region deployment specifically for latency-sensitive applications with a genuinely geographically-distributed user base, while avoiding the added complexity for applications that don't need it.

---

## 13. Exercises

**Easy:** Provision a free-tier virtual machine on any major cloud provider and SSH into it, confirming basic connectivity.
**Medium:** Write a Terraform configuration provisioning a small managed Kubernetes cluster, and apply it (on a free-tier or trial account if available).
**Hard:** Estimate the monthly cost difference between on-demand and reserved pricing for a hypothetical steady-state production workload (using each provider's published pricing), and compute the break-even point at which reserved pricing becomes cost-effective given the upfront commitment.
**Mathematical:** Using Section 6's latency estimator, compute the theoretical minimum round-trip latency for users in 3 different global regions accessing a single-region deployment, and discuss at what latency threshold you'd consider multi-region deployment necessary for a given application type (e.g., real-time chat vs. an overnight batch report).
**Coding:** Extend the Section 5 `recommend_compute_pricing_model` function to also factor in estimated monthly usage hours, producing an estimated cost comparison across all three pricing models for a given workload profile.

---

## 14. Mini Project

Design and document a **complete cloud deployment architecture for your actuarial API application**: choose a specific cloud provider and justify the choice; write Terraform (or equivalent IaC) configuration provisioning a managed Kubernetes cluster, appropriate compute pricing models for different workload types (the API itself vs. any periodic batch retraining jobs), and necessary supporting services (managed database, object storage for model artifacts); estimate monthly cost under your chosen configuration; and produce an architecture diagram and written justification covering your region choice, pricing model choices, and availability/redundancy tradeoffs.

---

## 15. Interview Preparation

- Explain the difference between IaaS, PaaS, and serverless, and give an example of when you'd choose each.
- What factors would lead you to choose reserved, spot, or on-demand pricing for a given workload?
- Why does multi-region deployment matter for latency, and what's the hard physical constraint underlying it?
- What are the tradeoffs of using a managed Kubernetes service (EKS/AKS/GKE) versus self-managing a cluster's control plane?

---

## 16. Summary

Cloud providers (AWS, Azure, GCP) supply the on-demand, elastically-scalable infrastructure that runs virtually every production ML/AI application built across this curriculum — offering a spectrum of managed responsibility from raw IaaS virtual machines through managed Kubernetes (directly extending Lesson 3) to fully serverless compute, alongside integrated managed AI/ML platforms reducing the operational burden of Phase 4-7's model development lifecycle. Matching compute pricing models (on-demand, reserved, spot) to actual workload characteristics, and understanding the hard physical latency constraints motivating (or not motivating) multi-region architecture, are the genuinely consequential engineering decisions this lesson equips you to make deliberately — directly setting up Lesson 10's broader cost-optimization discipline and completing the infrastructure foundation for Lesson 12's full production AI systems.

---

## 17. References

- AWS, Azure, and GCP official documentation and architecture best-practices guides (each provider's "Well-Architected Framework" or equivalent)
- Terraform official documentation (HashiCorp)
- Erl, Puttini, Mahmood — *Cloud Computing: Concepts, Technology & Architecture*
- Each provider's managed ML platform documentation (SageMaker, Azure ML, Vertex AI)
