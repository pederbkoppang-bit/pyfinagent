---
name: terminal-outcome-vocabularies
description: Prior art for terminal outcome vocabularies and not-counting infrastructure failures against a retry ceiling -- k8s, Temporal, SRE, OTel overflow, Stripe idempotency, and the live 2026 exclude-vs-score-zero debate
metadata:
  type: reference
---

Fetched-in-full prior art for "what vocabulary does a bounded retry loop record, and how
does it avoid counting an infrastructure failure against the ceiling". Reusable for any
attempt-accounting or termination step.

- **Kubernetes pod failure policy** --
  https://kubernetes.io/blog/2024/08/19/kubernetes-1-31-pod-failure-policy-for-jobs-goes-ga/
  The discriminator is a first-class condition ON THE RECORD (`DisruptionTarget`), and the
  POLICY chooses: `Ignore` = "Do not count the failure towards the backoffLimit";
  `Count` = the default. Terminal Job conditions are a closed set (`Complete`, `Failed`,
  `FailureTarget`, `SuccessCriteriaMet`, `Suspended`) each carrying a machine REASON
  (`BackoffLimitExceeded`, `DeadlineExceeded`, `PodFailurePolicy`, ...).
  Split record-half from policy-half; that is the mainstream design.
- **Temporal** -- https://docs.temporal.io/workflow-execution -- six CLOSED statuses:
  Completed, Failed, Timed Out, Cancelled, Terminated, Continued-As-New. Three distinct
  non-success ends, not one.
- **Google SRE** -- https://sre.google/sre-book/handling-overload/ -- runs TWO budgets:
  a per-request cap of 3 attempts AND a cumulative client-side "ratio of requests that
  correspond to retries ... below 10%". Also "only the layer immediately above" retries.
- **OTel cardinality limits** --
  https://opentelemetry.io/blog/2026/cardinality-limits-in-opentelemetry/ -- the
  unknown-key answer: default 2000 combos, then fold into ONE visible overflow point
  (`otel.metric.overflow=true`) with the original attributes removed. Not dropped, not
  unbounded.
- **Stripe idempotency** -- https://docs.stripe.com/api/idempotent_requests -- existence is
  not validity ("compares incoming parameters to those of the original request and errors
  if they're not the same"), and key EXPIRY silently mints new work ("We generate a new
  request if a key is reused after the original is pruned"). Also: a request failing
  validation is deliberately NOT recorded.
- **LIVE DEBATE, both 2026, cite both:** DeepSWE https://arxiv.org/html/2607.07946v1 §5.6
  EXCLUDES infra-terminated rollouts "from both the numerator and the denominator ... rather
  than scoring them as failures" (excluded fraction 0%-5.3%, not resampled), while
  https://arxiv.org/html/2607.12227v1 §A.5 scores them "r=0 rather than excluded from the
  average". Recording the outcome keeps both computable; deciding it inside the counter
  does not.
- **Absence is a finding:** HarnessFix https://arxiv.org/html/2606.06324v2 shows published
  agent-harness research still does NOT separate harness faults from task failures in its
  headline metric -- so borrow vocabulary from workflow/batch systems, not from that
  literature.
