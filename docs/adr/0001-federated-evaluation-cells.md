# ADR 0001: Federated evaluation cells, not a single central platform

**Status:** Accepted (design). **Date:** 2026-11.

## Context
One company, plants in the US, Europe, India and Asia. Operator queries and retrieved
documents may contain personal data, trade secrets or export-controlled content, and several
jurisdictions restrict cross-border transfer. Some plants are air-gapped.

## Decision
A global control plane owns definitions (metrics, policies, judge versions, dataset catalog).
Each data-residency zone runs its own evaluation cell holding traces, goldens with real content
and the run ledger. Only aggregate scores, counts and IDs cross borders.

## Consequences
- (+) Residency and IP risk are contained by construction.
- (+) A cell can run fully offline with a local judge.
- (-) Cross-cell comparability needs discipline: one metric implementation, pinned judge packs,
  and a small non-sensitive anchor set run in every cell to detect judge drift.
- (-) More deployments to operate. LangChain's guidance for self-hosted LangSmith is one
  instance and one organization per geography, which fits this model.
