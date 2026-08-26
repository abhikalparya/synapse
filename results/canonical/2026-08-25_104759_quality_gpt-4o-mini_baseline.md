# Benchmark report — 2026-08-25T10:47:59.890625+00:00

- Benchmark: `quality`
- Dataset: `learning_graph_quality_v1`
- Dataset version: `learning_graph_quality_v1`
- Model: `gpt-4o-mini`
- Provider: `openai`
- Seed: `42`
- Examples: 33
- Repetitions: 3
- Prompt variant: `baseline`
- Prompt version: `baseline@1655002ff87c9db0`
- Prompt hash: `1655002ff87c9db0`

## Graph quality (macro-average)

| System | Topic F1 | Dependency F1 | Missing Prereq | Direction Error | Extra Dep | Redundant Transitive | Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synapse | 0.487 | 0.100 | 0.823 | 0.039 | 0.885 | 0.010 | 1.000 |
| domain_curriculum_prior | 0.787 | 0.188 | 0.637 | 0.094 | 0.757 | 0.065 | 1.000 |

## Latency (graph-quality path)

| System | samples | p50_ms | p95_ms | mean_ms |
| --- | ---: | ---: | ---: | ---: |
| synapse | 99 | 4014.21 | 5154.262 | 4087.563 |
| domain_curriculum_prior | 99 | 6661.755 | 8738.981 | 6492.28 |

## Estimated cost

| System | avg_cost_usd | total_cost_usd | note |
| --- | ---: | ---: | --- |
| synapse | 0.0002857469696969697 | 0.02828895 | USD estimates from versioned pricing table; null when model/tokens unavailable. |
| domain_curriculum_prior | None | None | USD estimates from versioned pricing table; null when model/tokens unavailable. |

## Failures

```
Failure Type                 Count
----------------------------------
Extra dependency               198
Incorrect dependency           198
INVALID_EXTRA_EDGE             198
Missing prerequisite           176
Hallucinated topic             157
Missing topic                  109
Wrong dependency direction      41
Redundant transitive edge       33
MATCHED_ACCEPTABLE_ALTERNATIVE_EDGE     8
Cycle attempt                    4
```

## Proposal / human-feedback metrics

```json
{
  "available": true,
  "note": "Rates use recorded apply/discard events only. Proposals cannot currently be edited in-app, so modification_rate reflects fingerprint mismatches (normally 0).",
  "counts": {
    "proposals_created": 0,
    "proposals_applied": 1,
    "proposals_discarded": 0,
    "proposals_modified_events": 0,
    "rollbacks": 1,
    "invalid_edges_caught": 0,
    "cycle_causing_edges_rejected": 0,
    "out_of_scope_or_invalid_refs_rejected": 0
  },
  "rates": {
    "acceptance_rate": 1.0,
    "rejection_rate": 0.0,
    "modification_rate": 0.0,
    "accepted_unchanged_rate": 1.0
  },
  "average_edits_before_application": 0.0,
  "deterministic_rejections_by_category": {},
  "confidence_calibration": {
    "0.0\u20130.2": {
      "n": 0,
      "acceptance_rate": null,
      "modification_rate": null,
      "rejection_rate": null,
      "note": "insufficient data"
    },
    "0.2\u20130.4": {
      "n": 0,
      "acceptance_rate": null,
      "modification_rate": null,
      "rejection_rate": null,
      "note": "insufficient data"
    },
    "0.4\u20130.6": {
      "n": 0,
      "acceptance_rate": null,
      "modification_rate": null,
      "rejection_rate": null,
      "note": "insufficient data"
    },
    "0.6\u20130.8": {
      "n": 0,
      "acceptance_rate": null,
      "modification_rate": null,
      "rejection_rate": null,
      "note": "insufficient data"
    },
    "0.8\u20131.0": {
      "n": 1,
      "acceptance_rate": 1.0,
      "modification_rate": 0.0,
      "rejection_rate": 0.0
    }
  },
  "claim": "Do not treat LLM self-reported confidence as calibrated probability unless bucket outcomes demonstrate correlation."
}
```

_Do not invent numbers. Empty/null rates mean insufficient recorded events._
