# Text-Conditioned Informed-Flow Arrival Distributions

## Overview

This project tests whether the content of corporate disclosures (SEC 8-K filings) predicts how trading activity changes after they are released. It does not forecast price direction. The target is the **inter-arrival time of trades**: how fast trades arrive, and what shape their timing distribution takes.

The motivation is market making. In the Avellaneda–Stoikov (AS) framework, a market maker's main risk is adverse selection, trading against someone who knows more. Informed traders tend to arrive in bursts around information events, so the arrival pattern is a signal of how toxic the incoming flow is. Existing AS extensions estimate the informed-flow probability ξ from order flow, which only reveals informed activity after it has started. Disclosure text is available before that. The most recent AS extension (Barzykin–Bergault–Guéant–Lemmel, 2025) assumes the market maker has no informational advantage. This project relaxes that assumption.

## Research question

Do 8-K disclosures change the rate and the distributional shape of trade arrivals, and does how *surprising* a disclosure is predict which change occurs?

| | Hypothesis | Observable |
|---|---|---|
| H1 Level | After an 8-K, expected trade durations shrink, and scheduled and surprise filings differ | Long-run effect γ/(1−β) < 0 |
| H2 Shape | The shape of the duration distribution changes after an 8-K, more for high-surprise filings | Weibull shape k falls below baseline in event windows |
| H3 Surprise | A perplexity-based surprise score explains H1 and H2 beyond item-code flags and beyond sentiment | Incremental out-of-sample fit |
| H4 Regime | Text predicts which arrival-distribution family governs a window | Gating accuracy against held-out family labels |

The decision value (lower adverse-selection cost for a market maker) is the motivation. It is not tested until the final stage.

## Approach

Each stage is built only if the previous one holds up on real data.

| Stage | Question | Model | Text input |
|---|---|---|---|
| 0. Data | Can events and trades be aligned reliably? | none | none |
| 1. Level | H1 | Log-ACD, exponential QMLE | Scheduled and surprise flags from 8-K item codes |
| 2. Shape | H2 | Weibull or generalized gamma errors, with k a function of event features | Same flags |
| 3. Surprise | H3 | Stages 1–2 re-run | Perplexity of the filing given the firm's prior filings |
| 4. Clustering | Does text set the clustering parameters? | Per-window Hawkes (α, β) regressed on text features | Stage 3 features |
| 5. Regime | H4 | Mixture of experts over near-Poisson, two Weibull regimes, and Hawkes | Stage 3 features |
| 6. Decision | Does it lower adverse-selection cost? | AS simulator, swept over lead time and signal reliability | Stage 4 or 5 output as ξ |

The bounded thesis scope is Stages 0–3, with Stage 4 if time allows. Stages 5–6 are follow-on work.

## Data

- **Trades:** Databento `trades` schema (Nasdaq TotalView-ITCH, 2018 onward), for a panel of liquid names across sectors.
- **Events:** SEC EDGAR 8-K acceptance timestamps, to the second.
- **Why not LOBSTER:** its free samples stop at June 2012, before EDGAR's API coverage begins (2015).

Cleaning rules: off-exchange TRF prints are dropped, since their timestamps record report time rather than execution time. Simultaneous prints from one sweeping order are merged into a single event. An optional buffer after the opening auction is set after inspecting the raw 09:30 bucket. EDGAR acceptance times are checked against the documented 06:00–22:00 ET window to resolve whether they are UTC or ET.

## Methodology

- **Log-ACD** instead of linear ACD, so a negative text effect cannot push expected duration below zero. The recursion also runs as a fast linear filter.
- **Diurnal adjustment:** durations are divided by a time-of-day cubic spline with 30-minute knots, fitted on non-event days only, plus a trailing 20-session level. Per-day normalization is avoided because it would divide out the effect being tested.
- **Reported quantity:** the long-run effect γ/(1−β). In simulation, γ is biased when β is, but the ratio is recovered reliably.
- **Inference:** robust (sandwich) Wald tests, because the exponential likelihood is misspecified on real durations. Results are checked against at least 200 matched placebo events.
- **Power:** the effect is identified by the number of event windows, not the number of trades. One ticker over two years gives roughly 8 earnings releases and 10–30 other 8-Ks, so a single ticker is a pilot. A real result needs a panel of 20–50 names.

## Pre-registered pass criteria for Stage 1

The specification is fixed before looking at real data: 1800-second half-life, exponential QMLE, 30-minute knots, spline fitted on non-event days, 20-day trailing level. A pass requires all of the following:

1. The scheduled and surprise effects are both negative.
2. The joint robust Wald test has p < .01.
3. The placebo randomization p-value is below .01 for each event class, with at least 200 placebos.
4. The placebo mean is within one placebo standard deviation of zero.
5. For intraday events, the log-likelihood over event-time offsets peaks within ±2 minutes of the filing time.

Diagnostics reported but not part of the test: half-life profile, Weibull k, Ljung–Box on residuals, naive likelihood-ratio test.

## Novelty

- **Stage 1 is not novel.** Exogenous regressors in ACD models date to the late 1990s, and item-code event flags are close to existing practice. It is a foundation.
- **Stage 2** is a modest extension: announcement features shifting the shape of the duration distribution.
- **Stage 3** is the first likely-novel step. Literature searches found no prior work combining text-derived surprise with duration or point-process models in microstructure. That is a negative search result, not proof that nothing exists.
- **Stages 4–5** (text-conditioned Hawkes kernel parameters, and text-driven family selection) were not found in the literature searches.

Nearest related work: Barzykin–Bergault–Guéant–Lemmel (2025), Chevalier–Hafsi–Ly Vath (2024, 2025), Lillo et al. (arXiv:1405.6047) on Hawkes processes around macro news, and Jafree–Jain–Firoozye (arXiv:2510.27334) on Hawkes-LOB reinforcement learning without text.

## Status

- The ACD estimator, simulator, and likelihood-ratio test are implemented. Parameter recovery, effect detection, and a silent placebo all pass on synthetic data. These checkpoints ran on the **linear** EACD, so they must be re-run on the log-ACD before any real-data fit.
- The EDGAR 8-K collector and a duration parser are written.
- Real-data estimation has not been run yet. Next steps: price and pull the Databento panel, run the timestamp diagnostic, re-validate the log-ACD, then fit the Stage 1 pilot under the pre-registered specification.

## Open decisions

- **Panel size**, set by Databento pricing for additional symbols.
- **8-K item taxonomy:** 8.01 is a grab bag and 2.01 is often anticipated, so 20–30 filings per ticker will be read before the split is fixed.
- **Perplexity model:** a small model trained only on filings before each event date, or a pretrained model with a look-ahead leakage audit. A pretrained LLM has probably seen these filings during pretraining.
