#ATPP
Alternate Trade Processing Prediction

# Project 8: Project Summary

## Research question

The team is testing whether the **content** of corporate disclosures predicts how trading activity changes after they are released. The question is not where the price goes. It is how fast trades arrive and how they cluster. Clustered, accelerating trading is a sign of informed traders, so a text signal that predicts it ahead of time would let a market maker protect its quotes before toxic flow shows up.

## Why market making, not return prediction

Most NLP-in-finance work turns sentiment into a return forecast. That space is crowded, and price direction is close to efficient. The team routes text through a different variable: the adverse-selection parameter ξ in the Avellaneda–Stoikov market-making framework, which is the probability that the next trade is informed.

Existing work estimates ξ from order flow, which only reveals informed activity after it has started. Text is available before that. The most recent extension of AS (Barzykin–Bergault–Guéant–Lemmel, 2025) explicitly assumes the market maker has no informational advantage. The project relaxes that assumption.

## The staged plan

The team builds in three stages. Each stage is only worth starting if the one before it holds up on real data.

- **Version C (current).** A duration model of the gaps between trades, with text-derived event features added as regressors. Question: does text shift arrival timing beyond what trade history and time of day explain?
- **Version A.** A Hawkes (self-exciting) process whose clustering parameters, the branching ratio and the decay rate, are functions of the text. Question: does text predict the *shape* of the burst, not just its level?
- **Version B.** A mixture-of-experts model in which the text selects the arrival regime (near-Poisson, two kinds of Weibull, or Hawkes) and estimates its parameters. Family labels come from held-out fits, and predictions are written down before training.

The long-term payoff (Project 1) is to show that a market maker with this signal has lower adverse-selection cost than one relying on order flow alone, measured as a function of lead time and signal reliability.

## Current design of Version C

Several choices changed from the original plan, each for a stated reason:

- **Log-ACD instead of linear ACD.** A negative text effect can push a linear model's expected duration below zero; in logs this cannot happen, and the recursion runs as a fast linear filter.
- **Two event features instead of one flag.** 8-K item codes split events into *scheduled* (Item 2.02 earnings) and *surprise* (other items). Each gets its own time-decayed feature.
- **Diurnal adjustment.** Durations are divided by a time-of-day cubic spline with 30-minute knots, fitted on **non-event days only**, together with a trailing 20-session level. Normalizing each day by its own mean would divide out the very effect being tested, so it is avoided.
- **Long-run effect as the reported quantity.** The team reports γ/(1 − β) rather than γ itself, because simulation showed γ is biased when β is, while the ratio is recovered reliably.
- **Inference.** Robust (sandwich) Wald tests replace the naive likelihood-ratio test, because the exponential likelihood is misspecified on real durations. Results are also checked against a distribution of at least 200 matched placebo events.

## Data

- **Trades:** about two years of Databento `trades` data for one stock. This replaced the free LOBSTER samples, which stop at June 2012 and so never overlap the period EDGAR's API covers (2015 onward).
- **Events:** SEC EDGAR 8-K acceptance timestamps, to the second.

Cleaning steps include:
- dropping off-exchange TRF prints, whose timestamps record report time rather than execution time;
- merging simultaneous prints from a single sweeping order into one event;
- an optional buffer after the opening auction;
- a diagnostic that resolves whether EDGAR acceptance times are in UTC or ET, using EDGAR's documented 06:00–22:00 ET acceptance window.

## What has been established

- **The estimator works.** On synthetic data, the code recovers the true parameters, detects a real text effect, and stays silent in placebo runs where no effect exists.
- **Power is the binding constraint.** The text effect is identified by the number of *event windows*, not the number of trades. One ticker has roughly 8 earnings releases and 10–30 other 8-Ks over two years. The analytic p-value can look extremely small (1e-77 in one synthetic run) while the placebo randomization p-value sits at its floor of 1/21 with 20 placebos. One ticker is therefore a pilot. A real result needs a panel of 20–50 liquid names with a shared text effect, which gives a few hundred event windows.
- **Most "scheduled" events land at the open.** Large-cap earnings 8-Ks usually arrive outside market hours and map to 09:30, so the scheduled effect is mostly identified as "the open on earnings mornings vs. the open on ordinary mornings."

## Novelty, stated honestly

The team considers the current Version C a necessary foundation, not the contribution:
- Adding exogenous regressors to ACD models dates to the late 1990s.
- Event flags, even split by item code, are close to existing practice.

The contribution starts with two things:
- Replacing the event flags with a **surprise measure**: the perplexity of the filing given the firm's disclosure history, which is separate from sentiment.
- Showing that text conditions the **clustering structure** of informed flow (Versions A and B).

Literature searches found no prior work combining text-derived surprise with duration or Hawkes models in microstructure. That is a negative search result, not proof the gap is empty.

## Pre-registered pass criteria for Version C

The specification is fixed before looking at real data: 1800-second half-life, exponential QMLE, 30-minute knots, spline fitted on non-event days, 20-day trailing level. A pass requires all of the following:

1. Both the scheduled and surprise effects are negative.
2. The joint robust Wald test has p < .01.
3. The placebo randomization p-value is below .01 for each event class, using at least 200 placebos.
4. The placebo mean is within one placebo standard deviation of zero.
5. For intraday events, a scan over event-time offsets peaks within ±2 minutes of the filing time. A peak well before the filing would mean the market reacted before the 8-K.

## Open decisions

- **Item taxonomy.** Item 8.01 is a grab bag, and 2.01 is often anticipated. The team plans to read 20–30 of the ticker's filings before fixing the taxonomy.
- **Panel timing.** Whether to expand to the multi-ticker panel now, or after the single-ticker pilot. This depends partly on Databento's cost for additional symbols.
- **Open buffer length.** To be set after inspecting the raw 09:30 bucket.

## Next steps

1. Run the timestamp diagnostic and the full single-ticker pilot under the pre-registered specification.
2. Price out the multi-ticker panel.
3. Build the perplexity-based surprise feature, the step where the novelty claim begins.
4. Move to Version A once Version C passes on the panel.

I can turn this into a Doc for sharing with the team or an advisor.
