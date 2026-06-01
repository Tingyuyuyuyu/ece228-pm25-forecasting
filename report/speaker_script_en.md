# Speaker Script - 3-minute presentation

Target total: 180 seconds. Speak at a conversational pace (~150 words/min).
The full body is ~450 words. Pause briefly between slides; don't rush.

Words in `*italics*` are optional / add if you have time, drop if you're behind.

---

## Slide 1 - Title (5 sec)

> Hi everyone, we're Team 17. I'm [name], and this is our project on
> short-term PM2.5 forecasting using the Beijing Multi-Site Air Quality
> dataset.

*[Click to next slide]*

---

## Slide 2 - Why forecast PM2.5? (30 sec, ~75 words)

> PM2.5 is fine particulate matter -- the main driver of haze in dense
> cities. It's small enough to reach deep into the lungs, and it's linked
> to both respiratory and cardiovascular disease. Seventy-five micrograms
> per cubic meter is the WHO and Chinese national-standard "unhealthy"
> threshold. If we can forecast a few hours ahead, cities can issue
> advisories *before* a haze episode hits.
>
> We set out to answer two questions: first, do sequence models actually
> beat simple feature-based regressors here? And second, does pulling in
> data from the *other* eleven stations help?

*[Click to next slide]*

---

## Slide 3 - Setup (30 sec, ~80 words)

> Our setup is a sliding 24-hour window of pollutant and weather
> measurements; we predict PM2.5 one hour or six hours ahead. We use a
> strict time-based split -- train on 2013 through 2015, validate on the
> first half of 2016, test on the rest.
>
> We standardize using train-set statistics only, and we clip features
> at plus-or-minus eight sigma -- that one fix solved an early problem
> where rainfall outliers were causing NaN losses in the GRU.
>
> The two input settings: *global* is one model across all twelve
> stations, each example seeing only its own station's history; the
> *multi-site* setting adds the channel-wise mean of the other eleven
> stations at every timestep. We mask the target station out to avoid
> leakage. Our headline metric is spike-recall: of all the unhealthy
> hours in the test set, how many did the model also flag.

*[Click to next slide]*

---

## Slide 4 - Models (25 sec, ~60 words)

> We compared seven models in three families. A persistence baseline
> that just predicts y at t-plus-h equals y at t -- surprisingly tough
> to beat at short horizons. Three feature-based baselines: Ridge
> regression, Random Forest, and Gradient Boosting, all on the flattened
> twenty-four-by-F window. And three sequence models: LSTM, GRU, and a
> small Transformer encoder, each with an eight-dimensional station
> embedding.
>
> Same data, same loss, same metrics across everything -- so the
> comparison is apples to apples.

*[Click to next slide]*

---

## Slide 5 - Result 1 (45 sec, ~110 words)

> Here's the headline result. At one hour ahead, persistence is already
> a strong baseline -- R-squared around point-nine-five. But the
> sequence models with multi-site context push MAE down to about
> nine-point-four. *Transformer multi-site is 9.41, GRU multi-site is
> 9.51.* Spike-recall stays around ninety-five percent -- meaning if a
> PM2.5 reading is about to cross the unhealthy threshold, we flag it
> nineteen times out of twenty.
>
> At six hours ahead, persistence collapses to R-squared point-five-six.
> *Every* model jumps to MAE around 30. The best single model here is
> GRU multi-site at 28.75, narrowly beating gradient boosting at 29.00.
> LSTM struggles -- that's a learning-rate and patience tuning issue at
> the longer horizon, not an architecture issue.
>
> The take-away: sequence models pay off most when they're paired with
> the multi-site context.

*[Click to next slide]*

---

## Slide 6 - Result 2 (35 sec, ~90 words)

> Our second result is about the multi-site setting itself. We fed in
> the channel-wise mean of the other eleven stations -- the simplest
> possible spatial summary. No graph, no attention.
>
> The MAE delta is *negative for every model at every horizon* -- meaning
> multi-site is consistently better than global. And the gains are
> *larger* at six hours ahead: GRU drops 2.55 micrograms per cubic meter,
> LSTM drops almost 4. That makes sense -- the further out you predict,
> the less your own station's recent history alone can tell you.
>
> On the right, you can see GRU multi-site one-hour predictions tracking
> the actual PM2.5 curve at Aotizhongxin station -- it catches the
> daily cycle and even the spikes above the unhealthy threshold.

*[Click to next slide]*

---

## Slide 7 - Wrap-up (15 sec, ~40 words)

> To wrap up: GRU multi-site is our best model at both horizons. A
> trivial spatial mean already gives a consistent boost. *Next steps:
> a wind-direction-aware aggregator, proper LR tuning for the six-hour
> horizon, and multi-task forecasting across all six pollutants.*
>
> Thanks -- happy to take questions.

---

## Timing notes

| Slide | Target | Cumulative |
|-------|--------|-----------:|
| 1     | 5s     | 0:05 |
| 2     | 30s    | 0:35 |
| 3     | 30s    | 1:05 |
| 4     | 25s    | 1:30 |
| 5     | 45s    | 2:15 |
| 6     | 35s    | 2:50 |
| 7     | 15s    | 3:05 |

If you're running short on time during practice, the safest things to
cut are:
- *italicized* lines (marked above)
- Detailed numbers on slide 5 (just say "around 9.4" instead of "9.41")
- The "twenty-four-by-F window" detail on slide 4

If you're running long, you can drop the *italicized* sentences on the fly.

---

## Delivery tips

1. **Pace** - 3 minutes is tight. Practice once with a stopwatch; aim
   for 2:50 so you have buffer for a slow start or a stumble.
2. **Pause** - half a second between slides reads as confident, not slow.
3. **Numbers** - drill the four key numbers so they come out smoothly:
   - 9.41 (best 1h MAE)
   - 28.75 (best 6h MAE)
   - 95% (spike-recall at h=1)
   - 2.55 (multi-site gain on GRU at h=6)
4. **The MAE chart on slide 5** is the visual centerpiece. Point at the
   short green and red bars when you mention GRU multi-site.
5. **End strong** - finish with "Thanks - happy to take questions"
   *before* the slide changes, then click. Don't read off the slide.
