# Speaker Script - 3-minute presentation (3-slide version)

Target total: 180 seconds across 3 slides. About 60 seconds per slide,
roughly 150 words spoken per slide at a comfortable pace.

Words in `*italics*` are optional / drop if you're behind.

---

## Slide 1 - Problem + setup (60 sec, ~150 words)

> Hi everyone, we're Team 17 -- Ting-Yu and Yun-Chen -- and our project
> is short-term PM2.5 forecasting on the Beijing Multi-Site Air Quality
> dataset.
>
> PM2.5 is fine particulate matter, the main driver of haze in dense
> cities. Seventy-five micrograms per cubic meter is the WHO and Chinese
> national-standard threshold for "unhealthy" air -- the level where
> cities issue advisories. If we can forecast a few hours ahead, those
> advisories can go out *before* a spike hits.
>
> We set out to answer two questions: first, do sequence models beat
> simple feature-based regressors here, and second, does pulling in data
> from the *other* eleven stations help.
>
> The data is hourly readings from twelve Beijing monitoring stations
> over four years. We use a 24-hour input window to predict PM2.5 one
> hour and six hours ahead, with a strict time-based train-val-test
> split. *We standardize on train only and clip to plus-or-minus eight
> sigma -- that one fix solved a rainfall-outlier issue that was causing
> NaN losses in the GRU.*
>
> We compare seven models across three families: persistence as a
> no-learning baseline; ridge, random forest, and gradient boosting as
> feature-based baselines; and LSTM, GRU, and a Transformer encoder as
> sequence models. Same data, same loss, same metrics across all seven.

*[Click to slide 2]*

---

## Slide 2 - Headline results (60 sec, ~140 words)

> Here's the headline. This is test MAE for every model, in both input
> settings, at both horizons. Green and red bars are multi-site; blue
> and orange are global.
>
> At one hour ahead, persistence is already a strong baseline -- R-squared
> around point-nine-five. But Transformer multi-site reaches MAE of
> nine-point-four-one, and GRU multi-site is nine-point-five-one. Both
> edge past every baseline. Spike-recall stays around ninety-five percent
> -- meaning we flag nineteen out of every twenty unhealthy hours.
>
> At six hours ahead, the picture changes. Persistence collapses to
> R-squared around point-five-six -- the problem is much harder. And the
> gap between methods opens up. *GRU multi-site reaches MAE of
> twenty-eight-point-seven-five, just edging out gradient boosting
> multi-site at twenty-nine-point-zero-zero.*
>
> The take-away is that sequence models pay off most when they're paired
> with the multi-site spatial context. *You can see LSTM struggling at
> both horizons -- that's a learning-rate and patience tuning issue at
> our default settings, not a model-family issue.*

*[Click to slide 3]*

---

## Slide 3 - Multi-site result + wrap-up (60 sec, ~145 words)

> Our second result is about the multi-site setting itself. We just
> feed in the channel-wise mean of the other eleven stations -- no
> graph, no attention, the simplest possible spatial summary.
>
> The MAE delta is negative for *every* model at *every* horizon --
> multi-site is consistently better. And the gains are *larger* at six
> hours ahead: GRU drops two-point-five-five micrograms per cubic meter,
> ridge drops one-point-eight-seven. That makes sense -- the further
> out you predict, the less your own station's recent history alone
> can tell you.
>
> On the right, you can see GRU multi-site one-hour predictions tracking
> the actual PM2.5 curve at Aotizhongxin station for three weeks of test
> data. It catches both the daily cycle and the multi-day haze episodes
> that cross the unhealthy threshold.
>
> To wrap up: GRU with multi-site context is our best model at both
> horizons. A trivial spatial mean already helps -- the next natural
> step is a wind-direction-aware aggregator, because upwind stations
> matter more than downwind ones.
>
> Thanks -- happy to take questions.

---

## Timing notes

| Slide | Target | Cumulative |
|-------|--------|-----------:|
| 1     | 60s    | 1:00 |
| 2     | 60s    | 2:00 |
| 3     | 60s    | 3:00 |

If you're running short, the safest cuts (in order):
1. The *italicized* sentences (marked above)
2. The specific number on slide 2 ("GRU multi-site reaches 28.75")
   → just say "GRU multi-site barely edges out gradient boosting"
3. The preprocessing aside on slide 1

If you're running long, you can drop one italicized line per slide on
the fly without losing the argument.

---

## Delivery tips

1. **Pace** - 1 minute per slide feels generous, but practice once with
   a stopwatch. Aim for 2:50 so you have buffer.
2. **Don't read the slide** - point at it (the MAE chart on slide 2,
   the timeseries on slide 3) and *narrate* what's there. The audience
   can read the bullets themselves.
3. **The four numbers to drill**:
   - **9.41** (best 1-hour MAE: Transformer multi-site)
   - **28.75** (best 6-hour MAE: GRU multi-site)
   - **95%** (spike-recall at h=1)
   - **2.55** (GRU MAE drop from multi-site at h=6)
4. **End strong**: finish "Thanks -- happy to take questions" *before*
   clicking past. Make eye contact.
5. **If you stumble on a number**, just say "around nine point four" or
   "around twenty-nine" -- nobody is checking decimals in real time.

---

## Mapping slide content to talking points

| Slide | Visual centerpiece | Key things to say |
|-------|--------------------|-------------------|
| 1 | Text-only, table of splits | Why this problem; two questions; what's in a window; seven models |
| 2 | MAE bar chart | h=1 numbers, h=6 numbers, sequence + multi-site wins |
| 3 | Delta table + timeseries plot | Multi-site helps every model, helps more at h=6; show the spike-tracking; future work |
