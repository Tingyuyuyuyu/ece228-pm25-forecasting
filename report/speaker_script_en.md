# Speaker Script - 3-slide presentation (target ~2 min, max 3 min)

Hard cap: **3 minutes** -- aim for **2:00** so you have a buffer for a
slow start or a stumble. Each slide ~40 seconds at a comfortable
~150 words/min pace.

Words in `*italics*` are optional padding -- add them if you finish a
slide early, drop them if you're behind.

---

## Slide 1 - Problem + setup (~40 sec, ~110 words)

> Hi, we're Team 17 -- Ting-Yu and Yun-Chen. Our project forecasts
> PM2.5 in Beijing, one and six hours ahead.
>
> PM2.5 is fine particulate matter -- seventy-five micrograms per
> cubic meter is the WHO and Chinese national-standard ``unhealthy''
> threshold, the level cities actually act on.
>
> We asked two questions: first, do sequence models beat feature-based
> regressors here, and second, does adding context from neighbor
> stations help?
>
> The data is UCI's twelve-station, four-year, hourly Beijing dataset.
> We use a twenty-four-hour input window and a strict time-based split.
> We compare seven models -- persistence, ridge, random forest,
> gradient boosting, LSTM, GRU, and a small Transformer -- in two input
> settings: \alert{global}, which sees only the target station's own
> history, and \alert{multi-site}, which adds the channel-wise mean of
> the other eleven stations.

*[Click to slide 2]*

---

## Slide 2 - Headline results (~40 sec, ~100 words)

> Here's the headline. This chart shows test MAE for every model and
> setting -- green and red bars are multi-site, blue and orange are
> global.
>
> At one hour ahead, persistence is already strong -- R-squared
> around point-nine-five. But sequence models with multi-site context
> push MAE down to about nine-point-four, and we recall around
> ninety-five percent of unhealthy hours.
>
> At six hours ahead, persistence collapses and the gap between
> methods opens up. GRU multi-site reaches MAE twenty-eight-point-
> seven-five, just edging out gradient boosting at twenty-nine.
>
> *The take-away*: sequence models win most when paired with the
> multi-site spatial context.

*[Click to slide 3]*

---

## Slide 3 - Multi-site analysis + wrap-up (~40 sec, ~100 words)

> For multi-site, we just add the channel-wise mean of the other
> eleven stations -- no graph, no attention, the simplest possible
> spatial summary.
>
> The MAE delta is negative for every model, with the largest gains
> at six hours -- GRU drops two-point-five-five micrograms per cubic
> meter.
>
> The plot on the right shows GRU multi-site predictions tracking
> actual PM2.5 over three weeks at Aotizhongxin -- it catches the
> haze spikes that cross the unhealthy threshold.
>
> Next steps: wind-direction-aware aggregation, *longer-horizon
> tuning, and multi-task forecasting across pollutants*.
>
> Thanks -- happy to take questions.

---

## Timing

| Slide | Target | Cumulative |
|-------|--------|-----------:|
| 1     | ~40s   | 0:40 |
| 2     | ~40s   | 1:20 |
| 3     | ~40s   | 2:00 |
| (buffer for clicks, pauses) | ~30s | 2:30 |
| **Hard cap** | | **3:00** |

If you're running over after slide 2, drop the italicized lines on
slide 3 -- "longer-horizon tuning, and multi-task forecasting across
pollutants" comes off cleanly.

---

## The four numbers to drill

1. **9.4** -- best 1-hour MAE (round; Transformer multi-site = 9.41,
   GRU multi-site = 9.51)
2. **28.75** -- best 6-hour MAE (GRU multi-site)
3. **95%** -- spike-recall at h=1
4. **2.55** -- GRU MAE drop from multi-site at h=6

---

## Delivery tips

1. **Don't read the slide.** Point at the chart (slide 2) or the
   timeseries (slide 3) and narrate what's there. The audience reads
   the bullets themselves.
2. **Pause** half a second between slides -- looks confident.
3. **If you stumble on a number**, just say "around nine point four"
   or "around twenty-nine." Nobody is checking decimals in real time.
4. **End strong**: say "Thanks -- happy to take questions" *before*
   clicking past. Make eye contact.
5. **Practice once with a stopwatch.** Anything between 2:00 and 2:45
   is great; under 2:00 means slow down a bit, over 2:45 means start
   cutting italics.
