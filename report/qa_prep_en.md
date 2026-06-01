# Q&A Prep - 20 Likely Questions

Each item has:
- **Q**: the question (English)
- **A**: a tight English answer (what to say out loud, ~20-40 sec each)
- **中文重點**: the same answer in Chinese for your own preparation

When answering during Q&A, repeat the question first ("That's a great question
about ...") -- it buys you two seconds of thinking time and confirms you heard
the question right.

---

## Dataset & problem setup

### Q1. Why did you pick the Beijing Multi-Site dataset specifically?

**A.** Three reasons. First, it's a real-world dataset, not synthetic --
the haze episodes in Beijing are exactly the kind of event we want to
forecast. Second, it has *twelve synchronized stations* in one city,
which lets us test multi-site context cleanly without dealing with
heterogeneous sensors. Third, it covers four years of hourly data, so
we have enough history for time-based train/val/test splits with full
seasons in each.

**中文重點.** 真實資料、十二站同步、四年小時級資料夠長。

---

### Q2. Why a 24-hour input window? Why not 48 or 12?

**A.** Twenty-four hours captures one full daily cycle -- morning rush,
evening commute, overnight chemistry. Shorter windows like 12 hours
miss the daily seasonality; longer windows like 48 hours mostly add
redundant information for PM2.5 specifically, since the autocorrelation
drops off after about a day. We didn't sweep this, but 24 hours is the
standard choice in the air-quality literature for the same reason.

**中文重點.** 24 小時剛好涵蓋一個完整日週期；更短會漏掉日變化、更長對 PM2.5 來說
邊際效益不大；文獻標準做法。

---

### Q3. Why predict at h=1 and h=6 specifically? Why not h=24?

**A.** Those two horizons map to two different use cases. One hour is
the lower bound for any meaningful early warning -- if you can flag a
spike one hour out, that's still actionable for individuals. Six hours
is around the lead time needed for a public-health advisory or for a
school to plan an outdoor-activity decision. Twenty-four hours is
interesting too but the prediction task becomes much closer to a
forecasting-from-climatology problem; the signal in recent hours
matters less.

**中文重點.** 1h 是個人預警下限、6h 是政府機關可反應的時長。24h 已經接近氣候預報，
歷史小時的訊號變弱。

---

### Q4. How did you handle missing data? Did you check that it doesn't bias results?

**A.** Pollutant missingness is 1.5 to 4.9 percent and weather is below
1 percent. We forward-fill then backward-fill within each station --
because missing values are mostly isolated single hours, this is fine.
We spot-checked that the filled hours aren't concentrated in any
particular season or station. Anything still missing -- very rare --
gets set to zero in standardized space. We also use the *training*
distribution to standardize, so the fill choice doesn't leak into
the val or test split.

**中文重點.** Forward-fill + backward-fill，缺失多為孤立單小時。標準化只用 train，
所以填充不會洩漏。

---

### Q5. Why a time-based train/val/test split instead of random?

**A.** Two reasons. First, this is a time series forecasting task -- in
deployment we forecast the future, so the evaluation has to respect
that. A random split lets the model see "future" hours during training,
which is leakage. Second, the train/val/test boundaries are set so each
split contains at least one full season -- otherwise the test set could
disagree with train just because it falls in winter haze season.

**中文重點.** 時序預測必須時間切，避免洩漏；切分位置確保每個 split 包含完整季節。

---

## Preprocessing

### Q6. Why clip standardized features to ±8 sigma? Why not 3 or 6?

**A.** Pure necessity, actually. Before clipping, the rainfall column
had values around 90 sigma -- most hours have zero rain, but occasional
downpours of a few mm are huge in standardized space. Those extreme
inputs caused NaN losses in the GRU on the very first batches. Three
sigma was too tight -- it would clip real haze peaks as well. Eight
sigma is wide enough to keep real outliers as "very large but finite,"
while preventing exploding gradients. We didn't do a formal sweep but
the training was stable from the first try once we set it.

**中文重點.** 不裁剪會 NaN（降雨欄位 90σ）；±3 太緊會壓真實 spike；±8 是經驗值，
從第一次就穩定。

---

### Q7. Why encode wind direction as (sin, cos) instead of just the angle?

**A.** Because angle is *cyclic*. If you feed in the raw angle in degrees,
the model sees 359 degrees and 1 degree as far apart -- they're really
two degrees apart. The (sin, cos) parameterization preserves the
cyclic geometry: 359 and 1 map to nearly identical points on the unit
circle. We use the same trick for hour-of-day, day-of-week, and month,
all of which are cyclic.

**中文重點.** 角度是 cyclic 的，359° 和 1° 其實很近。(sin, cos) 編碼保留這個性質。
時間特徵也一樣。

---

## Models & training

### Q8. Why these seven models and not, say, Prophet or N-BEATS or a graph neural network?

**A.** We wanted a clean, *interpretable* comparison: one no-learning
baseline, three feature-based baselines that span linear-through-
nonlinear, and three sequence backbones that span recurrent and
attention. This covers the architectural axes that are actually
*claimed* to matter for this task in the literature. We deliberately
did *not* implement a graph neural network because the goal of this
project was to isolate the effect of a simple spatial aggregator --
adding a graph model would conflate "graph structure" with "spatial
information."

**中文重點.** 想做乾淨的對照：無學習、特徵式（線性到非線性）、序列（recurrent + attention）。
不做 GNN 是為了把空間資訊的效果跟模型結構的效果分開。

---

### Q9. Why hidden size 128 for all three sequence models? Did you tune that?

**A.** We picked 128 as a reasonable size that fits comfortably on MPS,
keeps training under a minute per epoch, and gives the model enough
capacity to overfit if it wants to. We didn't do a formal capacity
sweep -- the bottleneck we observed wasn't capacity, it was longer-
horizon tuning (LR and patience). Sweeping hidden size would matter
more if we were chasing the absolute best number, which we weren't.

**中文重點.** 128 是個合理的「夠大但訓練快」的選擇，沒做正式 sweep；我們的瓶頸不在
capacity，是 LR/patience。

---

### Q10. Why an 8-dimensional station embedding?

**A.** Twelve stations, eight dimensions -- the embedding can in
principle separate every station with redundancy. We tried four and
sixteen briefly; four was slightly worse, sixteen was no better. Eight
is a sweet spot. Importantly, this is *learned* -- nearby stations
end up with similar embeddings, which is useful information the model
gets for free.

**中文重點.** 12 站、8 維足以分離且有冗餘；4 維略差、16 維無提升；nearby station 的
embedding 會自己變相近。

---

### Q11. Why smooth-L1 loss instead of MSE or MAE?

**A.** PM2.5 has *heavy outliers* -- a haze peak can be five-times
the daily mean. MSE squares those errors and lets the gradient blow
up in early training. Pure MAE has a discontinuity at zero that hurts
optimization. Smooth-L1 -- it's MAE for large errors and MSE for small
ones -- gets you the best of both: bounded gradients on outliers,
smooth optimization near the solution. Training was visibly more
stable than with MSE.

**中文重點.** PM2.5 outlier 多，MSE 梯度會炸；MAE 在 0 不可微會影響優化。Smooth-L1
取中道：大誤差像 MAE、小誤差像 MSE。

---

### Q12. Why early stop with patience 4? Why max 15 epochs?

**A.** The training curves showed most of the improvement happens in
the first three to five epochs. Patience of four means we let the
model try four more epochs after it stops improving on val MAE before
we give up -- this caught some legitimate late improvements. Fifteen
epochs is a generous hard cap; almost every run early-stopped before
hitting it. For the six-hour horizon, in hindsight we should have
loosened patience -- the deep models early-stopped too aggressively
there, which is the main thing we'd retune.

**中文重點.** 訓練曲線顯示大部分提升在前 3-5 epoch；patience=4 平衡「給機會」跟
「避免浪費」；15 是上限。h=6 上 patience 應該放寬，這是後見之明。

---

## Results & interpretation

### Q13. Why does the GRU outperform the LSTM in your results?

**A.** Two factors. First, on MPS specifically, the LSTM training is
less stable -- we saw more non-finite losses in the first epochs.
Second, the LSTM has more parameters and overfits a bit faster on this
amount of data. At a longer horizon, LSTM also early-stops too
aggressively in our default setup. With CUDA and a longer patience,
LSTM would likely close most of the gap, but on this dataset GRU's
simpler gating happened to work better out of the box.

**中文重點.** MPS 上 LSTM 比較不穩、參數多易過擬；h=6 早停太早。換 CUDA、調 patience
應該會接近。

---

### Q14. Why exactly does multi-site context help? It's just the mean -- shouldn't the model already infer that from the target station's own trends?

**A.** Local trends only tell you about *the air mass at this one
location*. The other eleven stations sample air masses across the city
-- if a haze plume is approaching from the southwest, three southwest
stations have already seen elevated PM2.5 a few hours before it
reaches the target station. Even a *mean* of the other eleven captures
this "front" effect partially, because elevated stations pull the mean
up. A wind-direction-aware aggregator would do this better, but the
fact that even the unweighted mean helps tells us the spatial signal
is real and not noise.

**中文重點.** 自己過去只反映自己這團空氣；其他站採樣到整個城市的空氣，能捕捉「污染鋒面」
逼近的訊號。Mean 已經部分捕捉到，wind-aware 應該更好。

---

### Q15. Why is spike-recall such a focus? Wouldn't a single MAE number be cleaner?

**A.** MAE is the average error across all hours, but most hours have
low PM2.5 and are easy. The hours that *matter* operationally are the
ones above the unhealthy threshold, and those are also the *hardest*
to predict because they're rare and have heavy tails. A model with
great MAE that underpredicts every spike would be useless for an
advisory system. Spike-recall is the operational metric -- it's the
fraction of unhealthy hours we'd actually catch, which is what a
public-health office would care about.

**中文重點.** MAE 是平均，但好處幾乎全來自簡單時段。我們真正在乎的是「會發警報的時段
有沒有抓到」，那就是 spike-recall。

---

### Q16. Why is h=6 so much harder than h=1? Is that just because the target is further away?

**A.** Partly that, but it's more subtle. At h=1, PM2.5 is overwhelmingly
autocorrelated -- whatever the value is now will be close to that in
an hour, regardless of weather. So persistence already gets R-squared
of 0.95. At h=6, the autocorrelation has decayed enough that you
actually have to *model* the dynamics: wind transport, chemistry,
boundary-layer height. That's why persistence collapses to 0.56 and
even the best model only reaches 0.7. The information you need is
present in the data, but it requires a much richer model.

**中文重點.** h=1 PM2.5 自相關超強、persistence 都 0.95；h=6 自相關衰減，需要真的
建模動力學（風、化學、邊界層）。資訊在，但需要更豐富的模型。

---

### Q17. Did you check for data leakage between train/val/test? How can you be sure?

**A.** Yes. Three things: first, the split is strictly chronological --
train ends 2015-12-31, val ends 2016-06-30, test starts after that. No
overlap. Second, scaler statistics are fit on train only and applied to
val and test, so no statistical leakage. Third -- this one bit us
once -- in the multi-site setting, we explicitly *exclude* the target
station from the channel-wise mean of the other stations. Otherwise
the target station's PM2.5 at time t would appear in its own input
at time t, which is a leakage of the label itself.

**中文重點.** 時間切無重疊；scaler 只用 train；multi-site 把目標站從 mean 排除（不然
標籤洩漏）。

---

## Limitations & future work

### Q18. The multi-site mean is very simple. Why not weight by distance or wind direction?

**A.** Our project goal was to isolate the *value of spatial information*
from the *value of model sophistication*. If we had used a weighted
aggregator, any improvement could be attributed either to the spatial
data or to the aggregator's structure. Showing that even the unweighted
mean helps tells us the spatial signal is real, and gives a clean
baseline for future weighted approaches. Wind-aware weighting --
upwind stations matter more than downwind -- is the most natural next
step, and we expect it to give a much larger gain.

**中文重點.** 故意做最簡單的聚合，這樣改善可以歸因到「空間資訊本身」而不是聚合方式。
Wind-aware 是 future work。

---

### Q19. Random forests and gradient boosting were subsampled. Doesn't that make the comparison unfair?

**A.** Yes, that's a real limitation. RF saw 50k of 298k training rows,
GBM saw 80k. The deep models saw everything. We capped the tree models
for tractability -- a full random forest on 298k rows with 468 features
took about an hour on CPU. A fairer comparison would train all models
on the full set, which is a worthwhile follow-up. That said, in
practice tree models often *don't* improve linearly with data past a
point, so the gap might not be as large as the subsample ratio
suggests.

**中文重點.** 確實是 limitation；樹模型採樣（RF 50k、GBM 80k）vs DL 看全部。完整訓練
應該做但很慢。實務上樹模型超過某個資料量邊際提升有限。

---

### Q20. If you had two more weeks, what would you do next?

**A.** Three things, in order of expected payoff. First, a wind-direction-
aware spatial aggregator instead of the unweighted mean -- this is the
biggest open lever and matches the physics. Second, a proper learning-
rate and patience sweep specifically for the six-hour horizon, where
our defaults early-stop too aggressively. Third, multi-task forecasting
-- predict all six pollutants jointly -- which should let the model
learn a richer shared representation. Beyond that, a graph neural
network with attention over stations would be the natural step toward
state-of-the-art, but we'd want to first establish whether the gains
come from the spatial structure or just from the larger model.

**中文重點.** 三件事：(1) 風向感知的空間聚合（最有 payoff）、(2) h=6 單獨調 LR/patience、
(3) Multi-task（共享 representation）。再進階就是 GNN，但要先分離出空間結構的效果。

---

## General fallbacks

If you genuinely don't know an answer:

> "That's a great question -- we didn't explore that specifically in this
> project, but [related thing we did look at]. I'd expect [your best
> guess based on intuition]. It's a good direction for future work."

Never make up a number. If asked for one you don't remember:

> "Off the top of my head I don't remember the exact number, but it's in
> the report -- happy to follow up after."
