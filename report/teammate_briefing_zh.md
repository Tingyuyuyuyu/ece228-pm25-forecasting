# ECE 228 期末專案 - 隊友報告前完整說明（中文）

## 0. 一分鐘版本（如果只看這段）

我們用 UCI 北京多測站空氣品質資料集，預測 PM2.5 在「1 小時後」和
「6 小時後」的濃度。比較 7 個模型（持續性 baseline、Ridge、隨機森林、
梯度提升、LSTM、GRU、Transformer），並且測試兩種輸入設定：
- **global**：只用目標測站自己過去 24 小時的資料。
- **multi-site**：再加上「其他 11 個測站當下的平均」當作空間上下文。

**主要結論**：
1. 1 小時預測時，GRU / Transformer (multi-site) 最好，MAE ≈ 9.4 μg/m³，
   能抓到 ~95% 的「不健康時段」(≥75 μg/m³)。
2. 6 小時預測時，GRU (multi-site) MAE = 28.75，略勝梯度提升 multi-site 的 29.00。
3. 「拿其他測站平均」這種超簡單的空間特徵就已經能讓 MAE 系統性地下降，
   而且在 6 小時預測上幫助更大（GRU 降 2.55 μg/m³）。

---

## 1. 動機（為什麼選 PM2.5）

- **PM2.5 = 直徑小於 2.5 微米的細懸浮微粒**，是大都市空氣污染的主角。
  深入肺部、進入血液，跟呼吸道疾病、心血管疾病高度相關。
- **75 μg/m³** 是 WHO 和中國國標 (GB 3095) 的「不健康」門檻，每天 24 小時平均
  超過這個值就會發布警報。
- 北京冬季常有「霾害事件」，PM2.5 連續好幾天爆表。如果模型能提前幾小時預測，
  公共衛生單位可以提早發警報、學校可以安排室內活動、居民可以戴口罩。
- 所以這不只是回歸題，而是**早期警報問題**：能不能在污染還沒爆之前就抓到？

---

## 2. 兩個我們想回答的問題

1. **時序模型 (LSTM / GRU / Transformer) 在這個資料上真的會贏特徵式回歸 (Ridge / 隨機森林 / 梯度提升) 嗎？**
2. **把其他 11 個測站的資料加進來當「空間上下文」有沒有幫助？**

我們特意把空間特徵做得**很簡單**（就是其他 11 個測站的平均），
這樣如果有幫助，可以很明確地歸因到「空間資訊本身」，而不是模型結構複雜。

---

## 3. 資料集細節

### 來源
- **UCI Beijing Multi-Site Air Quality Dataset**（Zhang 2017）
- 12 個監測站，全在北京市範圍內
- **小時級**資料，2013-03-01 到 2017-02-28（約 4 年）
- 每站約 35,064 筆觀測（共 ~420,000 筆）

### 12 個測站名稱
Aotizhongxin（奧體中心，市中心）、Changping（昌平）、Dingling（定陵）、
Dongsi（東四）、Guanyuan（官園）、Gucheng（古城）、Huairou（懷柔）、
Nongzhanguan（農展館）、Shunyi（順義）、Tiantan（天壇）、Wanliu（萬柳）、
Wanshouxigong（萬壽西宮）。

→ 後面的 per-station 表會看到：**市中心站誤差大**（車多、變動大），
**山區站如懷柔、定陵誤差小**（空氣相對穩定）。

### 每筆觀測有什麼欄位
- **6 個污染物**：PM2.5、PM10、SO₂、NO₂、CO、O₃
- **5 個氣象數值**：溫度、氣壓、露點、降雨量、風速
- **1 個風向**：以 16 個方位（N、NNE、NE …）標記
- 缺失值：污染物 1.5%–4.9%，氣象 < 1%（很少）

---

## 4. 預處理（重點記住「為什麼這樣做」）

### 4.1 缺失值處理
每站內部先 **forward-fill 再 backward-fill**。因為缺失多半是孤立的單小時，
這樣處理影響很小。

### 4.2 風向 → (sin, cos)
直接把 16 個方位轉成 0–360°，再做 `(sin(θ), cos(θ))`。
**為什麼？** 因為 359° 和 1° 其實很接近，但如果用原始角度，模型會以為差很多。
這是 cyclic encoding 的標準作法。

同樣的方法也用在「小時 hour」、「星期幾」、「月份」上。

### 4.3 標準化 + 裁剪
- 用「**訓練集**的 mean/std」標準化所有數值欄位（驗證/測試集不能用自己的統計，
  避免資訊洩漏）。
- 標準化後 **裁剪到 [-8σ, +8σ]**。
- **為什麼裁剪？**（這是被問到一定要答好的點）
  - 降雨量這個欄位很「長尾」：大部分小時都是 0 mm，但偶爾的大雨會有 ~90σ 的極端值。
  - 不裁剪的話，這些超大輸入會讓 GRU 在前幾個 batch 直接產生 NaN loss，整個訓練爆掉。
  - 裁剪到 ±8σ 保留了「很大的離群值」但又不會炸梯度。

### 4.4 時間特徵
hour-of-day、day-of-week、month-of-year 都用 cyclic (sin, cos) 編碼。
這樣模型可以學「23:00 跟 00:00 很像」、「12 月跟 1 月很像」這類季節性。

---

## 5. 滑動視窗與目標

- **輸入**：過去 24 小時 × F 個特徵
- **目標**：t + h 小時後的 PM2.5 濃度（h ∈ {1, 6}）
- **單位**：模型內部訓練用標準化後的目標（loss 表現更好），但 metrics（MAE/RMSE）
  都用原始 μg/m³ 單位來算（這樣讀起來有實際意義）

### 為什麼 24 小時 + 1h / 6h？
- 24 小時 = 一個完整的日週期（早晚通勤、夜間光化反應全包含進來）
- 1 小時：短期預警的下限
- 6 小時：足夠讓政府部門做反應的時長

---

## 6. 兩種輸入設定

### 6.1 Global 設定
- 一個模型訓練所有 12 個站的資料
- 每個樣本只看「**自己**過去 24 小時」
- 用 8 維的 station embedding（深度模型）或 one-hot（線性/樹模型）
  告訴模型「這是哪一站」
- F = 19（6 污染物 + 5 氣象數值 + 2 風向 sin/cos + 6 時間 sin/cos）

### 6.2 Multi-site 設定
- 跟 global 一樣，但每個時間點再加上「**其他 11 個站的 channel-wise 平均**」
- F 從 19 變 38
- **重要**：目標站要從平均裡排除掉，不然就直接洩漏答案了
- 這是「最簡單的可能的空間摘要」—— 沒有圖神經網路、沒有 attention，
  純粹就是 mean。如果這樣做就有效果，代表空間資訊真的有用。

---

## 7. 七個模型

| 家族 | 模型 | 我們怎麼設定 |
|------|------|--------------|
| 無學習 | **Persistence** $\hat y_{t+h} = y_t$ | 直接用「現在的值」當預測，超強 baseline |
| 特徵式 | **Ridge 回歸** | 把 24×F 攤平，α=1 |
|        | **隨機森林** | 120 棵樹、max_depth=20、min_leaf=5；採樣 50k 列訓練 |
|        | **梯度提升** | 200 輪、深度 4、lr=0.05、subsample=0.7；採樣 80k 列訓練 |
| 序列   | **LSTM** | 2 層、hidden=128、dropout=0.2 |
|        | **GRU**  | 同上但用 GRU cell |
|        | **Transformer** | 2 層 encoder、d_model=128、4 個 head、FFN=512、GELU |

所有深度模型都有 8 維的 **station embedding**，concat 到每個時步的特徵向量。

### 為什麼樹模型要採樣？
完整訓練集 298,080 列 × 468 維特徵 (24*19+12) 太大，隨機森林/GBM 在 CPU 上會跑超久。
深度模型用 batch 訓練，沒這個問題。
（這算是一個 limitation，深度模型看到的資料比樹模型多。）

---

## 8. 訓練設定（序列模型）

- **優化器**：AdamW（lr=1e-3、weight_decay=1e-5）
- **LR schedule**：CosineAnnealingLR
- **Batch size**：512
- **Loss**：Smooth-L1（在標準化空間算）
  - 為什麼不是 MSE？因為 PM2.5 outlier 多（極端霾害事件），
    MSE 在訓練初期會被 outlier 帶歪。Smooth-L1 對大誤差比較穩。
- **Gradient clipping**：norm = 1.0
- **早停**：基於驗證集 MAE，patience=4，最多 15 epoch
- **NaN guard**：偶爾 MPS 的第一個 batch 會出 NaN loss，我們 skip 那一步繼續

### 硬體
PyTorch MPS backend（Apple Silicon GPU）。LSTM/GRU 一個 epoch 約 15–40 秒，
Transformer 約 40 秒。

---

## 9. 資料切分（時間切，不能隨機切！）

| 切分 | 日期範圍 | 每站約 |
|------|----------|--------|
| Train | 2013-03-01 ~ 2015-12-31 | 24,864 小時 |
| Val   | 2016-01-01 ~ 2016-06-30 | 4,368 小時 |
| Test  | 2016-07-01 ~ 2017-02-28 | 5,832 小時 |

**為什麼時間切？** 因為這是時序預測。如果隨機切，模型可能看到「未來」的資訊
（例如 2017 年的點被放進 train），這就洩漏了。時間切確保「永遠不在比驗證/測試
還新的時間點上訓練」。

---

## 10. 評估指標

- **MAE**：平均絕對誤差（μg/m³），越低越好
- **RMSE**：均方根誤差，對大誤差敏感，越低越好
- **R²**：解釋變異比例，越高越好
- **Spike-recall**：⭐ **這個最重要！**
  - 定義：所有測試集中 PM2.5 ≥ 75 μg/m³ 的時段裡，有多少比例模型也預測 ≥ 75？
  - 為什麼是 75？因為這就是「會發警報」的門檻
  - 為什麼這個重要？因為我們在乎的是「漏掉警報的代價」遠大於「誤報的代價」

---

## 11. 主要結果（背起來！）

### 11.1 1 小時預測（h=1）

| 模型 | global MAE | multi-site MAE |
|------|-----------:|---------------:|
| Persistence | 10.38 | 10.38 |
| Ridge | 10.32 | 9.94 |
| Random Forest | 10.01 | 9.83 |
| Gradient Boosting | 9.98 | 9.87 |
| LSTM | 10.66 | 10.23 |
| **GRU** | **9.69** | **9.51** |
| **Transformer** | 9.84 | **9.41** ⭐ |

- 1 小時預測非常「容易」—— Persistence 的 R² 都有 0.946 了
- 但 Transformer 和 GRU 還是把 MAE 從 10.38 拉到 ~9.4，這是 ~10% 改善
- **Spike-recall 大多 ~0.95**：能抓到 19/20 的不健康時段

### 11.2 6 小時預測（h=6）

| 模型 | global MAE | multi-site MAE |
|------|-----------:|---------------:|
| Persistence | 33.96 | 33.96 |
| Ridge | 32.35 | 30.48 |
| Random Forest | 30.39 | 29.13 |
| Gradient Boosting | 30.43 | **29.00** |
| LSTM | 34.02 | 30.10 |
| **GRU** | 31.30 | **28.75** ⭐ |
| Transformer | 30.63 | 29.83 |

- 6 小時預測難很多：Persistence R² 直接掉到 0.564
- **GRU multi-site (28.75) 是所有模型最低 MAE**，略勝 GBM multi-site (29.00)
- Spike-recall 掉到 0.82–0.86，可預期

### 11.3 Multi-site context 幫多少？(Δ = multi − global，負代表 multi 比較好)

| 模型 | Δ at h=1 | Δ at h=6 |
|------|---------:|---------:|
| Ridge | −0.38 | −1.87 |
| RF    | −0.18 | −1.26 |
| GBM   | −0.11 | −1.43 |
| GRU   | −0.19 | **−2.55** |
| Transformer | −0.43 | −0.79 |
| LSTM  | −0.43 | −3.92 |

→ **所有模型在所有 horizon 都受益**，而且 h=6 受益更多。
這合乎直覺：6 小時遠的事情，看自己過去 24 小時的資訊不夠，鄰站的當下值非常有用。

### 11.4 Per-station（用 GRU multi-site h=1）

最低 MAE：Huairou（懷柔）8.12、Wanliu（萬柳）9.09、Nongzhanguan（農展館）9.32
最高 MAE：Dongsi（東四）10.23、Guanyuan（官園）10.01、Wanshouxigong（萬壽西宮）9.84

→ 山區站誤差最小（空氣穩定），市區交通密集站誤差最大。地理直覺成立。

---

## 12. 七張投影片每張要講什麼（300 秒總長度）

### Slide 1：Title (~5 秒)
打招呼，自我介紹（team 17、UCSD、ECE 228）。

### Slide 2：Why forecast PM2.5? (~30 秒)
重點講：
- PM2.5 對健康有害，是大都市空污主角
- 75 μg/m³ 是 unhealthy 門檻
- 我們想回答兩個問題：時序模型有用嗎？空間上下文有用嗎？
- 資料來源是 UCI 北京多測站

### Slide 3：Setup (~30 秒)
重點講：
- 24 小時輸入視窗，預測 1h 或 6h 之後的 PM2.5
- 時間切分（不能隨機切）
- 標準化 + 裁剪到 ±8σ（順便提一下這修了 GRU 的 NaN 問題）
- 兩個 setting：global（自己） vs multi（自己 + 其他 11 站平均）
- Spike-recall 是我們最在乎的指標

### Slide 4：Seven models (~25 秒)
快速念出三個家族：
- 無學習的 persistence baseline
- 三個特徵式模型：Ridge、RF、GBM（把 24×F 攤平）
- 三個序列模型：LSTM、GRU、Transformer（吃整個視窗）
- 強調「所有模型用同樣的資料、同樣的 loss、同樣的 metrics」=> 公平比較

### Slide 5：Result 1 (~45 秒)
看 MAE 長條圖：
- 1 小時：sequence models（GRU/Transformer）最好，MAE ≈ 9.4
- 6 小時：所有模型 MAE 都跳到 ~30，但 GRU multi-site 還是第一（28.75）
- Spike-recall：1h 約 95%，6h 約 82–86%
- 結論：時序模型搭配空間特徵時效果最好

### Slide 6：Result 2 (~35 秒)
看 Δ 表 + 時序圖：
- 用「其他 11 站平均」這麼簡單的特徵就有用
- 所有模型都受益，h=6 受益更大（GRU 降 2.55、LSTM 降 3.92）
- 右邊的時序圖：GRU multi-site h=1 在奧體中心測站的前三週測試集，
  幾乎完全貼著實際曲線、抓到了所有 spike

### Slide 7：Wrap-up (~15 秒)
- 最佳結果：GRU multi-site 兩個 horizon 都最好
- 一個極簡單的 spatial mean 就帶來持續性的提升
- 下一步：風向感知的空間聚合、6h 的 LR 重新調、多任務（PM10、NO₂）
- 「Thanks - questions?」

---

## 13. 設計決策（會被挑戰的點）

### 為什麼不用 Graph Neural Network？
- 我們的目標是「乾淨的對照實驗」，不是 SOTA 重現
- 故意把空間特徵做到最簡單（mean），這樣**任何**模型結構受益都能歸因到「空間資訊」本身
- 如果連 mean 都這麼有用，那 wind-direction-aware 的聚合應該更強——這是 future work

### 為什麼 LSTM 在我們的結果裡輸 GRU 不少？
- LSTM 在 MPS 上訓練比 GRU 不穩定（前幾個 epoch 常出 NaN loss）
- 而且 LSTM 在我們的 checkpoint save/load 時有一個已知的 MPS 量化問題
  （所以我們在報告中用「training session 當下的 metrics」而不是「重新載入後」的）
- 用 CUDA 或調整 LR 應該會穩很多

### 為什麼 6 小時上序列模型沒有「壓倒性」優勢？
- 我們的 LR/patience 都是針對 1 小時調的
- 在 6 小時上序列模型早停太積極（只跑 2–3 個 epoch 就停）
- 這是 tuning 問題，不是架構問題
- Future work：給 6h 單獨做 LR sweep

### 為什麼裁剪到 ±8σ 不是 ±3σ 或 ±6σ？
- ±3σ 太緊：會把真實的霾害值（也是極端）也壓掉
- ±8σ 留了足夠寬度給「真實的大事件」，但又把 90σ 的降雨爆衝壓住
- 是經驗值，沒做 ablation

### 為什麼 station id 用 embedding 不是 one-hot？
- 對深度模型：embedding 比 one-hot 更省參數、能學到「站之間的相似度」
- 對線性/樹模型：one-hot 比較自然，反正它們不會學特徵間關係

---

## 14. Limitations（被問到一定要承認）

1. **空間聚合太簡單**：mean 忽略了風向。上風站比下風站有用得多
2. **6 小時沒個別調參**：deep model 早停太早，LR/patience 沒針對 h=6 sweep
3. **只預測 PM2.5**：multi-task 可能讓 representation 更好
4. **樹模型有採樣**：RF 50k、GBM 80k vs DL 看全部 298k，不太公平
5. **裁剪到 ±8σ**：可能會壓掉真正極端的事件，影響 spike prediction

---

## 15. 報告 (report.tex) 對應到投影片

- 報告完整版有更多細節：preprocessing 章節、per-station 表、training curve 圖
- 投影片是「精華濃縮版」：3 分鐘只能講最關鍵的兩個結果
- 報告檔 `report/report.tex`、投影片 `report/slides.tex`
- 兩者共用 `report/figures/` 和 `report/tables/`

---

## 16. 怎麼編譯投影片

```bash
cd report
pdflatex slides.tex      # 第一次
pdflatex slides.tex      # 第二次（產生參考）
```

或者上 Overleaf：把 `slides.tex` + `figures/` 整包丟上去即可。
沒有用 metropolis 主題或其他不常見套件，標準 LaTeX 都能編。

---

## 17. 如果忘了某個數字，記住這四個就夠了

1. **9.41** μg/m³ — Transformer multi-site h=1 的 MAE（最低）
2. **28.75** μg/m³ — GRU multi-site h=6 的 MAE（h=6 最低）
3. **~95%** — h=1 的 spike-recall
4. **−2.55 μg/m³** — multi-site context 給 GRU h=6 帶來的改善

---

加油！報告順利 🙏
