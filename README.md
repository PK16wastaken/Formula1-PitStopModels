# Formula1-PitStopModels

An end-to-end Machine Learning and Data Engineering framework designed to predict real-time Formula 1 pit stop strategies. By analyzing high-frequency race telemetry, tire degradation dynamics, and trackside traffic windows, this repository models optimal pit stop timing for unseen race campaigns.

---

## 📊 Model Architecture & Predictive Analytics

### 1. Random Forest Ensemble Baseline (`RandomForestClassifier`)

The primary baseline model utilizes a **100-tree Random Forest Classifier** trained on multi-year Formula 1 historical race telemetry (2022–2024) and evaluated on unseen deployment seasons (2025).

```
                          [ Raw Telemetry Data ]
                                    │
                                    ▼
                 [ Feature Engineering & Traffic Engine ]
                                    │
                                    ▼
       ┌────────────────────────────┴────────────────────────────┐
       │                 Chronological Data Split                │
       │   Historical Training (2022–2024) │ Deployment (2025)   │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
                      [ 100-Tree Random Forest ]
                                    │
                                    ▼
                  [ F1-Optimal Threshold Optimization ]
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       │                                                         │
       ▼                                                         ▼
[ MATLAB Interactive App ]                            [ Python Backtest Suite ]
  - Strategy Axes & Overlay                             - Feature Importance
  - Real-time Strategy Trigger                          - Confusion Profile Matrix
```

#### A. Engineered Feature Pipeline
The model ingests a hybrid feature space combining raw lap telemetry with higher-order physics and track position metrics:

* **`TyreLife`**: Number of completed laps on the current set of tires.
* **`Cumulative_Degradation`**: Total calculated tire performance loss over a stint.
* **`Degradation_Velocity`**: First-order differential ($\Delta$) measuring the rate of degradation acceleration per lap.
* **`LapTime_Delta`**: Pace variation relative to the driver's theoretical personal best.
* **`RaceProgress`**: Normalized race completion percentage ($0.0 	o 1.0$).
* **`Position`**: Real-time track position.
* **`Pit_Into_Traffic`**: Binary spatial flag indicating whether a projected pit exit window ($\pm 2.5	ext{s}$ delta accounting for a 22.0-second pit lane loss) lands into active track traffic.

#### B. Anti-Leakage Validation Strategy
To prevent data leakage across consecutive laps in the same race stint:
* **Group-Based Validation:** Splitting is governed by `Race_Group` (`Year_Race`) using `LeaveOneGroupOut` principles. Entire race events are held out for validation rather than random row-wise sampling.
* **Dynamic Threshold Optimization:** Rather than defaulting to a $50\%$ decision boundary, the alert threshold is dynamically selected by optimizing the $F_1	ext{-score}$ ($ eta = 1.0$) across validation folds:

$$	ext{Threshold}^* =  rg\max_{	ext{th} \in [1, 99]} F_1\left(y_{	ext{val}},\, \hat{y}_{	ext{val}}(	ext{th})
ight)$$

---

### 2. Interactive Deployment: MATLAB Strategy App Designer

To bring model inferences into a realistic pit-wall environment, predictions are exported and visualized via an interactive **MATLAB App Designer Dashboard** (`.mlapp`).

* **Chronological 2025 Filtering:** Automatically restricts runtime choices to deployment season data, avoiding historical training overlap.
* **Dual-Marker Strategy Overlay:**
  * **Model Alert Triangles ($\hat{y} \ge 	ext{Threshold}$):** Plotted along the threshold line (or probability curve) in orange (`^`) to indicate laps where the Random Forest flagged an impending pit stop.
  * **Actual Pit Stop Circles ($y = 1$):** Rendered with `MarkerFaceAlpha = 0.55` transparency in red (`o`) directly over the probability curve so users can easily verify true track decisions against model triggers.
* **Multi-Compound Visualizer:** Color-coded stint lines according to official F1 Pirelli compound palettes (Soft, Medium, Hard, Intermediate, Wet).

---

### 3. Deep Learning Architecture (Neural Network / Transformer)

> *[ PLACEHOLDER: Deep Learning Architecture Section ]*

```
[ PLACEHOLDER: Neural Network / LSTM / Transformer Architecture Diagram ]
```

#### A. Network Topology & Loss Formulation
> *[ PLACEHOLDER: Insert Details on Network Layers, Activation Functions, Dropout Rates, and Custom Loss Functions (e.g., Focal Loss for Class Imbalance) ]*

#### B. Sequence & Telemetry Embedding
> *[ PLACEHOLDER: Describe Sequential Input Processing (e.g., LSTM/GRU windows, Attention Mechanisms) ]*

#### C. Performance Comparison & Benchmarking
> *[ PLACEHOLDER: Comparative Table contrasting Random Forest vs. Neural Network metrics (Precision, Recall, F1-Score, Inference Latency) ]*
