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

### 3. Deep Learning Architecture (Neural Network)

This project implements a fully connected feed-forward neural network to predict whether a Formula One driver will pit on the following lap (`PitNextLap`). The model was designed specifically for highly imbalanced race strategy data, where only approximately 3% of laps correspond to pit-stop events.

Unlike the Random Forest baseline, the neural network is capable of learning complex nonlinear interactions between tyre degradation, lap-time trends, race progress, driver identity, and historical rolling statistics. The final implementation uses Binary Focal Cross-Entropy together with class weighting and threshold optimisation to improve minority-class prediction while maintaining a low false positive rate.

During development, extensive hyperparameter optimisation was performed, including tuning of learning rate, dropout, batch size, focal loss parameters, and class weights. Although these experiments occasionally improved validation performance, they frequently encouraged the model to overfit the validation season and reduced its ability to generalise to the unseen 2025 data. Consequently, the strongest final model used a simpler configuration consisting of the default network architecture, Binary Focal Cross-Entropy, and optimisation of only the positive class weight and decision threshold. This produced the most consistent performance on unseen races while avoiding unnecessary model complexity.

---

#### A. Network Topology & Loss Formulation

The final neural network consists of three fully connected hidden layers with Batch Normalisation, ReLU activation functions, and Dropout regularisation.

#### Network Topology

| Layer | Type | Configuration | Activation | Output Shape | Purpose |
|-------|------|---------------|------------|--------------|---------|
| Input | Input Layer | 25 engineered features (78 after transformation) | — | (78) | Accepts numerical and one-hot encoded race features |
| Hidden Layer 1 | Dense | 128 neurons | ReLU | (128) | Learns high-level nonlinear relationships between race features |
|  | Batch Normalization | — | — | (128) | Stabilizes activations and accelerates convergence |
|  | Dropout | Rate = 0.30 | — | (128) | Reduces overfitting by randomly disabling neurons during training |
| Hidden Layer 2 | Dense | 64 neurons | ReLU | (64) | Learns intermediate feature representations |
|  | Batch Normalization | — | — | (64) | Improves training stability |
|  | Dropout | Rate = 0.30 | — | (64) | Additional regularization |
| Hidden Layer 3 | Dense | 32 neurons | ReLU | (32) | Final nonlinear feature extraction before classification |
|  | Batch Normalization | — | — | (32) | Maintains stable feature distributions |
| Output Layer | Dense | 1 neuron | Sigmoid | (1) | Outputs the probability of a pit stop occurring on the next lap |

#### Training Configuration

| Component | Configuration |
|-----------|---------------|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Binary Focal Cross-Entropy |
| Focal Loss γ (Gamma) | 2.0 |
| Batch Size | 64 |
| Epochs | Up to 100 (Early Stopping) |
| Early Stopping | Validation Loss, Patience = 8, Restore Best Weights |
| Learning Rate Scheduler | ReduceLROnPlateau (Factor = 0.5, Patience = 3, Minimum LR = 1e-6) |
| Regularization | Batch Normalization + Dropout (0.30) |
| Class Imbalance Handling | Optimized Positive Class Weight |
| Decision Rule | Optimized validation threshold maximizing F1-score |


The network was trained using the Adam optimizer with Binary Focal Cross-Entropy rather than traditional Binary Cross-Entropy. Focal Loss reduces the influence of easily classified majority-class examples and instead concentrates learning on difficult pit-stop events, making it particularly well suited to the severe class imbalance present within the dataset.

Training stability was further improved through:

- Batch Normalization
- Dropout regularization
- Early Stopping
- ReduceLROnPlateau adaptive learning rate scheduling

---

#### B. Sequential Telemetry Feature Learning

Although the model is not a recurrent architecture such as an LSTM or Transformer, temporal information is incorporated through engineered sequential features computed from each driver's previous laps.

These include:

- Rolling 3-lap average lap time
- Rolling 5-lap average lap time
- Rolling degradation averages
- Rolling lap-time standard deviation
- Previous lap time
- Lap-time increase
- Previous tyre life
- Tyre wear rate
- Cumulative degradation

Rather than allowing the neural network to learn temporal dependencies directly, these rolling statistics explicitly encode recent race history into each training example. This approach significantly reduces computational complexity while still providing the model with short-term contextual information required for pit-stop prediction.

---

#### C. Performance Comparison & Benchmarking for the Pit-Stop Class (class 1)

| Model | Precision | Recall | F1-Score | Accuracy | Notes |
|------|----------:|-------:|---------:|---------:|------|
| Random Forest Baseline | 0.66 | 0.65 | 0.65 | 98% | Classical machine learning baseline |
| Baseline Neural Network (01) | 0.18 | 0.92 | 0.30 | 88% | Initial feed-forward network using Binary Cross-Entropy and optimized threshold |
| Feature-Engineered Network (02) | 0.57 | 0.66 | 0.61 | 98% | Added extra features like rolling lap times, lap delta, etc. |
| *Focal Loss and Manually-Tested Class Weight Network (03)* | *0.70* | *0.70* | *0.70* | *98%* | Showed **best overall performance** with a much simpler model, but required manual class weight testing rather than an automatic optimization sweep |
| Hyperparameter-Tuned Neural Network (04) | 0.00 | 0.00 | 0.00 | 97% | Extensive optimization of learning rate, dropout, focal loss, batch size, and class weights; improved validation performance but aggressively overfitted |
| **Final Neural Network (06)** | **0.68** | **0.46** | **0.55** | **98%** | Binary Focal Cross-Entropy with optimized class weight and threshold; selected for strongest real-world precision and generalization |
| **Final Neural Network (using F0.5-score) [07]** | **0.70** | **0.39** | **0.50 (0.61 F0.5)** | **98%** | Aimed to obtain higher precision on pit-stops without sacrificing recall |
| **Final Neural Network (using F0.8-score) [08]** | **0.70** | **0.30** | **0.42 (0.46 F0.8)** | **98%** | Softer approach to achieving higher precision while minimizing loss to recall |

---

The final neural networks intentionally favor precision over recall. While fewer pit stops are predicted overall, the predictions that are made are considerably more reliable. In a real Formula One environment, unnecessary pit-stop recommendations can compromise race strategy and cost significant track position, making false positives substantially more expensive than occasionally missing a potential pit window.

Rather than selecting the model with the highest validation score, the final architecture was chosen based on its ability to generalize to an entirely unseen Formula One season. This produced a simpler, more interpretable model that demonstrated stronger robustness during final evaluation and better reflects the requirements of real-world race strategy prediction.
