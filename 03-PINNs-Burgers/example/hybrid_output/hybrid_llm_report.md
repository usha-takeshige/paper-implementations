# LLM-based Hyperparameter Optimization Report
## Burgers PINNs Hyperparameter Tuning

Started: 2026-03-18T09:30:19Z

---

## 1. Configuration

| Parameter         | Value  |
|-------------------|--------|
| objective         | Accuracy  -rel_l2_error |
| n_initial         | 5   |
| n_iterations      | 20   |
| seed              | 123   |
| optimizer         | LLM (Gemini via LangChain) |

### Search Space

| Hyperparameter    | Type  | Low    | High   | Scale  |
|-------------------|-------|--------|--------|--------|
| n_hidden_layers   | int   | 2.0    | 8.0    | linear |
| n_neurons         | int   | 10.0   | 100.0  | linear |
| lr                | float | 0.0001 | 0.01   | log    |
| epochs_adam       | int   | 500.0  | 5000.0 | linear |

---

## 2. Phase 1: Sobol Initial Exploration

| Trial | n_hidden_layers | n_neurons | lr       | epochs_adam | Rel L2 Error | Time (s) | Objective  |
|-------|-----------------|-----------|----------|-------------|--------------|----------|------------|
| 0     | 3               | 62        | 6.54e-04 | 4958        | 5.7725e-01     | 63.43     | -5.7725e-01   |
| 1     | 6               | 34        | 9.89e-03 | 2269        | 5.6710e-01     | 36.72     | -5.6710e-01   |
| 2     | 8               | 96        | 2.22e-04 | 3165        | 4.7539e-01     | 196.46     | -4.7539e-01   |
| 3     | 4               | 22        | 2.91e-03 | 757         | 5.4349e-01     | 5.87     | -5.4349e-01   |
| 4     | 5               | 82        | 4.81e-03 | 1169        | 5.7290e-01     | 38.03     | -5.7290e-01   |

**Best after Phase 1** — Trial #2:
objective = -4.7539e-01, rel_l2 = 4.7539e-01

---

## 3. Phase 2: LLM-Guided Search

### Iteration 1 / 20  (Trial #5)

#### LLM Analysis of Current State

Preliminary results indicate that the best performance (Trial 2, Objective: -0.475) was achieved with the deepest (8 layers) and widest (96 neurons) architecture coupled with a relatively low learning rate (2.22e-4). Conversely, higher learning rates (Trial 1, Trial 4) and shallower networks with fewer neurons (Trial 3) generally yielded worse results. The relationship between training duration (epochs) and accuracy is still ambiguous, as high epochs in Trial 0 did not prevent a poor result when combined with mid-range depth and learning rate. The high-complexity, low-learning-rate region appears most promising.

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
| n_hidden_layers   | 7   |
| n_neurons         | 45   |
| lr                | 0.0001   |
| epochs_adam       | 5000   |

**Reasoning:**
This proposal explores the 'Deep and Low LR' region identified as promising by Trial 2, but tests it with a lower neuron count (45) and the maximum number of epochs (5000). This will help determine if the high performance in Trial 2 was driven specifically by the width (96 neurons) or if depth and extended training with a very small learning rate can achieve similar or better results. Exploring 7 layers and the minimum LR boundary (0.0001) provides diversity while focusing on the high-performing architectural trend.

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | -4.4070e-01   |
| Rel L2 Error      | 4.4070e-01   |
| Elapsed Time      | 111.31 s   |

**Best so far** — Trial #5: objective = -4.4070e-01

---

### Iteration 2 / 20  (Trial #6)

#### LLM Analysis of Current State

The trial history suggests that deeper networks (7 to 8 layers) and lower learning rates (0.0001 to 0.0002) are the most promising regions for minimizing the relative L2 error. Trial 5 and Trial 2 are the current leaders, both utilizing high hidden layer counts and low learning rates. Specifically, Trial 5 achieved the best objective value so far with 7 layers and a learning rate of 0.0001. Conversely, shallower networks and higher learning rates (e.g., Trial 1 and Trial 4) have consistently resulted in higher errors. The relationship between network width (n_neurons) and performance is still somewhat ambiguous, as Trial 5 used 45 neurons while Trial 2 used 96.

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
| n_hidden_layers   | 8   |
| n_neurons         | 85   |
| lr                | 0.00015   |
| epochs_adam       | 5000   |

**Reasoning:**
This configuration explores the upper boundary of the promising region (high depth and low learning rate) identified in Trials 2 and 5. By setting n_hidden_layers to 8 and epochs_adam to 5000, I am testing the limits of model capacity and training time. I chose 85 neurons to see if a wider network at this depth improves upon Trial 5's result (which had 45 neurons) or Trial 2's result (which had 96 neurons but fewer epochs). The learning rate of 0.00015 is a slight variation of the best-performing rate to help refine the optimal step size for this architecture.

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | -9.2943e-02   |
| Rel L2 Error      | 9.2943e-02   |
| Elapsed Time      | 292.38 s   |

**Best so far** — Trial #6: objective = -9.2943e-02

---

### Iteration 3 / 20  (Trial #7)

#### LLM Analysis of Current State

The trial history indicates a clear trend: lower learning rates (0.0001 to 0.0002) and higher epoch counts (5000) are critical for achieving lower relative L2 errors in this Burgers' equation PINN. Trial 6 is the current best performer by a significant margin, utilizing 8 layers and 85 neurons. Trials with higher learning rates (above 0.001) or fewer epochs consistently failed to converge to a good solution. There is a strong correlation between network depth/complexity and performance, but the boundaries between 6 to 8 layers with high neuron counts and very low learning rates remain under-explored.

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
| n_hidden_layers   | 6   |
| n_neurons         | 100   |
| lr                | 0.0001   |
| epochs_adam       | 5000   |

**Reasoning:**
This configuration explores the interaction between maximum width (100 neurons) and moderate depth (6 layers) using the most successful learning rate (0.0001) and epoch count (5000) identified so far. By reducing depth slightly compared to the current best but increasing width to the search space limit, we can determine if the network's total capacity or its specific depth is the primary driver of the performance gains seen in Trial 6. This supports the exploration phase's goal of identifying the boundaries of the high-performance region.

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | -4.9106e-01   |
| Rel L2 Error      | 4.9106e-01   |
| Elapsed Time      | 257.58 s   |

**Best so far** — Trial #6: objective = -9.2943e-02

---

### Iteration 4 / 20  (Trial #8)

#### LLM Analysis of Current State

The trial history indicates a significant performance improvement when using deep networks (n_hidden_layers=8) combined with maximum training duration (epochs_adam=5000). Trial 6 is the current best, achieving a relative L2 error of ~0.09 with 8 layers, 85 neurons, and a learning rate of 1.5e-4. Trials with higher learning rates (above 1e-3) or fewer epochs consistently failed to produce low error rates. While lower learning rates (1e-4 range) are clearly more stable, there is a distinct gap in performance between 6 layers (Trial 7: -0.49) and 8 layers (Trial 6: -0.09), suggesting that the architecture depth is a critical factor for Burgers' equation in this search space. The neuron count also appears to have a sweet spot between 45 and 96, with 85 currently being the most successful.

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
| n_hidden_layers   | 8   |
| n_neurons         | 72   |
| lr                | 0.00018   |
| epochs_adam       | 5000   |

**Reasoning:**
This configuration is designed to explore the high-performing region around Trial 6 while investigating the sensitivity of the neuron count and learning rate. By fixing n_hidden_layers at 8 and epochs_adam at 5000 (the most successful settings found so far), I am focusing on the architecture's capacity and convergence time. Choosing 72 neurons (down from 85) and a slightly higher learning rate of 1.8e-4 (up from 1.5e-4) will help determine the boundaries of the optimal region. This trial serves to check if the success of Trial 6 was dependent on a specific neuron count/learning rate combination or if a broader region of the 8-layer subspace is equally promising.

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | -5.2389e-01   |
| Rel L2 Error      | 5.2389e-01   |
| Elapsed Time      | 238.48 s   |

**Best so far** — Trial #6: objective = -9.2943e-02

---

### Iteration 5 / 20  (Trial #9)

#### LLM Analysis of Current State

The trial history suggests a highly localized region of high performance centered around maximum network depth (n_hidden_layers=8) and low learning rates (lr ≈ 1.5e-4). Trial 6 is the clear outlier with an objective of -0.0929, whereas slightly shifting the learning rate to 1.8e-4 (Trial 8) or the neuron count (Trial 2 vs Trial 6) resulted in a significant performance drop, even at the same depth. This indicates that the objective function for this Burgers' equation PINN is extremely sensitive to the interaction between network capacity and optimization hyperparameters. Regions with lower depth (n_hidden_layers < 6) or higher learning rates (lr > 5e-4) have consistently failed to break the -0.5 objective threshold.

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
| n_hidden_layers   | 8   |
| n_neurons         | 50   |
| lr                | 0.00015   |
| epochs_adam       | 5000   |

**Reasoning:**
I am proposing a configuration that tests the lower boundary of 'width' (n_neurons) within the successful high-depth/low-LR regime. By keeping n_hidden_layers at 8 and lr at 0.00015 (the best performing values so far) but reducing n_neurons to 50, we can determine if the high accuracy in Trial 6 requires high width (85+ neurons) or if the depth and learning rate are the primary drivers. This will help identify whether the next phase should focus exclusively on the top-right corner of the parameter space (high layers, high neurons) or if a narrower, deep architecture is sufficient.

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | -2.6286e-01   |
| Rel L2 Error      | 2.6286e-01   |
| Elapsed Time      | 144.38 s   |

**Best so far** — Trial #6: objective = -9.2943e-02

---

### Iteration 6 / 20  (Trial #10)

#### LLM Analysis of Current State

The trial history indicates a clear performance peak in Trial 6 (objective -0.0929) using the maximum depth of 8 layers, 85 neurons, a low learning rate of 0.00015, and the maximum 5000 epochs. Comparing Trial 6 to Trial 9 (50 neurons) and Trial 8 (72 neurons) suggests that at high depths, the objective value is highly sensitive to both neuron count and learning rate, with a possible sweet spot for width between 80 and 100 neurons. Trial 5 (7 layers, 45 neurons) also showed relatively better performance (-0.44) compared to initial trials, but still lagged significantly behind the 8-layer configurations. This suggests the search space boundary for high performance is likely concentrated at n_hidden_layers >= 7 and lr <= 0.0002.

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
| n_hidden_layers   | 7   |
| n_neurons         | 82   |
| lr                | 0.00015   |
| epochs_adam       | 5000   |

**Reasoning:**
This configuration is chosen to explore the boundary between 7 and 8 layers while keeping the most successful learning rate (0.00015) and epoch count (5000) found in Trial 6. By increasing the neuron count to 82 (up from 45 in the previous 7-layer Trial 5), we aim to determine if 7-layer networks can approach the accuracy of 8-layer networks when given sufficient width. This helps map the 'depth vs. width' trade-off in the high-performance region identified so far.

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | -4.5977e-01   |
| Rel L2 Error      | 4.5977e-01   |
| Elapsed Time      | 244.63 s   |

**Best so far** — Trial #6: objective = -9.2943e-02

---

### Iteration 7 / 20  (Trial #11)

#### LLM Analysis of Current State

The exploration has revealed a highly promising region for deep architectures with low learning rates. Trial 6 is currently the best configuration, achieving an objective of -0.0929 with 8 hidden layers, 85 neurons, a learning rate of 0.00015, and 5000 epochs. Analysis of the trial history suggests that: 1. Depth is crucial; models with 8 layers significantly out-perform those with 6 or 7 layers under similar learning rates (compare Trial 6 to Trials 7 and 10). 2. Learning rates above 0.0005 lead to poor performance, with the most effective range being 0.0001 to 0.0002. 3. There is high sensitivity in the width (n_neurons) dimension; Trial 6 (85 neurons) outperformed both Trial 9 (50 neurons) and Trial 2 (96 neurons), though Trial 2 had fewer epochs and a slightly higher learning rate. 4. Maximizing training duration (epochs_adam: 5000) consistently yields better results.

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
| n_hidden_layers   | 8   |
| n_neurons         | 92   |
| lr                | 0.00012   |
| epochs_adam       | 5000   |

**Reasoning:**
This configuration targets the most promising region (8 layers, 5000 epochs) while exploring the interaction between high network width and very low learning rates. By choosing 92 neurons, I am testing the space between the successful Trial 6 (85 neurons) and the mediocre Trial 2 (96 neurons). I have reduced the learning rate to 0.00012 to investigate if a slightly more conservative update helps stabilize learning in wider, deep networks, which might have been the issue in Trial 2. This will help define the boundaries of the high-performance 'island' discovered around Trial 6 before moving into the refinement phase.

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | -2.9362e-01   |
| Rel L2 Error      | 2.9362e-01   |
| Elapsed Time      | 331.81 s   |

**Best so far** — Trial #6: objective = -9.2943e-02

---

### Iteration 8 / 20  (Trial #12)

#### LLM Analysis of Current State

Analysis of the 12 trials indicates that the configuration of 8 hidden layers combined with the maximum allowable epochs (5000) is the most promising region. Performance appears highly sensitive to the learning rate and neuron count. Specifically, learning rates in the range of 1.2e-4 to 1.8e-4 have shown significantly better results than higher rates, with the current best performance (Trial 6) occurring at 1.5e-4. The neuron count also has a non-linear impact; while 85 neurons performed best, trials at 92 and 72 neurons showed a sharp drop in objective value. This suggests a narrow optimal peak within the 8-layer, high-epoch, low-learning-rate subspace.

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
| n_hidden_layers   | 8   |
| n_neurons         | 90   |
| lr                | 0.00016   |
| epochs_adam       | 5000   |

**Reasoning:**
This proposal explores the boundaries of the high-performing region identified in Trial 6. By selecting 90 neurons and a learning rate of 0.00016, we are testing the space between the current best (85 neurons, 0.00015 lr) and other relatively successful trials (Trial 11 at 92 neurons, Trial 8 at 1.8e-4 lr). Maintaining 8 layers and 5000 epochs is a strategic choice, as these parameters have consistently yielded the best results. This specific combination aims to determine if the optimum lies at a slightly higher neuron count and learning rate than the current best, helping to define the range for the subsequent Bayesian refinement phase.

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | -2.2039e-01   |
| Rel L2 Error      | 2.2039e-01   |
| Elapsed Time      | 358.16 s   |

**Best so far** — Trial #6: objective = -9.2943e-02

---

### Iteration 9 / 20  (Trial #13)

#### LLM Analysis of Current State

The exploration has identified that the maximum depth (n_hidden_layers=8) is currently the most promising region of the search space, especially when paired with a high number of training epochs (5000). The most successful trials (6, 12, 9, 11) all utilize 8 layers and learning rates in the narrow range of 0.00012 to 0.00016. However, there is significant sensitivity to the n_neurons parameter; while 85 neurons (Trial 6) yielded the best results, performance dropped when moving to 72 (Trial 8) or 96 (Trial 2). Lower depths (3-7) have consistently underperformed compared to the 8-layer models.

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
| n_hidden_layers   | 8   |
| n_neurons         | 78   |
| lr                | 0.00014   |
| epochs_adam       | 5000   |

**Reasoning:**
This configuration explores the interior of the high-performing 8-layer region. By selecting 78 neurons, we are testing the gap between the successful 85-neuron trial (Trial 6) and the lower-performing 72-neuron trial (Trial 8). Using a learning rate of 0.00014 helps determine if the previous performance drop-off was due to the neuron count or an slightly over-aggressive learning rate. This will help define the robust boundaries of the 'optimal' basin before the refinement phase.

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | -4.6507e-01   |
| Rel L2 Error      | 4.6507e-01   |
| Elapsed Time      | 438.52 s   |

**Best so far** — Trial #6: objective = -9.2943e-02

---

### Iteration 10 / 20  (Trial #14)

#### LLM Analysis of Current State

The trial history shows that the highest accuracy is concentrated at the maximum depth of the search space (n_hidden_layers=8) combined with maximum training iterations (epochs_adam=5000). The optimal learning rate appears to be very low, specifically in the 0.0001 to 0.0002 range. Within this 'deep and slow' region, there is high sensitivity to the number of neurons; Trial 6 (85 neurons) achieved the best result (-9.29e-2), while nearby configurations at 78, 90, and 92 neurons performed significantly worse. This suggests either a very narrow objective peak or a high degree of stochasticity in the training process for this region.

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
| n_hidden_layers   | 8   |
| n_neurons         | 68   |
| lr                | 0.00018   |
| epochs_adam       | 5000   |

**Reasoning:**
I am proposing a configuration that stays within the promising 'deep and slow' region (8 layers, 5000 epochs) but explores an under-sampled neuron count (68). Previous trials have explored the 50-neuron (Trial 9) and 78-100 neuron ranges (Trials 6, 12, 13, 11, 7). Probing 68 neurons will help determine if the high performance seen at 85 neurons is an isolated peak or if a broader region of effective configurations exists between 50 and 80 neurons. The learning rate of 0.00018 tests the upper boundary of the currently successful range.

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | -1.5677e-01   |
| Rel L2 Error      | 1.5677e-01   |
| Elapsed Time      | 376.27 s   |

**Best so far** — Trial #6: objective = -9.2943e-02

---

## 4. Final Best Result

Completed: 2026-03-18T11:31:13Z

**Trial ID**: 6
**Objective**: -9.2943e-02

| Hyperparameter    | Value  |
|-------------------|--------|
| n_hidden_layers   | 8   |
| n_neurons         | 85   |
| lr                | 0.00015   |
| epochs_adam       | 5000   |

**Metrics**:
- Relative L2 Error: 9.2943e-02
- Elapsed Time: 292.38 s

---

## 5. Convergence

Best objective per trial (cumulative max):

| Trial | Best Objective So Far |
|-------|-----------------------|
| 0     | -5.7725e-01              |
| 1     | -5.6710e-01              |
| 2     | -4.7539e-01              |
| 3     | -4.7539e-01              |
| 4     | -4.7539e-01              |
| 5     | -4.4070e-01              |
| 6     | -9.2943e-02              |
| 7     | -9.2943e-02              |
| 8     | -9.2943e-02              |
| 9     | -9.2943e-02              |
| 10    | -9.2943e-02              |
| 11    | -9.2943e-02              |
| 12    | -9.2943e-02              |
| 13    | -9.2943e-02              |
| 14    | -9.2943e-02              |
