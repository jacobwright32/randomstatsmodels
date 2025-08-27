| Rank (fit cost) | Class (type) | Core method | Seasonality | Trend | Non-linear | Robust | Data need | Fit cost | Predict cost | Good for | Why this rank / notes |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| 1 | WindowAverageForecaster (forecaster) | Rolling mean of last `window` | No | Flat | No | Low | Short | Very low | Very low | Quick baselines, smooth series | Pure rolling mean; O(1) per step; almost no params. |
| 2 | RollingMedianForecaster (forecaster) | Rolling median of last `window` | No | Flat | No | **High** | Short | Very low | Very low | Outlier-ridden series | Median swap for robustness; tiny overhead vs mean. |
| 3 | TrimmedMeanForecaster (forecaster) | Mean after trimming `alpha` tails | No | Flat | No | High | Short | Very low | Very low | Lightly noisy series | Slightly slower than mean, faster than median for large windows. |
| 4 | KNNForecaster (forecaster) | Analog search over past windows (`k`-NN) | No | Local patterns | **Yes** | Medium | Moderate | Low–Med | Low–Med | Repeating motifs, regimes | Simple Euclidean window matching; library precomputed. |
| 5 | FourierForecaster (forecaster) | Harmonics (sine/cosine) + optional linear trend (LS) | **Yes** | Optional | No | Low | Moderate | Low | Low | Clean periodic signals | Small design matrix; LS solve once. |
| 6 | NEOForecaster (forecaster) | Polynomial features on lags (LS) | No | Yes (poly) | **Yes** | Low | Moderate | Low–Med | Low | Smooth nonlinear dynamics | Closed-form, but features grow with degree & lags. |
| 7 | SeasonalARForecaster (forecaster) | AR with seasonal Fourier features (LS) | **Yes** | Yes | Mild | Low | Moderate | Med | Low | Multiple seasonalities (e.g., 7 & 365) | Adds time index / Fourier terms; still LS. |
| 8 | PALF (forecaster) | Proximal aggregation of lag “anchors” w/ robust penalties | No | Level-stabilized | Mild | **High** | Moderate | Med | Low | Spiky & irregular series | Per-step scalar minimization; robust loss; optional level penalty. |
| 9 | MELDForecaster (forecaster) | Multiscale embeddings + RFF (kernel) + analog blend | Optional | From features | **Yes** | Medium | Long | Med–High | Med | Nonlinear w/ repeating shapes | Heavier feature lift, but still closed-form ridge; plus kNN blend. |
| 10 | HybridForecastNet (forecaster) | Fourier + trend + AR linear core **+ GRU residual** | **Yes** | Yes | **Yes** | Medium | Long | **High** | Low–Med | Hard nonlinearities; residual patterns | Adds GRU training on residuals; most compute-intensive of file. |
| — | AutoWindow (auto) | Tunes rolling-mean window | — | — | — | — | — | Low | — | Fast baselines | One-step rolling validation; tiny grid. |
| — | AutoRollingMedian (auto) | Tunes median window | — | — | — | — | — | Low | — | Robust baselines | Same as above; median. |
| — | AutoTrimmedMean (auto) | Tunes (`window`,`alpha`) | — | — | — | — | — | Low–Med | — | Robust baselines | Small 2-D grid. |
| — | AutoKNN (auto) | Tunes (`window`,`k`) | — | — | — | — | — | Med | — | Motif series | One-step rolling with library growth. |
| — | AutoFourier (auto) | Tunes harmonics + trend | — | — | — | — | — | Low | — | Periodic data | Very small grid. |
| — | AutoNEO (auto) | Tunes lags × degree × window | — | — | — | — | — | Med | — | Smooth nonlinear | Grid over poly degree blows features. |
| — | AutoSeasonalAR (auto) | Tunes lags × degree × seasonal periods × fourier order | — | — | — | — | — | Med–High | — | Multi-seasonal signals | Largest classical grid here. |
| — | AutoPALF (auto) | Tunes penalties/decay/hyper-loss | — | — | — | — | — | Med | — | Noisy/irregular | Robust-loss grid adds combos. |
| — | AutoHybridForecaster (auto) | Tunes HybridForecastNet (Fourier/trend/AR/hidden) | — | — | — | — | — | **High** | — | Complex nonlinear | Heaviest search due to GRU training. |
| — | AutoMELD (auto) | Tunes MELD (lags/scales/RFF/…/kNN) | — | — | — | — | — | Med–High | — | Nonlinear, regimes | Several interacting grids (RFF, kNN, etc.). |
