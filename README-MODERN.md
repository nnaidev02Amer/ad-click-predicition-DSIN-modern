# DSIN – Ad Click Prediction (modernised)

Modern (Python 3.9 / TensorFlow 2.15 / Keras) version of **Deep Session Interest Network** (Feng et al., IJCAI 2019, <https://arxiv.org/abs/1905.06482>) with a Streamlit dashboard. The original `DSIN/` folder is untouched; all work is in `DSIN-modern/`.

## Architecture (kept from the paper)
The network is DeepCTR 0.9.4's `DSIN` (tf.keras), which implements: session division (30-min gap, ≤5 sessions × ≤10 behaviours of category+brand) → session interest extractor (multi-head self-attention, 8 heads, bias or positional encoding) → session interest interacting layer (Bi-LSTM) → two target-attention "activating" layers keyed on the ad → concat with user profile / ad / price embeddings → DNN(200,80) → sigmoid CTR. `tests/test_model.py` checks these layers exist and that gradients flow.

Deviations / fixes (all documented):
| Area | Original | Modern |
|---|---|---|
| TF/Keras | TF 1.4 `ConfigProto`, `Session` | TF 2.15, `tf.keras`, CPU |
| Pickle names | wrote `*_0.25.pkl`, looked for `*_0.25_.pkl` | consistent names under `processed_data/<mode>/` |
| Behaviour log | read whole `behavior_log.csv` (README says `raw_behavior_log.csv`) | both names accepted; read in chunks, filtered to `pv` + sampled users |
| Sessions | fallback `len-1` could use sessions *after* the impression (future leak) | only sessions that started before the impression are used |
| Vocab size | `max(data)+1000` – can be smaller than ids in the behaviour log | covers the full cate/brand encoders |
| Optimiser | `adagrad` (Keras-1 default lr) | `Adagrad(lr=0.01)`, configurable; AUC metric, validation split, early stopping, checkpoint |
| Bias encoding | optional | `--bias-encoding` flag (forward pass tested; original authors warn of NaN loss on real data) |
| DIN/DIEN | scripts present | not ported (out of scope); original files kept in `code/` unchanged |

## Dataset status – IMPORTANT
The original Taobao dataset is **not present** on this machine, so **no real training was done and no real AUC/LogLoss exist**. Everything shown in *Demo* mode is trained on **synthetic** records (`DEMO MODE – SYNTHETIC DATA`) and says nothing about real performance.

Download <https://tianchi.aliyun.com/dataset/dataDetail?dataId=56> and extract into `raw_data\`:
```
raw_data\raw_sample.csv
raw_data\ad_feature.csv
raw_data\user_profile.csv
raw_data\behavior_log.csv      (or raw_behavior_log.csv – both accepted)
```
Note the behaviour log is ~23 GB and this laptop has ~8 GB RAM and little free disk; use a small user fraction (`--frac 0.05` or lower). The script refuses to build inputs that would not fit in free RAM.

## Setup (PowerShell, from `DSIN-modern`)
```powershell
cd "C:\Users\SURESH\Desktop\kits project\ad click predicition\DSIN-modern"
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1     # creates venv\ with Python 3.9 and installs everything inside it
.\venv\Scripts\python.exe verify_environment.py
```
Installed stack: Python 3.9.4, tensorflow-cpu 2.15.1, keras 2.15.0, deepctr 0.9.4 (installed `--no-deps`; it has no TF pin), numpy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1, joblib 1.5.3, requests 2.32.5, tqdm 4.70.1, streamlit 1.50.0, plotly 7.1.0, h5py 3.14.0, pytest 8.4.2 (full list: `requirements-modern.txt`).

## Demo mode (synthetic, works now)
```powershell
.\venv\Scripts\python.exe run_preprocessing.py --mode demo --make-demo
.\venv\Scripts\python.exe train_model.py --mode demo --epochs 6 --batch-size 512
.\venv\Scripts\python.exe evaluate_model.py --mode demo
```
## Real mode (needs the dataset)
```powershell
.\venv\Scripts\python.exe run_preprocessing.py --mode real --validate-only
.\venv\Scripts\python.exe run_preprocessing.py --mode real --frac 0.05
.\venv\Scripts\python.exe train_model.py --mode real --epochs 3 --batch-size 4096
.\venv\Scripts\python.exe evaluate_model.py --mode real
```
Split (as original): impressions before 2017‑05‑13 train (latest 10 % of them = validation), from 2017‑05‑13 test. Metrics: AUC, LogLoss on the held-out test split; confusion matrix at threshold 0.5.

Outputs: `processed_data/<mode>/` (inputs, schema, lookups), `models/<mode>/` (`dsin.weights.h5`, `model_config.json`), `results/<mode>/` (`evaluation.json`, `training_history.json`, `roc_curve.csv`, `prediction_samples.csv`, `dataset_validation.json`). Synthetic and real artefacts are in separate folders and never mixed.

## Dashboard
```powershell
.\venv\Scripts\streamlit.exe run app.py        # or: .\venv\Scripts\python.exe -m streamlit run app.py
```
Open <http://localhost:8501>. Four sections: **Home**, **Predict Advertisement Click**, **Prediction Results**, **About DSIN**. Prediction runs the trained DSIN model (never a random number) and is disabled with an explanation if no model is trained. The app uses the real-data model if one exists, otherwise the synthetic demo model, always labelled `DEMO MODE – SYNTHETIC DATA`. Predictions are saved locally in `results/prediction_history.sqlite`.
A predicted CTR is a model estimate, not a guarantee of a click. The user-id feature is set to an unseen "cold-start" slot in the form.

## Tests
```powershell
.\venv\Scripts\python.exe -m pytest tests -q      # ~5 min on CPU; uses a temp dir and synthetic data only
```

## Troubleshooting
* *`DATASET NOT READY`* – files missing/misnamed in `raw_data\`.
* *`MemoryError ... lower FRAC`* – use a smaller `--frac` and `--force`.
* *Low disk space* – `processed_data/` and TF need several GB; delete `processed_data\<mode>` to redo.
* *NaN loss with `--bias-encoding`* – known issue from the original authors; use positional encoding (default).
* TensorFlow prints deprecation warnings from DeepCTR internals; they are harmless.
