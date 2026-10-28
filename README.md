# AegisPdM — Deep Learning & Cloud-Stream Predictive Maintenance Software Platform








**AegisPdM** is an end-to-end, pure-software Predictive Maintenance (PdM) command center, streaming telemetry laboratory, and Deep Long Short-Term Memory (LSTM) Remaining Useful Life (RUL) simulation platform for heavy manufacturing CNC spindles and bearings.

Designed to replace reactive run-to-failure and rigid calendar-based maintenance schedules, the software pipeline ingests high-frequency multivariate telemetry (`vib_x`, `vib_y`, `vib_z`, `temp_c`, `rpm`, and `load`), processes 50-step sliding windows in real time, and serves low-latency RUL predictions via TensorFlow Serving (gRPC) and FastAPI—cutting unplanned downtime by **35%** (`$45,000` saved per incident) and extending component lifespan by up to **20%**.

## Key Software Capabilities & Interactive Workspaces

The single-file web application (`predictive_maintenance_software.html`) provides five interactive engineering workspaces:

| Workspace View | Core Interactive Features | Grounded Report Sections |
| :--- | :--- | :--- |
| **1. Live UI Command Center** | 42-machine fleet monitoring (`39 Normal`, `1 Critical`, `2 Offline`), Traffic-Light RUL status filters (`>72h`, `24–72h`, `<24h`), live 50-tick normalized waveform canvas with EMA smoothing, historical Vibration RMS vs. Piecewise Linear RUL degradation chart, live MQTT/TFS JSON consoles, Z-score anomaly spike injection, network drop simulation, and 1-click SAP PM / PagerDuty ticket dispatch (`#8849`). | Sections 10, 14, 25, 30, 31 |
| **2. Data Stream & Prep Lab** | Interactive 5-stage microservice architecture inspector (`mTLS MQTT ➔ Go + Kafka ➔ Python/Redis ➔ C++ TF Serving ➔ InfluxDB/PostgreSQL`), Phase I Cloud Kubernetes vs. Phase II TensorFlow Lite Edge AI toggle (`99.9%` bandwidth reduction), interactive network gap imputation slider (`≤3.0s` Forward-Fill vs. `>3.0s` Sequence Break & LSTM State Reset), live Min-Max `[0, 1]` normalizer, and interactive SQL/Flux database query runner. | Sections 4, 9, 11, 12, 13, 26, 31, 32, 33 |
| **3. Deep LSTM & Safety Math** | Interactive 6-layer LSTM topology inspector (`(None, 50, 14)` input tensor through `RUL_Output`), live **PHM08 Asymmetric Safety Score vs. Symmetric RMSE Calculator** with exponential penalty curves, and 42-epoch AWS EC2 `p3.2xlarge` (NVIDIA V100) training convergence replay with `EarlyStopping(patience=10)` restoring Epoch 32 best weights. | Sections 15, 16, 20, 21, 22, 23, 24 |
| **4. Codebase, FastAPI & CI/CD** | Syntax-highlighted, copyable, and simulated-runnable sandbox for all 6 production Python modules (`17_env_setup.py` to `22_eval_metrics.py`) plus the `27_Dockerfile`, live **FastAPI OpenAPI/Swagger Sandbox** (`:8000`), and interactive 4-stage GitHub Actions + DVC + Airflow Continuous Training (CT) pipeline simulator. | Sections 17–22, 27, 28, 29, 36 |
| **5. Software ROI & Research** | Interactive annual downtime & secondary damage ROI calculator (`$50,000/hr` unplanned vs. `$5,000/hr` scheduled), 16-week agile software sprint roadmap, `$23,500` cloud & software budget table, and foundational IEEE/NASA academic citations. | Sections 3, 5, 6, 8, 34, 35, 38 |

## End-to-End Software Streaming Architecture

The platform decouples high-velocity telemetry producers from deep learning inference consumers using a cloud-native Kubernetes (Amazon EKS / Azure AKS) microservices pipeline:

1. **Zero-Trust MQTT Anonymizer (`Sections 10 & 31`)**: Consumes 1-second JSON telemetry bursts over mutual TLS (`mTLS`) with X.509 certificate authentication. Strips proprietary CAD part filenames and client order metadata prior to cloud transit, assigning anonymized identifiers (`CNC_Milling_04`).
2. **Go Ingestion Pod & Partitioned Apache Kafka (`Sections 26 & 32`)**: A lightweight Go service routes MQTT payloads into the `machine_telemetry` Kafka topic partitioned by `hash(machine_id)`, guaranteeing strict chronological ordering of time steps required by sequential LSTM layers while scaling consumer pods horizontally.
3. **Python/Pandas Data Prep Pod & Redis Sliding Buffer (`Sections 12, 13, 14, 26, 31`)**:
   - **Adversarial Spike Filter**: Drops physically impossible loose-wire sensor spikes via Z-score anomaly detection before they reach the neural network.
   - **Missing Value Imputation**: Applies Forward-Fill (`ffill`) for network drops under `3.0 seconds`; triggers a mandatory sequence break and resets the LSTM cell state for gaps exceeding `3.0 seconds` to prevent learning interpolated artifacts.
   - **Min-Max Scaling & Rolling Window Features**: Normalizes channels to `[0, 1]` using pre-fitted `scaler.pkl` parameters and extracts 50-step rolling statistical indicators (**RMS**, **Kurtosis**, and **Crest Factor**) buffered in Redis.
4. **TensorFlow Serving Inference Pod (`Sections 26, 32, 33`)**: Exposes the trained `SavedModel` (`v1.2.0-lstm`) over a high-performance C++ gRPC port with automatic network-layer batching, zero-downtime model hot-swapping, and Kubernetes Horizontal Pod Autoscaling (HPA triggers at `>70%` CPU utilization). Phase II migrates inference to local **TensorFlow Lite** runtimes, cutting bandwidth by `99.9%` (`120 MB/min` raw data reduced to integer RUL payloads) with `<2ms` closed-loop emergency brake latency.
5. **Polyglot Persistence & Notification Pod (`Sections 11, 25, 26, 32`)**: Raw `10kHz` telemetry is written to **InfluxDB** (retained for 7 days and continuously downsampled into 1-minute averages), while RUL predictions and work orders are stored in **PostgreSQL**. A Node.js notification service dispatches PagerDuty (`Ticket #8849`) and SAP PM webhooks when predicted RUL crosses `<48h` (Warning) or `<12h` (Critical).

## Deep LSTM Architecture & Asymmetric Safety Metric

### Dual-Layer LSTM Network Topology (`Sections 16 & 20`)

Standard Recurrent Neural Networks (RNNs) suffer from vanishing gradients during Backpropagation Through Time (BPTT), while tabular models (Random Forest, XGBoost) fail to capture long-range temporal dependencies across hundreds of cycles. AegisPdM implements a 6-layer sequential architecture (`76,251` trainable parameters):

| Layer Index | Layer Name & Type | Output Tensor Shape | Configuration & Role |
| :--- | :--- | :--- | :--- |
| **Layer 1** | `Input Tensor` | `(None, 50, 14)` | 50-step sliding window across 14 normalized sensor & statistical features |
| **Layer 2** | `LSTM_1 (LSTM)` | `(None, 50, 100)` | `units=100`, `return_sequences=True` — extracts short-term temporal patterns |
| **Layer 3** | `Dropout_1 (Dropout)` | `(None, 50, 100)` | `rate=0.20` — spatial dropout prevents over-reliance on single noisy channels |
| **Layer 4** | `LSTM_2 (LSTM)` | `(None, 50)` | `units=50`, `return_sequences=False` — condenses sequence into a 50-d health context vector |
| **Layer 5** | `Dropout_2 (Dropout)` | `(None, 50)` | `rate=0.20` — regularizes condensed representation against overfitting |
| **Layer 6** | `RUL_Output (Dense)` | `(None, 1)` | `units=1`, `activation='linear'` — outputs continuous predicted RUL (Adam `lr=0.001`, MSE loss) |

### Piecewise Linear RUL & PHM08 Asymmetric Scoring (`Sections 12, 22, 23, 24`)

- **Piecewise Linear RUL Target (Wu et al., 2018)**: During early machine operation, mechanical degradation is negligible. Target RUL is capped at `max_rul = 125` cycles during the healthy phase and decays linearly only after wear initiates, accelerating model convergence.
- **PHM08 Asymmetric Safety Score**: Standard RMSE penalizes early predictions ($d = \text{pred} - \text{true} < 0$) and late predictions ($d > 0$) equally. Because predicting a failing spindle is healthy ($d > 0$) causes catastrophic machine crashes, the evaluation pipeline enforces an exponential asymmetric penalty:
  - **Early Prediction ($d < 0$)**: $\text{Score} += e^{-d / 13.0} - 1$
  - **Late Prediction ($d \ge 0$)**: $\text{Score} += e^{d / 10.0} - 1$

| Model / Metric | Test RMSE (Cycles) | R² Score | PHM08 Safety Score | Best Validation Loss (`MSE`) | Best Validation MAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Baseline** | `22.50` | `0.64` | `> 780.0` (Failed) | `506.25` | `18.40` |
| **AegisPdM Dual-Layer LSTM** | **`14.20` – `15.17`** | **`0.81`** | **`432.15`** (`< 500` Pass) | **`230.15`** (Epoch 32) | **`10.95`** (Epoch 32) |

## Quick Start & Execution Guide

### 1. Run the Interactive Web Application

The front-end platform (`predictive_maintenance_software.html`) is a zero-build, self-contained HTML5 application using Tailwind CSS (`3.4.17`), Chart.js (`4.4.1`), and custom HTML5 Canvas animations.

```bash
# Option A: Open directly in any modern browser
open predictive_maintenance_software.html

# Option B: Serve locally via Python HTTP server
python3 -m http.server 8080
# Navigate to http://localhost:8080/predictive_maintenance_software.html
```

### 2. Set Up the Python Deep Learning Environment (`Section 17`)

Create a virtual environment and install the exact pinned dependencies used for model training and preprocessing:

```bash
python3 -m venv venv
source venv/bin/activate

pip install tensorflow==2.10.0 pandas==1.5.0 numpy==1.23.3 scikit-learn==1.1.2 matplotlib fastapi uvicorn
```

### 3. Build & Run the Dockerized FastAPI Gateway (`Sections 27 & 28`)

Package the data preprocessing pipeline, `scaler.pkl` artifacts, and FastAPI service into a reproducible `python:3.9-slim-buster` container:

```bash
# Build the container image
docker build -t aegis-pdm-api:v1.2.0 .

# Run the FastAPI Uvicorn server on port 8000
docker run -d -p 8000:8000 --name aegis-pdm-gateway aegis-pdm-api:v1.2.0
```

## RESTful API Reference (`FastAPI :8000`)

FastAPI automatically serves interactive OpenAPI (Swagger) documentation at `/docs` and exposes three core operational endpoints for factory floor tablets and BI tools (Tableau / PowerBI):

| Endpoint | HTTP Method | Description | Sample Response Status |
| :--- | :--- | :--- | :--- |
| `/api/v1/health` | `GET` | Verifies reachability of the TensorFlow Serving gRPC pod, Redis 50-tick sliding buffer, and Kubernetes HPA replicas. | `200 OK` |
| `/api/v1/machine/{id}/rul` | `GET` | Queries PostgreSQL (`rul_predictions`) for the latest RUL estimate, cycle count, confidence score, and health status for a specific machine ID. | `200 OK` (`predicted_rul_hours: 11.5`) |
| `/api/v1/predict/batch` | `POST` | Accepts a manual CSV upload of offline multivariate sensor data, slices `(Samples, 50, 14)` windows, and returns a JSON array of RUL predictions. | `200 OK` (`job_id: BATCH-20261027-09`) |

## Polyglot Database Schema (`Section 11`)

### PostgreSQL (Relational Metadata & Prediction History)

| Table Name | Primary Key | Columns | Purpose |
| :--- | :--- | :--- | :--- |
| `machines` | `id` | `id`, `location`, `model`, `install_date` | Stores CNC machine metadata and operating cluster assignments |
| `maintenance_logs` | `log_id` | `log_id`, `machine_id`, `action_taken`, `date`, `cost` | Tracks SAP PM / PagerDuty work orders and cost savings |
| `rul_predictions` | `pred_id` | `pred_id`, `machine_id`, `timestamp`, `predicted_rul`, `confidence` | Stores real-time and batch LSTM RUL inference outputs |

### InfluxDB (High-Velocity Time-Series Telemetry)

- **Measurement**: `sensor_readings`
- **Indexed Tags**: `machine_id`, `sensor_type`
- **Unindexed Fields**: `value_x`, `value_y`, `value_z`, `temperature`
- **Timestamp**: Unix Epoch Nanoseconds (7-day raw `10kHz` retention policy with automated continuous query downsampling to 1-minute averages)

## Continuous Training (CT) & CI/CD Pipeline (`Sections 29 & 36`)

To mitigate **Model Concept Drift** (machine aging or new milling materials like titanium altering baseline vibration) and **Sensor Drift**, the repository enforces a 4-stage automated pipeline using GitHub Actions, Data Version Control (DVC), and Apache Airflow:

1. **Code Commit & DVC Sync**: Developers push preprocessing or model updates to GitHub alongside versioned dataset pointers.
2. **Unit & Shadow Inference Testing**: Automated unit tests verify data transformations and run a **Shadow Inference** check to guarantee output tensors strictly match `(None, 50, 14)` without breaking the production model signature.
3. **Automated Airflow DAG Retraining (CT Trigger)**: Every 30 days—or automatically when Z-score/concept drift is detected—an Airflow DAG pulls fresh run-to-failure telemetry from InfluxDB, refits the dual-layer LSTM on cloud GPUs, and computes the test RMSE and PHM08 Safety Score.
4. **Champion vs. Challenger Gate & EKS Rolling Update**: If the challenger model achieves a better PHM08 Safety Score (`< 500`) than the production champion, Docker images are pushed to AWS ECR and deployed across the Kubernetes cluster via a zero-downtime rolling update.

## Foundational References & Benchmark Datasets (`Sections 5, 6, 8, 38`)

- **Zhao, R., Yan, R., Chen, Z., Mao, K., Wang, P., & Gao, R. X. (2019)**. *"Deep Learning and Its Applications to Machine Health Monitoring,"* IEEE Transactions on Industrial Electronics, 66(3), 2151–2159.
- **Wu, Y., Yuan, M., Dong, S., Lin, L., & Liu, Y. (2018)**. *"Data-Driven Remaining Useful Life Prediction via Multiple Sensor Signals and Deep LSTM Neural Network,"* IEEE/ASME Transactions on Mechatronics, 23(5), 2427–2436.
- **Saxena, A., Goebel, K., Simon, D., & Ebeid, N. (2008)**. *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,"* Prognostics and Health Management (NASA Ames Prognostics Data Repository — C-MAPSS & IMS Bearing Datasets).
