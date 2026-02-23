# Gas Turbine Energy Prediction MLOps Pipeline

## 1. Overview

This project implements an end-to-end MLOps pipeline for predicting gas turbine energy output (CO and NOx emissions) using AWS SageMaker. The pipeline demonstrates production-ready machine learning workflows including data preprocessing, feature engineering, model training, CI/CD automation, model monitoring, and real-time inference dashboards.

**Dataset:** Gas Turbine CO and NOx Emission Data Set (36,733 instances, 11 sensor measurements)
**Source:** UCI Machine Learning Repository
**Location:** Turkey (2011-2015)
**Team:** USD MS AAI 540 - Team 1

```mermaid
graph LR
    A[Raw Data<br/>2011-2015] --> B[Data Processing]
    B --> C[Feature Store]
    C --> D[Model Training]
    D --> E[Model Evaluation]
    E --> F{Quality Check}
    F -->|Pass| G[Model Registry]
    F -->|Fail| H[Alert & Retrain]
    G --> I[Real-time Inference]
    G --> J[Batch Inference]
    I --> K[Dashboard]
    J --> L[Monitoring]
```

---

## 2. Goal

### Primary Objectives
- **Predict turbine energy yield** from sensor measurements (temperature, pressure, humidity, etc.)
- **Implement production MLOps best practices** using AWS SageMaker
- **Automate ML lifecycle** from data ingestion to deployment
- **Enable continuous model improvement** through monitoring and CI/CD

### Success Metrics
- **Model Performance:** R² > 0.98, MSE < 2.72
- **Automation:** Fully automated training pipeline with conditional deployment
- **Observability:** Real-time monitoring with CloudWatch alarms
- **Scalability:** Production-ready inference endpoints

---

## 3. High-Level Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        S3[(S3 Buckets)]
        Athena[AWS Athena]
        FS[SageMaker<br/>Feature Store]
    end

    subgraph "ML Pipeline"
        NB1[1. Data<br/>Preprocessing]
        NB2[2. Athena<br/>Integration]
        NB3[3. Feature<br/>Engineering]
        NB4[4. Model<br/>Training]
        NB5[5. Model<br/>Evaluation]
    end

    subgraph "CI/CD Layer"
        Pipeline[SageMaker<br/>Pipeline]
        Condition{MSE & R²<br/>Threshold}
        Registry[Model<br/>Registry]
    end

    subgraph "Deployment"
        RT[Real-time<br/>Endpoint]
        Batch[Batch<br/>Transform]
        Dashboard[Inference<br/>Dashboard]
    end

    subgraph "Monitoring"
        CW[CloudWatch<br/>Alarms]
        SNS[SNS Alerts]
    end

    S3 --> NB1
    NB1 --> Athena
    Athena --> NB2
    NB2 --> FS
    FS --> NB3
    NB3 --> NB4
    NB4 --> NB5
    NB5 --> Pipeline
    Pipeline --> Condition
    Condition -->|Pass| Registry
    Condition -->|Fail| NB4
    Registry --> RT
    Registry --> Batch
    RT --> Dashboard
    Batch --> Dashboard
    RT --> CW
    CW --> SNS
```

---

## 4. Components

### 4.1 Data Pipeline

| Notebook | Description | Key Technologies |
|----------|-------------|------------------|
| `1_Data_Preprocessing.ipynb` | Merges 2011-2015 CSV files, handles missing data | Pandas, S3 |
| `2_Load_Processed_Data.ipynb` | Loads merged data from S3 | Boto3 |
| `3_Create_Athena_Database_fp.ipynb` | Creates Athena database for SQL queries | AWS Athena |
| `4_Register_CSV_with_Athena.ipynb` | Registers CSV data with Athena | PyAthena |
| `5_Convert_S3_csv_to_parquet.ipynb` | Converts CSV to Parquet for performance | Parquet |
| `6_Create_Feature_Store.ipynb` | Ingests features into SageMaker Feature Store | SageMaker Feature Store |

**Data Flow:**
```mermaid
flowchart LR
    A[Raw CSVs<br/>2011-2015] --> B[Merge & Clean]
    B --> C[Upload to S3]
    C --> D[Athena Database]
    D --> E[CSV → Parquet]
    E --> F[Feature Store]
    F --> G[Training Data]
```

### 4.2 Model Training

| Notebook | Model | Algorithm | Performance |
|----------|-------|-----------|-------------|
| `7a_Linear_regression.ipynb` | Linear Regression | SageMaker Linear Learner | Baseline model |
| `7b_Xgboost_regression.ipynb` | XGBoost | SageMaker XGBoost 1.7-1 | R² = 0.987, MSE = 2.72 |

**Training Configuration:**
- **Train/Val/Test Split:** 40% / 40% / 20%
- **Instance Type:** ml.m5.large
- **XGBoost Hyperparameters:**
  - `objective`: reg:squarederror
  - `num_round`: 200
  - `max_depth`: 6
  - `eta`: 0.1

```mermaid
graph LR
    A[Athena Query] --> B[Train Data<br/>40%]
    A --> C[Validation<br/>40%]
    A --> D[Test<br/>20%]
    B --> E[XGBoost<br/>Training]
    C --> E
    E --> F[Model Artifact<br/>S3]
    D --> G[Batch<br/>Transform]
    F --> G
    G --> H[Evaluation<br/>Metrics]
```

### 4.3 CI/CD Pipeline

**File:** `9_cicd.ipynb`

**Pipeline Steps:**
1. **Training Step:** Trains XGBoost model on S3 data
2. **Evaluation Step:** Runs evaluation script, generates metrics JSON
3. **Condition Step:** Checks if MSE ≤ 2.72 AND R² ≥ 0.987
   - **If Pass:** Register model → Create endpoint → Batch transform
   - **If Fail:** Pipeline fails with error message

```mermaid
stateDiagram-v2
    [*] --> Train: Start Pipeline
    Train --> Evaluate: Model Artifact
    Evaluate --> Condition: evaluation.json
    Condition --> Register: MSE ≤ 2.72 & R² ≥ 0.987
    Condition --> Fail: Metrics Below Threshold
    Register --> CreateModel
    CreateModel --> BatchTransform
    BatchTransform --> [*]: Success
    Fail --> [*]: Pipeline Failed
```

**Key Technologies:**
- SageMaker Pipelines
- Step Functions
- Model Registry
- Conditional Steps
- PropertyFile for metric passing

### 4.4 Model Monitoring

**File:** `8_Model_Monitoring_Alarm.ipynb`

**Monitoring Strategy:**
- **Metrics Tracked:** Invocations, Latency, Model Errors
- **Alarms:** CloudWatch alarms trigger SNS notifications
- **Threshold:** Configurable error rate and latency limits

```mermaid
graph LR
    A[Endpoint] --> B[CloudWatch<br/>Metrics]
    B --> C{Threshold<br/>Breach?}
    C -->|Yes| D[SNS Alert]
    C -->|No| E[Continue<br/>Monitoring]
    D --> F[Team<br/>Notification]
```

### 4.5 Real-Time Dashboard

**File:** `10_Model_Inference_Dashboard.ipynb`

**Features:**
- Real-time endpoint deployment (ml.m5.large)
- Historical vs. predicted energy yield visualization
- PST timezone-aware timestamp plotting
- Seaborn-styled interactive charts

**Dashboard Components:**
```mermaid
graph TB
    A[Batch Data] --> B[Real-time<br/>Endpoint]
    C[Historical Data] --> D[Dashboard Plot]
    B --> D
    D --> E[Energy Yield<br/>Predictions]
    E --> F[Visualization<br/>10 historical +<br/>5 future points]
```

---

## 5. Project Structure

```
MSAAI540/
├── Data/                          # Raw CSV files (2011-2015)
├── src/                           # Source notebooks
│   ├── 1_Data_Preprocessing.ipynb
│   ├── 2_Load_Processed_Data.ipynb
│   ├── 3_Create_Athena_Database_fp.ipynb
│   ├── 4_Register_CSV_with_Athena.ipynb
│   ├── 5_Convert_S3_csv_to_parquet.ipynb
│   ├── 6_Create_Feature_Store.ipynb
│   ├── 7a_Linear_regression.ipynb
│   ├── 7b_Xgboost_regression.ipynb
│   ├── 8_Model_Monitoring_Alarm.ipynb
│   ├── 9_cicd.ipynb
│   ├── 10_Model_Inference_Dashboard.ipynb
│   ├── code/
│   │   ├── evaluation.py         # Model evaluation script
│   │   ├── preprocessing.py      # Data preprocessing utilities
│   │   └── pre_processor_handler.py
│   └── inference.py               # Inference handler
├── models/                        # Saved model artifacts
├── testdata/                      # Test datasets
└── README.md                      # This file
```

---

## 6. Getting Started

### Prerequisites
- AWS Account with SageMaker access
- IAM Role: `LabRole` (or configure custom role)
- S3 Bucket: `usdmsaai540-spring2026-team1` (or create your own)
- Python 3.8+
- Jupyter Notebook / SageMaker Studio

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MSAAI540
   ```

2. **Configure S3 Bucket**
   - Update `DEFAULT_BUCKET = False` in notebooks
   - Set `bucket = "your-bucket-name"`

3. **Upload Raw Data**
   ```bash
   aws s3 cp --recursive Data/ s3://your-bucket/CCPP/data/uci_dataset/
   ```

4. **Run Notebooks Sequentially**
   - Start with `1_Data_Preprocessing.ipynb`
   - Follow numeric order through `10_Model_Inference_Dashboard.ipynb`

5. **Deploy CI/CD Pipeline**
   ```python
   # In 9_cicd.ipynb
   pipeline.upsert(role_arn=role)
   execution = pipeline.start()
   ```

---

## 7. Key Features

| Feature | Implementation |
|---------|----------------|
| **Data Versioning** | S3 with timestamped paths |
| **Feature Engineering** | SageMaker Feature Store |
| **Automated Training** | SageMaker Pipelines |
| **Model Registry** | SageMaker Model Registry |
| **Conditional Deployment** | ConditionStep with MSE/R² thresholds |
| **Monitoring** | CloudWatch + SNS |
| **Real-time Inference** | SageMaker Endpoints |
| **Batch Inference** | SageMaker Batch Transform |
| **Visualization** | Matplotlib + Seaborn dashboards |

---

## 8. Model Performance

### XGBoost Final Results

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **R²** | 0.987 | 0.986 | 0.987 |
| **MSE** | 2.68 | 2.72 | 2.71 |
| **RMSE** | 1.64 | 1.65 | 1.65 |

**Prediction Quality:**
```mermaid
graph LR
    A[Input Features] --> B[XGBoost Model]
    B --> C[Predicted kW]
    D[Actual kW] --> E[Comparison]
    C --> E
    E --> F[R² = 0.987]
```

---

## 9. Links

### AWS Resources
- [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker Feature Store](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)
- [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [AWS Athena](https://docs.aws.amazon.com/athena/)

### Dataset
- [UCI Gas Turbine Dataset](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)

### Course Materials
- USD MS AAI 540: ML Operations

---

## 10. References

1. **Dataset:** Kaya, H., Tüfekci, P., & Gürgen, F. S. (2012). Gas Turbine CO and NOx Emission Data Set. UCI Machine Learning Repository.
2. **AWS Documentation:** Amazon SageMaker Developer Guide (2026)
3. **MLOps Best Practices:** Continuous Delivery for Machine Learning (CD4ML)
4. **Course Labs:** USD MS AAI 540 - Lab 4, Lab 6
5. **Turkish Holidays:** [timeanddate.com/holidays/turkey](https://www.timeanddate.com/holidays/turkey/)

---

## 11. Contributors

**Team 1 - USD MS AAI 540 Spring 2026**
- Data Pipeline Development
- Model Training & Evaluation
- CI/CD Implementation
- Dashboard & Monitoring

---

## 12. License

This project is part of the University of San Diego MS in Applied Artificial Intelligence program coursework.

---

## 13. Acknowledgments

- USD Faculty for guidance on MLOps principles
- AWS for SageMaker platform and documentation
- UCI Machine Learning Repository for dataset access
