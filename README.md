# LaserNet

Official PyTorch implementation of **LaserNet**, an end-to-end framework for contactless ECG waveform reconstruction based on structured-light-assisted defocused speckle imaging (SL-DSI).

LaserNet reconstructs ECG waveforms from contactless cardiac mechanical signals acquired by optical sensing. The input consists of complementary **Camera-SCG** and **Camera-PCG** signals, which characterize chest-wall motion and high-frequency cardiac mechanical activity, respectively.

The framework combines local and long-range temporal modeling to learn the nonlinear mapping from cardiac mechanical activity to the corresponding electrical ECG waveform.

---

## Introduction

Conventional ECG monitoring relies on skin-contact electrodes and lead wires, which may reduce patient comfort and can be affected by electromagnetic interference in MRI environments.

To address these limitations, we developed a contactless optical sensing framework based on **structured-light-assisted defocused speckle imaging (SL-DSI)**. Subtle cardiac-induced chest-wall motion is captured using high-frame-rate optical imaging and converted into mechanical cardiac signals.

Two complementary representations are used:

- **Camera-SCG** for chest-wall mechanical motion
- **Camera-PCG** for high-frequency heart-sound-related mechanical components

These signals are used to reconstruct the corresponding ECG waveform through an end-to-end deep learning framework.

---

## Dataset

The dataset used in this study contains synchronized optical-mechanical signals and reference ECG recordings.

A total of **66 participants** were included. Data were acquired under two representative experimental conditions:

- **noCoil**: without coil occlusion
- **ufCoil**: with ultra-flexible coil occlusion

The recordings contain synchronized Camera-SCG, Camera-PCG, and reference ECG signals for ECG waveform reconstruction.

### Dataset Availability

The dataset used in this work is currently **not publicly hosted**.

Researchers interested in accessing the dataset for **academic research purposes** are welcome to contact us by email.

**Email:** `liamgao825@gmail.com`

When requesting the dataset, please briefly provide your:

- Name
- Institution / affiliation
- Research topic
- Intended use of the dataset

The dataset can be provided upon reasonable request.

---

## Requirements

The code is implemented in **Python** and **PyTorch**.

A CUDA-enabled GPU is recommended for model training.

Typical dependencies include:

```text
Python
PyTorch
NumPy
Pandas
SciPy
scikit-learn
Matplotlib
mamba-ssm
```

Please install the corresponding versions according to your CUDA and PyTorch environment.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/aijian411/LaserNet.git
cd LaserNet
```

We recommend creating a separate Conda environment:

```bash
conda create -n lasernet python=3.9 -y
conda activate lasernet
```

Install PyTorch according to your CUDA environment, and then install the remaining dependencies.

---

## Data Preparation

The input data should contain synchronized cardiac mechanical signals and reference ECG.

For the proposed method, the main input signals are:

```text
Camera-SCG
Camera-PCG
```

and the reconstruction target is:

```text
ECG
```

The signals should be temporally aligned before training.

In our experiments, the signals were segmented into fixed-length sequences before being fed into the network.

---

## Training

Model training is performed through `run.py`.

A typical command follows the form:

```bash
python run.py \
  --is_training 1 \
  --data PCGECG \
  --root_path ./dataset/ \
  --seq_len 512 \
  --pred_len 256
```

Please modify the dataset path and model parameters according to your local configuration.

---

## Testing

After training, the model can be evaluated using:

```bash
python run.py \
  --is_training 0 \
  --data PCGECG \
  --root_path ./dataset/ \
  --seq_len 512 \
  --pred_len 256
```

The predicted ECG waveform can then be compared with the synchronized reference ECG.

---

## Evaluation

The reconstructed ECG is evaluated from both waveform morphology and temporal accuracy.

The primary evaluation metrics include:

- **PCC** — Pearson Correlation Coefficient
- **RMSE** — Root Mean Square Error
- **MDR** — Missed Detection Rate
- **R Error** — Absolute R-peak timing error

In addition, the temporal errors of characteristic ECG events can be evaluated for:

```text
Q peak
R peak
S peak
T peak
```

---

## Citation

If you find this repository or dataset useful for your research, please consider citing our work.

```bibtex
@article{gao2026lasernet,
  title   = {LaserNet: Contactless ECG Reconstruction Based on Structured-Light-Assisted Optical Sensing},
  author  = {Qiannan Gao and Yingen Zhu and Jiayu Zhang and Wenchao Weng and Wenjin Wang and Chun Sing Lai and Zhekang Dong},
  year    = {2026}
}
```

The citation information will be updated after the paper is formally published.

---

## Contact

For questions regarding the code, dataset, or research collaboration, please contact:

**Qiannan Gao**

School of Electronics and Information  
Hangzhou Dianzi University

Email: `liamgao825@gmail.com`

---

## Acknowledgement

This work was supported by the National Key R&D Program of China, the National Natural Science Foundation of China, and the Zhejiang Provincial Natural Science Foundation of China.
