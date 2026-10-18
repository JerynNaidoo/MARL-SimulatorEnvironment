# Multi-Agent Reinforcement Learning (MARL) Air Traffic Simulator

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Simulator](https://img.shields.io/badge/Simulator-BlueSky-orange.svg)](https://github.com/ProfLoek/BlueSky)

A Multi-Agent Reinforcement Learning (MARL) environment for air traffic management, conflict detection, and resolution. This project extends the **BlueSky-Gym** wrapper to support multi-aircraft control and coordination using deep reinforcement learning policies.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage & Execution](#usage--execution)
  - [1. Evaluation](#1-evaluation)
  - [2. Model Training](#2-model-training)
  - [3. Scenario Generation & Visualization](#3-scenario-generation--visualization)
- [Pre-trained Checkpoints & Benchmarks](#pre-trained-checkpoints--benchmarks)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Traditional air traffic simulators typically handle centralized or single-agent heuristics. This repository adapts the [BlueSky](https://github.com/ProfLoek/BlueSky) Open Air Traffic Simulator into a Gymnasium-compatible Multi-Agent RL environment (`bluesky_gym`).

Multiple aircraft agents learn to safely navigate airspace, maintain required separation minima, avoid conflicts, and reach designated waypoints cooperatively.

---

## Repository Structure

```text
MARL-SimulatorEnvironment-main/
│
├── bluesky_gym/                # Custom BlueSky multi-agent environment wrapper
├── scenarios/                  # BlueSky airspace & traffic scenario definitions (.scn)
├── visualisations/             # Output plots, trajectory charts, and metrics
├── results/                    # Generated test runs and evaluation logs
│
├── evaluate.py                 # Evaluation script with CLI arguments
├── final.py                    # Primary / final training and execution pipeline
├── intermediate.py             # Intermediate experiment script
├── initial.py                  # Baseline / initial prototype script
├── flight_paths.py             # Trajectory extraction and path plotting
├── comparison.py               # Performance comparison across policies
├── scnGenerator.py             # Dynamic scenario generator for BlueSky
│
├── actor_policy_final.pt       # Pre-trained final actor policy checkpoint
├── final_test.pt               # Test evaluation checkpoint
├── intermediate_test.pt        # Intermediate policy checkpoint
├── final_thesis_benchmark_table.csv # Benchmark results across scenarios
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Build & package metadata
```

---

## Prerequisites

- **Python 3.10** (recommended)
- **Pip** package manager
- A terminal / code editor (e.g., [VS Code](https://code.visualstudio.com/))

---

## Installation & Setup

### 1. Extract the Project Files

If working from a downloaded archive, extract `Simulator.zip` to your chosen workspace directory.

### 2. Open Project in Terminal

Open your terminal or IDE (VS Code) and navigate to the project directory:

```bash
cd MARL-SimulatorEnvironment-main
```

### 3. Create & Activate a Python 3.10 Virtual Environment

- **On Windows:**

  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies

Ensure your virtual environment is active, then install all required packages:

```bash
pip install -r requirements.txt
```

---

## Usage & Execution

### 1. Evaluation

Evaluate pre-trained policies across multi-agent scenarios using `evaluate.py`:

```bash
python evaluate.py --model actor_policy_final.pt --episodes 10 --aircraft 4 --steps 500
```

#### Available CLI Arguments:

| Argument     | Description                                   | Default                 |
| :----------- | :-------------------------------------------- | :---------------------- |
| `--model`    | Path to the `.pt` policy checkpoint           | `actor_policy_final.pt` |
| `--episodes` | Number of evaluation episodes                 | `10`                    |
| `--aircraft` | Number of simultaneous aircraft agents        | `4`                     |
| `--steps`    | Maximum simulation steps per episode          | `500`                   |
| `--output`   | Destination directory or filename for results | `results/`              |

---

### 2. Model Training

Run the training pipelines using the progression scripts:

- **Final Model Pipeline:**
  ```bash
  python final.py
  ```
- **Intermediate / Baseline Experiments:**
  ```bash
  python intermediate.py
  python initial.py
  ```

---

### 3. Scenario Generation & Visualization

- **Generate Custom Airspace Scenarios:**
  ```bash
  python scnGenerator.py
  ```
- **Plot Aircraft Flight Paths & Conflicts:**
  ```bash
  python flight_paths.py
  ```
- **Compare Benchmark Results:**
  ```bash
  python comparison.py
  ```

Generated graphs and plots are saved to the `visualisations/` folder.

---

## Pre-trained Checkpoints & Benchmarks

This repository includes trained PyTorch policy weights ready for testing:

- **`actor_policy_final.pt`**: Tuned multi-agent actor network.
- **`final_test.pt`** & **`intermediate_test.pt`**: Milestone checkpoints from training runs.

Full performance data across various agent densities and scenario complexities is tabulated in:

```text
final_thesis_benchmark_table.csv
```

## Acknowledgments

- Built on top of the open-source [BlueSky](https://github.com/ProfLoek/BlueSky) Open Air Traffic Simulator developed by TU Delft.
- Inspired by the original `bluesky-gym` environment wrapper.
