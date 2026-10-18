# MARL Simulator Environment

A multi-agent reinforcement learning (MARL) simulation environment. Follow the instructions below to set up your environment and run the project.

---

## Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Python 3.10** (recommended)
- A code editor of your choice (e.g., [Visual Studio Code](https://code.visualstudio.com/))
- Git / Terminal access

---

## Installation & Setup

### 1. Extract the Project Files

Locate `Simulator.zip` on your machine and extract its contents to your desired directory.

### 2. Open the Project

Open your chosen code editor (e.g., VS Code) and open the extracted project directory.

### 3. Navigate to the Working Directory

Open your terminal and change into the project root directory:

```bash
cd MARL-SimulatorEnvironment-main
```

### 4. Create and Activate a Virtual Environment

It is strongly recommended to use a virtual environment with Python 3.10 to prevent dependency conflicts.

- **On Windows:**

  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 5. Install Dependencies

Ensure your virtual environment is active, then install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Once the dependencies are installed, you can execute any script in the project directly from the terminal:

```bash
python <file_name>.py
```

> **Example:**
>
> ```bash
> python final.py
> ```

to run the final iteration of the model.

---

## Notes

`evaluate.py` has optional arguments to pass:

Run `python evaluate.py --episodes <number of episodes>` to define how many episodes you want the model to evaluate over

Run `python evaluate.py --aircraft <number of aircaft>` to test the model against any number of aircraft agents

Run `python evaluate.py --model <"model_name.pt">` to select a different .pt file in the project to evaluate (Allows for testing previous versions of the model i.e. final_test.pt and intermediate_test.pt)

Run `python evaluate.py --steps <number of steps>`. Default number of steps is 60. 60 steps = 300 seconds in real life

Run `python evaluate.py --output <"output_file_name.csv">`

_All optional arguments can be run in a single line_

- To deactivate the virtual environment when you are finished, run:
  ```bash
  deactivate
  ```
- Make sure your virtual environment remains activated whenever you run scripts or install additional packages.
