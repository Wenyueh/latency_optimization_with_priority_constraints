# Semantic Scheduling for LLM Inference

## Introduction

<p align="center">
  <img width="550" alt="Screenshot 2025-06-17 at 11 41 57 PM" src="https://github.com/user-attachments/assets/6775df2e-88f5-448f-8350-013696da8a6f" />
</p>

Conventional operating system scheduling algorithms are largely contentignorant, making decisions based on factors such as latency or fairness without considering the actual intents or semantics of processes. Consequently, these algorithms often do not prioritize tasks that require urgent attention or carry higher importance, such as in emergency management scenarios. However, recent advances in language models enable semantic analysis of processes, allowing for more intelligent and context-aware scheduling decisions. In this paper, we introduce the concept of semantic scheduling in scheduling of requests from large language models (LLM), where the semantics of the process guide the scheduling priorities. We present a novel scheduling algorithm with optimal time complexity, designed to minimize the overall waiting time in LLM-based prompt scheduling. To illustrate its effectiveness, we present a medical emergency management application, underscoring the potential benefits of semantic scheduling for critical, time-sensitive tasks.

## QuickStart
```
conda create -n latency python=3.10
conda activate latency
pip install -r requirements.txt
```

## Simulation

We present five different scheduling mechanism with different simulation settings: 

One A100 with 4B model

One A100 with 7B model

One A5000 with 4B model

### Semantic Scheduling with Continuous Batching

This simulation implements priority-aware scheduling for large language model serving, using semantic importance and estimated remaining computation time to optimize request ordering.

- Semantic Priority (f_p): An integer value (0-4) where 0 represents highest urgency (e.g., emergency services) and 4 represents lowest priority
- Remaining Time (f_t): Estimated tokens left to generate, updated dynamically after each iteration

To run the simulation:

```bash
cd main
python main.py --user_request_gap 0.01 --max_concurrent_user_requests 5 --setting A100_4B
```
Here are several parameters you can try:

- user_request_gap (float): Average time interval (in seconds) between consecutive request batches. Lower values simulate higher traffic intensity and increased system load. Default: 0.01
- max_concurrent_user_requests (int): Maximum number of requests that can arrive simultaneously in a single batch. Higher values increase peak load and stress-test the scheduling algorithm's ability to handle burst traffic. Default: 5
- setting (str): Hardware and model configuration for the simulation. Available options:
    - A100_4B: NVIDIA A100 GPU with 4B parameter model
    - A100_7B: NVIDIA A100 GPU with 7B parameter model
    - A5000_4B: NVIDIA A5000 GPU with 4B parameter model
 
### Example Usage
Simulate high-load scenario with frequent bursts:
```bash
python main.py --user_request_gap 0.001 --max_concurrent_user_requests 50 --setting A100_4B
```
Simulate moderate load with smaller model:
```bash
python main.py --user_request_gap 0.1 --max_concurrent_user_requests 3 --setting A100_4B
```
The simulation tracks key metrics including per-priority-level latency, overall system throughput, and cache utilization efficiency under various load conditions.


## Shortest Job First (SJF) Scheduling
This baseline algorithm schedules requests purely based on predicted remaining computation time, without considering semantic importance. It implements the classic SJF policy adapted for LLM serving, where "job length" corresponds to the estimated number of tokens yet to be generated.

```bash
cd SJF
python main.py --user_request_gap 0.001 --max_concurrent_user_requests 50 --setting A100_4B
```

Characteristics:

- Minimizes average completion time across all requests
- Can lead to starvation of long requests under sustained high load
- Ignores semantic urgency, potentially delaying critical requests
- Requires accurate output length prediction for optimal performance


### Experiment Result Replication
To replicate the experiment results, run commands in experiment_scripts


## Priority Job First Scheduling
This algorithm schedules requests based solely on their semantic priority level, ignoring completion time estimates. Requests are processed in strict priority order (0 → 4), with FIFO ordering within each priority class.

```bash
cd PJF
python main.py --user_request_gap 0.001 --max_concurrent_user_requests 50 --setting A100_4B
```

Characteristics:

- Guarantees timely processing of high-priority requests
- May result in poor overall system throughput
- Can cause head-of-line blocking when high-priority requests are long
- Does not optimize for quick task completion within priority levels

## First-Come-First-Served (FCFS) Scheduling

The simplest baseline that processes requests in arrival order, without any reordering based on priority or estimated completion time. This represents standard queue behavior in most existing LLM serving systems.

```bash
cd FCFS
python main.py --user_request_gap 0.001 --max_concurrent_user_requests 50 --setting A100_4B
```

Characteristics:

- Fair in terms of arrival order
- No starvation possible
- Suboptimal for both latency and priority-sensitive workloads
- Serves as the baseline for measuring improvement from intelligent scheduling








