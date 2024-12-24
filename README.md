# latency optimization with priority constraints

## latest code under Dec_23 directory

## Abstract

In contemporary LLM servers, the efficient scheduling of user requests is paramount for optimizing performance and ensuring timely responses. User requests often arrive with varying priorities, dependencies, and constraints, presenting complex challenges for system designers aiming to minimize latency and maximize resource utilization. This project focuses on developing latency optimization algorithms for user request scheduling under constraints.

Various scenarios where constraints need to be respected under latency optimization may occur. For example, in  quality of service (QoS) Guarantees: service providers often have agreements specifying performance metrics, including maximum allowable latency. Relative constraints based on service level agreements (SLAs) that prioritize certain user requests over others. Failing to meet QoS guarantees can lead to customer dissatisfaction, contractual penalties, and damage to reputation. In another common example, emergency and critical response systems: applications in healthcare, defense, and emergency services where timely responses are vital. Priority of different user requests and strict protocols for processing and disseminating information. Delays can have life-threatening consequences or lead to significant property damage.

To study latency optimization under various constraints, we first categorized all possible settings. We start by categorizing constraints into relative constraints and hard constraints. Relative constraints impose an ordering among user requests and are further divided into graded partially ordered sets (graded posets), where a consistent rank function allows prioritization based on criteria like importance or urgency, and non-graded partially ordered sets (non-graded posets), where inconsistent chain lengths due to complex dependencies prevent such ranking. Hard constraints enforce strict deadlines for processing completion. We also explored different processing types: sequential processing and batch processing; scheduling strategies like batch processing with bulk arrivals and online processing with continuous batching, examining how these interact with the categorized constraints to inform effective scheduling decisions for optimizing latency in various operational settings.

## QuickStart
```
conda create -n latency python=3.10
conda activate latency
pip install -r requirements.txt
```

To run simulation code:
```
cd Dec_23
python main.py
```
