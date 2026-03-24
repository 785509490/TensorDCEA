# TensorizedDCEA: A GPU-Accelerated Framework for Real-Time Dynamic Constrained Multiobjective Optimization

> **Note on Anonymity:** This repository is currently under submission for peer review. In accordance with the **anonymous review policy** of the target venue, all author-related information has been withheld from this repository until the review process is complete.

## 1. Introduction

This repository contains the source code for the algorithm presented in the paper:

*TensorizedDCEA: A GPU-Accelerated Framework for Real-Time Dynamic Constrained Multiobjective Optimization*

The code of this project is developed based on the extension of the [EvoX](https://github.com/EMI-Group/evox) framework.

This paper introduces a novel evolutionary algorithm framework designed to leverage the power of GPUs for solving **dynamic constrained multi-objective optimization problems** efficiently in real time.

## 2. Supported Algorithms and Test Problems

This implementation includes the proposed algorithm and a set of established benchmark problems:

### Supported Algorithms

| Algorithm      | Description                                                         |
|----------------|---------------------------------------------------------------------|
| **DC-NSGA2-A** | Dynamic Constrained NSGA-II Variant A                               |
| **DC-NSGA2-B** | Dynamic Constrained NSGA-II Variant B                               |
| **DC-MOEA**    | https://www.sciencedirect.com/science/article/pii/S2210650217302717 |
| **dCMOEA**     | https://ieeexplore.ieee.org/abstract/document/8926382/              |
| **TDCEA**      | https://ieeexplore.ieee.org/abstract/document/10246310              |
| **TensorDCEA** | Proposed    Algorithm                                               |

### Test Problems

We provide the following dynamic constrained benchmark test suite:

- **DCP1 ~ DCP9** — 9 dynamic constrained multi-objective test problems

These test problems represent a diverse range of dynamic constrained multi-objective optimization challenges, allowing for a comprehensive evaluation of algorithm performance.

## 3. How to Run the Algorithm

This codebase is built as an extension of the [EvoX](https://github.com/EMI-Group/evox) framework. To set up the environment, please refer to the EvoX README for detailed instructions.

To run a test demo of our algorithm, follow these steps:

1. **Environment Setup:** Ensure you have the EvoX environment configured correctly, including the necessary GPU drivers and CUDA toolkit.
2. **Navigate to the test file:** Locate the `testDCMOEA.py` file in the `evox/unit_test/test/` directory.
3. **Run the test:** Execute the test file. The default configuration runs the TDCEA algorithm on the DCP test problems.

```bash
python evox/unit_test/test/testDCMOEA.py
```

You can modify the test file to experiment with different algorithms and test problems.

## 4. Acknowledgement

Thank you for your interest in our work. We hope this codebase proves valuable for your research and applications in dynamic constrained multi-objective optimization.

Once the review process is complete, citation information will be made available here.
