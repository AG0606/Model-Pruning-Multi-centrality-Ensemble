# Enhanced Neural Network Pruning via Multi-Centrality Graph Analysis

This repository contains the code and documentation for an advanced approach to neural network pruning using graph-theoretic centrality measures. By modeling neural network layers as directed bipartite graphs, this project identifies and removes structurally redundant neurons and convolutional filters while preserving predictive accuracy.

## Project Overview

Traditional neural network pruning often relies on simple magnitude-based heuristics or localized metrics like degree centrality. This project extends the graph-centrality-based pruning paradigm with a more robust, globally-aware algorithm. 

### Key Contributions & Innovations

1. **Multi-Centrality Ensemble:** 
   Instead of relying solely on local connectivity (Degree Centrality), this approach fuses Degree, Betweenness, and Eigenvector centrality into a single, unified importance score $S(v)$. This ensures that neurons acting as critical information bottlenecks (bridges) or those in high-prestige neighborhoods are preserved.
2. **Iterative Pruning with Annealing:** 
   One-shot pruning can severely disrupt network representations. This implementation uses a multi-round iterative pruning algorithm with an exponentially decaying threshold (annealing schedule), allowing the surviving network to adapt gracefully through fine-tuning.
3. **Channel-Level Abstraction:** 
   The algorithm natively supports modern convolutional architectures by treating entire `Conv2d` filters as nodes. The spatial dimensions are collapsed using an $L_2$-norm, allowing the bipartite graph analysis to prune complete channels without altering the underlying centrality algorithm.
4. **Cross-Dataset Validation:** 
   The pruning pipeline is comprehensively validated across datasets of varying visual complexity (MNIST, Fashion-MNIST, and CIFAR-10) using separate specialized architectures (`GrayCNN` and `ColorCNN`).

## Repository Structure

* **`02_enhanced_pruning.ipynb`**
  The core notebook containing the final, enhanced implementation. It includes the dataset factory, model architectures (GrayCNN/ColorCNN), multi-centrality ensemble algorithm, iterative pruning pipeline, cross-dataset validation, and a comprehensive visualization dashboard.
* **`01_baseline_pruning.ipynb`**
  The baseline case study implementation. This notebook introduces the fundamental concept of reframing fully-connected layers as bipartite graphs and applying basic Degree Centrality for pruning.
* **`Project_Paper_IEEE.pdf`**
  The official research paper built on the work in these two notebooks. It details the methodology, theoretical background, and the results achieved through this enhanced multi-centrality graph pruning algorithm.

## Technical Details

### Bipartite Graph Construction
* **Input Nodes:** $V_{in}$ (Input channels/neurons)
* **Output Nodes:** $V_{out}$ (Output filters/neurons)
* **Edges:** Connections are weighted based on absolute weight magnitude (for FC layers) or the $L_2$-norm of the filter slice (for Conv layers). A dynamic percentile threshold is used to maintain sparsity and computational tractability.

### The Ensemble Score $S(v)$
For each node $v$ in the output partition, the algorithm calculates:
* **Degree Centrality ($C_D$)** - Fast, local connectivity measure.
* **Betweenness Centrality ($C_B$)** - Global bottleneck detection via Brandes' algorithm.
* **Eigenvector Centrality ($C_E$)** - Recursive neighborhood prestige.

The final score is a weighted, min-max normalized combination of the three. Neurons/filters with the lowest $S(v)$ are selected for pruning.

### Performance Highlights
As demonstrated in the `02_enhanced_pruning.ipynb` dashboard, the iterative annealing schedule allows the network to maintain near-baseline accuracy even after significant parameter reduction, adapting seamlessly across different dataset complexities.

## Requirements

To run the notebooks, the following dependencies are required:
* Python 3.10+
* PyTorch & Torchvision
* NetworkX
* Matplotlib
* NumPy
* SciPy

## Getting Started

1. Clone this repository.
2. Ensure you have the required dependencies installed (e.g., via `pip install torch torchvision networkx matplotlib numpy scipy`).
3. Open `02_enhanced_pruning.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the cells sequentially to observe dataset EDA, baseline training, centrality analysis, the iterative pruning process, and the final summary dashboard.
