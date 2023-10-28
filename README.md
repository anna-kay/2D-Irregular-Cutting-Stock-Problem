# 2d-irregular-cutting-stock-problem

## Description

The **2D bin-cutting** or **bin-packing problem** is a challenging optimization problem that often arises in logistics, manufacturing, and resource allocation scenarios.
In the case of irregular shapes within a heterogeneous bin, this problem involves efficiently placing a variety of irregularly shaped items into one or more bins of different sizes and shapes while minimizing wasted space and possibly other cost-related objectives.

The following images represent our instance of the 2d-irregular cutting stock problem:

Stock (bins of different sizes)
![stock](https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/f474e054-20c7-4046-b475-04ec8526c993)

3 orders (of irregularly shaped items/pieces) to be executed sequentially
![orders](https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/2339c9b8-7f08-421c-bbdd-0f1584eed74c)


Solved using a **two-stage implementation**:  
      1. Metaheuristics; Particle Swarm Optimization or Differential Evolution  
      2. Heuristic placement routine (e.g. bottom left fit)

**Particle Swarm Optimization** (PSO) is a computational optimization technique inspired by the social behavior of birds and fish. 
It is a population-based, stochastic search algorithm that is used to find approximate solutions to optimization and search problems.

**Differential Evolution** (DE) is a computational optimization technique inspired by the process of natural evolution, particularly the mechanisms of selection, mutation, and crossover that occur in biological evolution.
It is considered an evolutionary algorithm, which is a class of optimization techniques that draws inspiration from the principles of natural selection and genetic evolution to solve complex problems.


## References and Literature

R. P. Abeysooriya, “Cutting Patterns for Efficient Production of Irregular-shaped
Pieces,” no. August, 2017.

Bansal et al., “Inertia Weight Strategies in Particle Swarm Inertia Weight Strategies in
Particle Swarm,” Pap. Conf. Technol. Inf. Kharagpur, Technol., no. May 2014, p. 7,
2011.

R. Dash and P. K. Dash, “A hybrid stock trading framework integrating technical
analysis with machine learning techniques,” J. Financ. Data Sci., vol. 2, no. 1, pp. 42–
57, 2016.

S. Das and S. Sil, “Kernel-induced fuzzy clustering of image pixels with an improved
differential evolution algorithm,” Inf. Sci. (Ny)., vol. 180, no. 8, pp. 1237–1256, 2010.

E. Hopper, “Two-dimensional Packing utilising Evolutionary Algorithms and other Meta-Heuristic Methods,” no. May, 2000.

https://www.youtube.com/watch?v=kHyNqSnzP8Y
