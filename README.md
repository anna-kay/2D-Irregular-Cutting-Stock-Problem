# 2d-irregular-cutting-stock-problem

2d bin-cutting/packing problem (irregular shape bin-cutting/packing in heterogeneous bin)

The **2D bin-cutting** or **bin-packing problem** is a challenging optimization problem that often arises in logistics, manufacturing, and resource allocation scenarios.
In the context of irregular shapes within a heterogeneous bin, this problem involves efficiently placing a variety of irregularly shaped items into one or more bins of different sizes and shapes while minimizing wasted space and possibly other cost-related objectives.

The following images represent our instance of the 2d-irregular cutting stock problem:

Stock (bins of different sizes)
![stock](https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/f474e054-20c7-4046-b475-04ec8526c993)

3 orders (irregularly shaped items) to be executed sequentially
![orders](https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/2339c9b8-7f08-421c-bbdd-0f1584eed74c)


Solved using a **two-stage implementation**:  
      1. Metaheuristics; Particle Swarm Optimization or Differential Evolution  
      2. Heuristic placement routine (e.g. bottom left fit)

**Particle Swarm Optimization** (PSO) is a computational optimization technique inspired by the social behavior of birds and fish. 
It is a population-based, stochastic search algorithm that is used to find approximate solutions to optimization and search problems.

**Differential Evolution** (DE) is a computational optimization technique inspired by the process of natural evolution, particularly the mechanisms of selection, mutation, and crossover that occur in biological evolution.
It is considered an evolutionary algorithm, which is a class of optimization techniques that draws inspiration from the principles of natural selection and genetic evolution to solve complex problems.


