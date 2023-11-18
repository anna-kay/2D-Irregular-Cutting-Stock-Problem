# 2d-irregular-cutting-stock-problem

## Problem Description

The **2D bin-cutting** or **bin-packing problem** is a challenging optimization problem that often arises in logistics, manufacturing, and resource allocation scenarios.
In the case of irregular shapes within a heterogeneous bin, this problem involves efficiently placing a variety of irregularly shaped items into one or more bins of different sizes and shapes while minimizing wasted space and possibly other cost-related objectives.

The following images represent our instance of the 2d-irregular cutting stock problem:

Stock (bins of different sizes)
![stock](https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/f474e054-20c7-4046-b475-04ec8526c993)

3 orders (of irregularly shaped items/pieces) to be executed sequentially
![orders](https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/2339c9b8-7f08-421c-bbdd-0f1584eed74c)


Solved using a **two-stage implementation**:  
      **Main stage**: Metaheuristic Algorithm; Particle Swarm Optimization or Differential Evolution  
      **Auxiliary stage**: Heuristic placement routine (e.g. bottom left fit)

**Particle Swarm Optimization** (PSO) is a computational optimization technique inspired by the social behavior of birds and fish. 
It is a population-based, stochastic search algorithm that is used to find approximate solutions to optimization and search problems.

**Differential Evolution** (DE) is a computational optimization technique inspired by the process of natural evolution, particularly the mechanisms of selection, mutation, and crossover that occur in biological evolution.
It is considered an evolutionary algorithm, which is a class of optimization techniques that draws inspiration from the principles of natural selection and genetic evolution to solve complex problems.

## Solution Design Choices

It was assumed that the stock pieces (Stock) are placed one after the other in an approximately square space, 20x20.

![search_space](https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/4c23cde7-fdfe-41f4-9f0b-1a2b3fa5d324)

This choice is convenient for setting the limits of the variables involved and the limits of the search space.

### Encoding - Genotype & Phenotype 

In the case of the PSO, each particle (member of the population) encodes a full solution of the problem, i.e. it comprises a number of triplets (coordinates x & y and the angle θ) that are necesary for the definition of the shift (by x & y) and the rotation of each of the polygons.
The number of the triplets of the particle is equal to the number of the items of the current order.

The mapping of the x & y values of the triplets to the x & y values by which the polygon will be shifted is direct.

For the angle θ, given that most of the polygons of this problem set are rectangles, and all of them, apart from the scalene triangle, have at least one line of summetry, it was considered sufficient to limit the rotations of the polygons to 0 or 90 degrees.

Thus the search space has been defined as the 3xn-D space, given the number n of the polygons that have to be placed, with the first and the second variable of each triplet belonging to [0, 20], and the third always set to the ones of the values of {0, 90}.

The exact same approach is adopted for DEGL with each agent (member of the population) encoding a full solution of the problem.

### Two-stage Solution

**Main stage**: Metaheuristic Algorithm; Particle Swarm Optimization or Differential Evolution 
In this stage, each order piece (polygon) is assigned to a stock piece (stock bin).
**Auxiliary stage**: Bottom Left Fill Heuristic Placement
For each stock piece, the order items are placed (and subsequently "cut") one-by-one in the most bottom-left point of the stock piece that it can fit to. 
More particularly, a greedy approach is adopted; the order items are sorted in descending order according to their area, and are placed one-by-one in the most bottom-left point after trying out two potential rotations of the polygon (0 and 90 degrees clockwise).
The rotation that gives that results in the higher smoothness of the polygon is chosen. If both rotations result in the same smoothness, the rotation with the smaller height is chosen.
The process is repeated for all of the stock pieces. If it is not possible to fit all the items following this approach, the items are left exaclty as suggested by the metaheuristic algorithm.

### Objective Function

A single-objective multiple criteria objective function, following the general formalism:

<img width="523" alt="objective_function_formalism" src="https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/55e1e06c-2f8f-4351-bdd8-e10952807ed1">

, where <img width="60" alt="a" src="https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/2ff238d6-5c59-4f9e-ad39-cd95c886780f"> and <img width="88" alt="sum" src="https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/46acc290-b83f-48a4-88e3-a8e896fd2c7b"> , was chosen.  

The selected criteria can be organized into two groups:
1. Criteria that search for an acceptable solution:

**Area of the overlap of the polygons of the order** (area overlap based on the solution given by the particle). The computation of the overlap is done per pair of polygons, taking into account all the possible pairs, and summing the areas.
**Area of the polygons that stays out of the stock**: The total area of the polygons that either protrudes outs of a stock piece or was not placed in one of the stock pieces.

2. Criterium that leads to the optimization of the solution

**Utilization Ratio**

It was considered that the minimization of the areas (group 1), and the maximization of utilization ratio are not contradictory, and thus can be combined in a single objcetive function:

![objective_function](https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/24e46790-5623-4672-a938-c9e6a47c8b3e)

, where:

intersectingArea is the area of the overlap of the polygons of the order,
sumPenaltyArea is the area of the polygons that stays out of the stock, and
sumOfBinUtilizationRatios the sum of the utilization ratios of the stock bins computed as follows:

![utilizationRatios](https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/2574a7a6-806f-46b0-80db-e862ad4fa15a)

(the multiplication by ![fraction](https://github.com/anna-kay/2D-Irregular-Cutting-Stock-Problem/assets/56791604/ffb05aae-2a2e-4de5-8a4f-09d70bc705bc) is performed in order to "encourage" the algorithm to fill the smaller bins of the stock, the value 2.35 was chosen empirically)

The objective was the **minimization** of the value of the function.

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
