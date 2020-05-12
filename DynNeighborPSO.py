# -*- coding: utf-8 -*-
"""
Particle swarm minimization algorithm with dynamic random neighborhood topology.

The DynNeighborPSO class implements a (somewhat) simplified version of the particleswarm algorithm from MATLAB's 
Global Optimization Toolbox.
"""

import numpy as np
import warnings

try:
    from joblib import Parallel, delayed
    import multiprocessing
    HaveJoblib = True
except ImportError:
    HaveJoblib = False


class DynNeighborPSO:
    """ Particle swarm minimization algorithm with dynamic random neighborhood topology.
        
        pso = DynNeighborPSO(ObjectiveFcn, nVars, ...) creates the DynNeighborPSO object stored in variable pso and 
            performs all swarm initialization tasks (including calling the output function once, if provided).
        
        pso.optimize() subsequently runs the whole iterative process.
        
        After initialization, the pso object has the following properties that can be queried (also during the 
            iterative process through the output function):
            o All the arguments passed during boject (e.g., pso.MaxIterations, pso.ObjectiveFcn,  pso.LowerBounds, 
                etc.). See the documentation of the __init__ member below for supported options and their defaults.
            o Iteration: the current iteration. Its value is -1 after initialization 0 or greater during the iterative
                process.
            o Swarm: the current iteration swarm (nParticles x nVars)
            o Velocity: the current velocity vectors (nParticles x nVars)
            o CurrentSwarmFitness: the current swarm's fitnesses for all particles (nParticles x 1)
            o PreviousBestPosition: the best-so-far positions found for each individual (nParticles x nVars)
            o PreviousBestFitness: the fitnesses of the best-so-far individuals (nParticles x 1)
            o GlobalBestFitness: the overall best fitness attained found from the beginning of the iterative process
            o GlobalBestPosition: the overall best position found from the beginning of the iterative process
            o AdaptiveNeighborhoodSize: the current neighborhood size
            o MinNeighborhoodSize: the minimum neighborhood size allowed
            o AdaptiveInertia: the current value of the inertia weight
            o StallCounter: the stall counter value (for updating inertia)
            o StopReason: string with the stopping reason (only available at the end, when the algorithm stops)
            o GlobalBestSoFarFitnesses: a numpy vector that stores the global best-so-far fitness in each iteration. 
                Its size is MaxIterations+1, with the first element (GlobalBestSoFarFitnesses[0]) reserved for the best
                fitness value of the initial swarm. Accordingly, pso.GlobalBestSoFarFitnesses[pso.Iteration+1] stores 
                the global best fitness at iteration pso.Iteration. Since the global best-so-far is updated only if 
                lower that the previously stored, this is a non-strictly decreasing function. It is initialized with 
                NaN values and therefore is useful for plotting it, as the ydata of the matplotlib line object (NaN 
                values are just not plotted). In the latter case, the xdata would have to be set to 
                np.arange(pso.MaxIterations+1)-1, so that the X axis starts from -1.
    """
    
    
    def __init__( self
                , ObjectiveFcn
                , nVars
                , LowerBounds = None
                , UpperBounds = None
                , SwarmSize = None
                , SelfAdjustmentWeight  = 2.05 
                , SocialAdjustmentWeight = 2.05 
                , InertiaRange = [0.4, 0.9] 
                , MinNeighborsFraction = 0.5
                , FunctionTolerance = 1.0e-6
                , MaxIterations = None
                , MaxStallIterations = 20
                , OutputFcn = None
                , UseParallel = False
                ):
        """ The object is initialized with two mandatory positional arguments:
                o ObjectiveFcn: function object that accepts a vector (the particle) and returns the scalar fitness 
                                value, i.e., FitnessValue = ObjectiveFcn(Particle)
                o nVars: the number of problem variables
            The algorithm tries to minimize the ObjectiveFcn.
            
            The arguments LowerBounds & UpperBounds lets you define the lower and upper bounds for each variable. They 
            must be either scalars or vectors/lists with nVars elements. If not provided, LowerBound is set to -1000 
            and UpperBound is set to 1000 for all variables. If vectors are provided and some of its elements are not 
            finite (NaN or +-Inf), those elements are also replaced with +-1000 respectively.
            
            The rest of the arguments are the algorithm's options:
                o SwarmSize (default:  min(100,10*nVars)): Number of particles in the swarm, an integer greater than 1.
                o SelfAdjustmentWeight (default: 1.49): Weighting (finite scalar) of each particle’s best position when
                    adjusting velocity.
                o SocialAdjustmentWeight (default: 1.49): Weighting (finite scalar) of the neighborhood’s best position 
                    when adjusting velocity.
                o InertiaRange (default: [0.1, 1.1]): Two-element real vector with same sign values in increasing 
                    order. Gives the lower and upper bound of the adaptive inertia. To obtain a constant (nonadaptive) 
                    inertia, set both elements of InertiaRange to the same value.
                o MinNeighborsFraction (default: 0.25): Minimum adaptive neighborhood size, a scalar in [0, 1].
                o FunctionTolerance (default: 1e-6): Iterations end when the relative change in best objective function 
                    value over the last MaxStallIterations iterations is less than options.FunctionTolerance.
                o MaxIterations (default: 200*nVars): Maximum number of iterations.
                o MaxStallIterations (default: 20): Iterations end when the relative change in best objective function 
                    value over the last MaxStallIterations iterations is less than options.FunctionTolerance.
                o OutputFcn (default: None): Output function, which is called at the end of each iteration with the 
                    iterative data and they can stop the solver. The output function must have the signature 
                    stop = fun(pso), returning True if the iterative process must be terminated. pso is the 
                    DynNeighborPSO object (self here). The output function is also called after swarm initialization 
                    (i.e., within this member function).
                o UseParallel (default: False): Compute objective function in parallel when True. The latter requires
                    package joblib to be installed (i.e., pip install joplib or conda install joblib).

        """
        self.ObjectiveFcn = ObjectiveFcn
        self.nVars = nVars
        
        # assert options validity (simple checks only) & store them in the object
        if SwarmSize is None:
            self.SwarmSize = min(100, 10*nVars)
        else:
            assert np.isscalar(SwarmSize) and SwarmSize > 1, \
                "The SwarmSize option must be a scalar integer greater than 1."
            self.SwarmSize = max(2, int(round(SwarmSize)))
        
        assert np.isscalar(SelfAdjustmentWeight), "The SelfAdjustmentWeight option must be a scalar number."
        self.SelfAdjustmentWeight = SelfAdjustmentWeight
        assert np.isscalar(SocialAdjustmentWeight), "The SocialAdjustmentWeight option must be a scalar number."
        self.SocialAdjustmentWeight = SocialAdjustmentWeight
        
        assert len(InertiaRange) == 2, "The InertiaRange option must be a vector with 2 elements."
        self.InertiaRange = np.array(InertiaRange, dtype=float)
        self.InertiaRange.sort()
        assert np.isscalar(MinNeighborsFraction) and MinNeighborsFraction >= 0.0 and MinNeighborsFraction <= 1.0, \
                "The MinNeighborsFraction option must be a scalar number in the range [0,1]."
        self.MinNeighborsFraction = MinNeighborsFraction
        
        assert np.isscalar(FunctionTolerance) and FunctionTolerance >= 0.0, \
                "The FunctionTolerance option must be a scalar number greater or equal to 0."
        self.FunctionTolerance = FunctionTolerance
        
        if MaxIterations is None:
            self.MaxIterations = 200*nVars
        else:
            assert np.isscalar(MaxIterations), "The MaxIterations option must be a scalar integer greater than 0."
            self.MaxIterations = max(1, int(round(MaxIterations)))
        assert np.isscalar(MaxStallIterations), \
            "The MaxStallIterations option must be a scalar integer greater than 0."
        self.MaxStallIterations = max(1, int(round(MaxStallIterations)))
        
        self.OutputFcn = OutputFcn
        assert np.isscalar(UseParallel) and (isinstance(UseParallel,bool) or isinstance(UseParallel,np.bool_)), \
            "The UseParallel option must be a scalar boolean value."
        self.UseParallel = UseParallel
        
        # lower bounds
        if LowerBounds is None:
            self.LowerBounds = -1000.0 * np.ones(nVars)
        elif np.isscalar(LowerBounds):
            self.LowerBounds = LowerBounds * np.ones(nVars)
        else:
            self.LowerBounds = np.array(LowerBounds, dtype=float)
        self.LowerBounds[~np.isfinite(self.LowerBounds)] = -1000.0
        assert len(self.LowerBounds) == nVars, \
            "When providing a vector for LowerBounds its number of element must equal the number of problem variables."
        # upper bounds
        if UpperBounds is None:
            self.UpperBounds = 1000.0 * np.ones(nVars)
        elif np.isscalar(UpperBounds):
            self.UpperBounds = UpperBounds * np.ones(nVars)
        else:
            self.UpperBounds = np.array(UpperBounds, dtype=float)
        self.UpperBounds[~np.isfinite(self.UpperBounds)] = 1000.0
        assert len(self.UpperBounds) == nVars, \
            "When providing a vector for UpperBounds its number of element must equal the number of problem variables."
        
        assert np.all(self.LowerBounds <= self.UpperBounds), \
            "Upper bounds must be greater or equal to lower bounds for all variables."
        
        
        # check that we have joblib if UseParallel is True
        if self.UseParallel and not HaveJoblib:
            warnings.warn("""If UseParallel is set to True, it requires the joblib package that could not be imported; swarm objective values will be computed in serial mode instead.""")
            self.UseParallel = False
        
        # PSO initialization: store everything into a self, which is also used be OutputFcn
        nParticles = self.SwarmSize
        
        # Initial swarm: randomly in [lower,upper] and if any is +-Inf in [-1000, 1000]
        lbMatrix = np.tile(self.LowerBounds, (nParticles, 1))
        ubMatrix = np.tile(self.UpperBounds, (nParticles, 1))
        bRangeMatrix = ubMatrix - lbMatrix
        
        #
        #   RANDOMLY INITIALIZE D PARTICLES
        #
        self.Swarm = lbMatrix + np.random.rand(nParticles,nVars) * bRangeMatrix
        #self.Swarm = np.around(self.Swarm, decimals=3)
        
        #
        #   RANDOMLY INITIALIZE EQUIVALENT VELOCITY VECTORS
        #
        # Initial velocity: random in [-(UpperBound-LowerBound), (UpperBound-LowerBound)]
        self.Velocity = -bRangeMatrix + 2.0 * np.random.rand(nParticles,nVars) * bRangeMatrix
        
        #
        #   CALCULATE INITIAL FITNESS 
        #
        # Initial fitness
        self.CurrentSwarmFitness = np.zeros(nParticles)
        self.__evaluateSwarm()
        
        
        #
        #   INITIALIZATION of PreviousBestPosition matrix with current values of our swarm,
        #   since they are the only ones available
        #
        # Initial best-so-far individuals and global best
        self.PreviousBestPosition = self.Swarm.copy() # size = numberOfParticles (swarmSize) * particleSize
        self.PreviousBestFitness = self.CurrentSwarmFitness.copy()
        
        # bInd -> index of the best particle of this swarm
        bInd = self.CurrentSwarmFitness.argmin() 
        self.GlobalBestFitness = self.CurrentSwarmFitness[bInd].copy()
        self.GlobalBestPosition = self.PreviousBestPosition[bInd, :].copy()
        
        # iteration counter starts at -1, meaning initial population
        self.Iteration = -1;
        
        # Initial neighborhood & inertia
        self.MinNeighborhoodSize = max(2, int(np.floor(nParticles * self.MinNeighborsFraction)));
        self.AdaptiveNeighborhoodSize = self.MinNeighborhoodSize;
        
        if np.all(self.InertiaRange >= 0):
            self.AdaptiveInertia = self.InertiaRange[1]
        else:
            self.AdaptiveInertia = self.InertiaRange[0]
        
        self.StallCounter = 0;
        
        # Keep the global best of each iteration as an array initialized with NaNs. First element is for initial swarm,
        # so it has self.MaxIterations+1 elements. Useful for output functions, but is also used for the insignificant
        # improvement stopping criterion.
        self.GlobalBestSoFarFitnesses = np.zeros(self.MaxIterations+1)
        self.GlobalBestSoFarFitnesses.fill(np.nan)
        self.GlobalBestSoFarFitnesses[0] = self.GlobalBestFitness
        
        # call output function, but neglect the returned stop flag
        if self.OutputFcn:
            self.OutputFcn(self)
    
    
    def __evaluateSwarm(self):
        """ Helper private member function that evaluates the population, by calling ObjectiveFcn either in serial or
            parallel mode, depending on the UseParallel option during initialization.
        """
        nParticles = self.SwarmSize
        if self.UseParallel:
            nCores = multiprocessing.cpu_count()
            self.CurrentSwarmFitness[:] = Parallel(n_jobs=nCores)( 
                    delayed(self.ObjectiveFcn)(self.Swarm[i,:]) for i in range(nParticles) )
        else:
            self.CurrentSwarmFitness[:] = [self.ObjectiveFcn(self.Swarm[i,:]) for i in range(nParticles)]
    
        
    def optimize( self ):
        """ Runs the iterative process on the initialized swarm. """
        
        nParticles = self.SwarmSize
        nVars = self.nVars
                
        # start the iteration
        doStop = False
        selfWeight = self.SelfAdjustmentWeight
        socialWeight = self.SocialAdjustmentWeight
        
        while not doStop:
            self.Iteration += 1

            for p in range(nParticles):
                # find neighbors
                # randomly select neighbors according to AdaptiveNeighborhoodSize
                neighbors = np.random.choice( nParticles-1, size=self.AdaptiveNeighborhoodSize, replace=False)
                neighbors[neighbors >= p] += 1; # do not select itself, i.e., index p
                
                bInd = self.PreviousBestFitness[neighbors].argmin()
                bestNeighbor = neighbors[bInd]
                
                # update velocity
                randSelf = np.random.rand(nVars)
                randSocial = np.random.rand(nVars)
                self.Velocity[p,:] = self.AdaptiveInertia * self.Velocity[p,:] \
                    + selfWeight * randSelf * (self.PreviousBestPosition[p,:] - self.Swarm[p,:]) \
                    + socialWeight * randSocial * (self.PreviousBestPosition[bestNeighbor,:] - self.Swarm[p,:])
                
                # update position
                self.Swarm[p,:] += self.Velocity[p,:]
                
                # check bounds violation
                posInvalid = self.Swarm[p,:] < self.LowerBounds
                self.Swarm[p,posInvalid] = self.LowerBounds[posInvalid]
                self.Velocity[p,posInvalid] = 0.0
                
                posInvalid = self.Swarm[p,:] > self.UpperBounds
                self.Swarm[p,posInvalid] = self.UpperBounds[posInvalid]
                self.Velocity[p,posInvalid] = 0.0
            
            # calculate new fitness & update best
            #
            # particlesProgressed = ta particles exoun proxwrisei, auta pou
            #                       exoun proxwrisei exoun timi True
            self.__evaluateSwarm()
            particlesProgressed = self.CurrentSwarmFitness < self.PreviousBestFitness
            self.PreviousBestPosition[particlesProgressed, :] = self.Swarm[particlesProgressed, :]
            self.PreviousBestFitness[particlesProgressed] = self.CurrentSwarmFitness[particlesProgressed]
            
            # update global best, adaptive neighborhood size and stall counter
            newBestInd = self.CurrentSwarmFitness.argmin()
            newBestFit = self.CurrentSwarmFitness[newBestInd]
            
            if newBestFit < self.GlobalBestFitness:
                self.GlobalBestFitness = newBestFit
                self.GlobalBestPosition = self.Swarm[newBestInd, :].copy()
                
                self.StallCounter = max(0, self.StallCounter-1)
                self.AdaptiveNeighborhoodSize = self.MinNeighborhoodSize
                
                if self.StallCounter < 2:
                    self.AdaptiveInertia *= 2.0
                else:
                    self.AdaptiveInertia /= 2.0;
                
                self.AdaptiveInertia = max( self.InertiaRange[0], min(self.InertiaRange[1], self.AdaptiveInertia) )
            else:
                self.StallCounter += 1
                self.AdaptiveNeighborhoodSize = min(
                        self.AdaptiveNeighborhoodSize+self.MinNeighborhoodSize, nParticles-1 )
                
            # first element of self.GlobalBestSoFarFitnesses is for self.Iteration == -1
            self.GlobalBestSoFarFitnesses[self.Iteration+1] = self.GlobalBestFitness
            
            # run output function and stop if necessary
            if self.OutputFcn and self.OutputFcn(self):
                self.StopReason = 'OutputFcn requested to stop.'
                doStop = True
                continue
            
            # stop if max iterations
            if self.Iteration >= self.MaxIterations-1:
                self.StopReason = 'MaxIterations reached.'
                doStop = True
                continue
            
            # stop if insignificant improvement
            if self.Iteration > self.MaxStallIterations:
                # The minimum global best fitness is the one stored in self.GlobalBestSoFarFitnesses[self.Iteration+1]
                # (only updated if newBestFit is less than the previously stored). The maximum (may be equal to the 
                # current) is the one  in self.GlobalBestSoFarFitnesses MaxStallIterations before.
                minBestFitness = self.GlobalBestSoFarFitnesses[self.Iteration+1]
                maxPastBestFit = self.GlobalBestSoFarFitnesses[self.Iteration+1-self.MaxStallIterations]
                if (maxPastBestFit == 0.0) and (minBestFitness < maxPastBestFit):
                    windowProgress = np.inf  # don't stop
                elif (maxPastBestFit == 0.0) and (minBestFitness == 0.0):
                    windowProgress = 0.0  # not progressed
                else:
                    windowProgress = abs(minBestFitness - maxPastBestFit) / abs(maxPastBestFit)
                if windowProgress <= self.FunctionTolerance:
                    self.StopReason = 'Population did not improve significantly the last MaxStallIterations.'
                    doStop = True
            
        
        # print stop message
        print('Algorithm stopped after {} iterations. Best fitness attained: {}'.format(
                self.Iteration+1,self.GlobalBestFitness))
        print(f'Stop reason: {self.StopReason}')
        
            
