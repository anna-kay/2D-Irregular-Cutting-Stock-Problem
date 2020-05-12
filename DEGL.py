"""
Agent population minimization algorithm with dynamic random neighborhood topology.

The DEGL class implements a (somewhat) simplified version of the agentpopulation algorithm from MATLAB's 
Global Optimization Toolbox.
"""

import numpy as np
import warnings
# from itertools import cycle

try:
    from joblib import Parallel, delayed
    import multiprocessing
    HaveJoblib = True
except ImportError:
    HaveJoblib = False



class DEGL:
    """ Agent population minimization algorithm with dynamic random neighborhood topology.
        
        degl = DEGL(ObjectiveFcn, nVars, ...) creates the DEGL object stored in variable degl and 
            performs all population initialization tasks (including calling the output function once, if provided).
        
        degl.optimize() subsequently runs the whole iterative process.
        
        After initialization, the degl object has the following properties that can be queried (also during the 
            iterative process through the output function):
            o All the arguments passed during boject (e.g., degl.MaxIterations, degl.ObjectiveFcn,  degl.LowerBounds, 
                etc.). See the documentation of the __init__ member below for supported options and their defaults.
            o Iteration: the current iteration. Its value is -1 after initialization 0 or greater during the iterative
                process.
            o Population: the current iteration population (nAgents x nVars)
            o Velocity: the current velocity vectors (nAgents x nVars)
            o CurrentPopulationFitness: the current population's fitnesses for all agents (nAgents x 1)
            o PreviousBestPosition: the best-so-far positions found for each individual (nAgents x nVars)
            o PreviousBestFitness: the fitnesses of the best-so-far individuals (nAgents x 1)
            o GlobalBestFitness: the overall best fitness attained found from the beginning of the iterative process
            o GlobalBestPosition: the overall best position found from the beginning of the iterative process
            o AdaptiveNeighborhoodSize: the current neighborhood size
            o MinNeighborhoodSize: the minimum neighborhood size allowed
            o AdaptiveInertia: the current value of the inertia weight
            o StallCounter: the stall counter value (for updating inertia)
            o StopReason: string with the stopping reason (only available at the end, when the algorithm stops)
            o GlobalBestSoFarFitnesses: a numpy vector that stores the global best-so-far fitness in each iteration. 
                Its size is MaxIterations+1, with the first element (GlobalBestSoFarFitnesses[0]) reserved for the best
                fitness value of the initial population. Accordingly, degl.GlobalBestSoFarFitnesses[degl.Iteration+1] stores 
                the global best fitness at iteration degl.Iteration. Since the global best-so-far is updated only if 
                lower that the previously stored, this is a non-strictly decreasing function. It is initialized with 
                NaN values and therefore is useful for plotting it, as the ydata of the matplotlib line object (NaN 
                values are just not plotted). In the latter case, the xdata would have to be set to 
                np.arange(degl.MaxIterations+1)-1, so that the X axis starts from -1.
    """
    
    
    def __init__( self
                , ObjectiveFcn
                , nVars
                , LowerBounds = None
                , UpperBounds = None
                , PopulationSize = None
                , CR = 0.9 # crossover probability [0.4-0.9] # 0.6 -> good resultls as well
                , aScaleFactor  = 0.8 # a good inital choice is 0.5 # 2.05 #
                , bScaleFactor = 0.8 # 2.05 # [0.4 - 0.95]
                , minWeight =  0.4 # 0.4 #
                , maxWeight =  0.8 # 0.9 #
                , NeighborhoodFraction = 0.1
                , FunctionTolerance = 1.0e-6
                , MaxIterations = None
                , MaxStallIterations = 20
                , OutputFcn = None
                , UseParallel = False
                ):
        """ The object is initialized with two mandatory positional arguments:
                o ObjectiveFcn: function object that accepts a vector (the agent) and returns the scalar fitness 
                                value, i.e., FitnessValue = ObjectiveFcn(Agent)
                o nVars: the number of problem variables
            The algorithm tries to minimize the ObjectiveFcn.
            
            The arguments LowerBounds & UpperBounds lets you define the lower and upper bounds for each variable. They 
            must be either scalars or vectors/lists with nVars elements. If not provided, LowerBound is set to -1000 
            and UpperBound is set to 1000 for all variables. If vectors are provided and some of its elements are not 
            finite (NaN or +-Inf), those elements are also replaced with +-1000 respectively.
            
            The rest of the arguments are the algorithm's options:
                o PopulationSize (default:  min(100,10*nVars)): Number of agents in the population, an integer greater than 1.
                o aScaleFactor (default: 1.49): Weighting (finite scalar) of each agent’s best position when
                    adjusting velocity.
                o bScaleFactor (default: 1.49): Weighting (finite scalar) of the neighborhood’s best position 
                    when adjusting velocity.
                o WeightRange (default: [0.1, 1.1]): Two-element real vector with same sign values in increasing 
                    order. Gives the lower and upper bound of the adaptive inertia. To obtain a constant (nonadaptive) 
                    inertia, set both elements of WeightRange to the same value.
                o NeighborhoodFraction (default: 0.25): Minimum adaptive neighborhood size, a scalar in [0, 1].
                o FunctionTolerance (default: 1e-6): Iterations end when the relative change in best objective function 
                    value over the last MaxStallIterations iterations is less than options.FunctionTolerance.
                o MaxIterations (default: 200*nVars): Maximum number of iterations.
                o MaxStallIterations (default: 20): Iterations end when the relative change in best objective function 
                    value over the last MaxStallIterations iterations is less than options.FunctionTolerance.
                o OutputFcn (default: None): Output function, which is called at the end of each iteration with the 
                    iterative data and they can stop the solver. The output function must have the signature 
                    stop = fun(degl), returning True if the iterative process must be terminated. degl is the 
                    DEGL object (self here). The output function is also called after population initialization 
                    (i.e., within this member function).
                o UseParallel (default: False): Compute objective function in parallel when True. The latter requires
                    package joblib to be installed (i.e., pip install joplib or conda install joblib).

        """
        self.ObjectiveFcn = ObjectiveFcn
        self.nVars = nVars
        
        # assert options validity (simple checks only) & store them in the object
        if PopulationSize is None:
            self.PopulationSize = min(200, 10*nVars)
        else:
            assert np.isscalar(PopulationSize) and PopulationSize > 1, \
                "The PopulationSize option must be a scalar integer greater than 1."
            self.PopulationSize = max(2, int(round(PopulationSize)))
        
        #######################################################################
        
        assert np.isscalar(CR) and CR >= 0.0 and CR <= 1.0, \
                "The CR option must be a scalar number in the range [0,1]."
        self.CR = CR
        
        #######################################################################        
        
        assert np.isscalar(aScaleFactor), "The aScaleFactor option must be a scalar number."
        self.aScaleFactor = aScaleFactor
        assert np.isscalar(bScaleFactor), "The bScaleFactor option must be a scalar number."
        self.bScaleFactor = bScaleFactor
        
        assert np.isscalar(minWeight), "The minWeight option must be a scalar number."
        self.minWeight = minWeight
        assert np.isscalar(maxWeight), "The maxWeight option must be a scalar number."
        self.maxWeight = maxWeight
        
        
        assert np.isscalar(NeighborhoodFraction) and NeighborhoodFraction >= 0.0 and NeighborhoodFraction <= 1.0, \
                "The NeighborhoodFraction option must be a scalar number in the range [0,1]."
        self.NeighborhoodFraction = NeighborhoodFraction
        
        assert np.isscalar(FunctionTolerance) and FunctionTolerance >= 0.0, \
                "The FunctionTolerance option must be a scalar number greater or equal to 0."
        self.FunctionTolerance = FunctionTolerance
        
        if MaxIterations is None:
            self.MaxIterations = 100*nVars
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
            warnings.warn("""If UseParallel is set to True, it requires the joblib package that could not be imported; population objective values will be computed in serial mode instead.""")
            self.UseParallel = False
        
        # degl initialization: store everything into a self, which is also used be OutputFcn
        nAgents = self.PopulationSize
        
        # Initial population: randomly in [lower,upper] and if any is +-Inf in [-1000, 1000]
        lbMatrix = np.tile(self.LowerBounds, (nAgents, 1))
        ubMatrix = np.tile(self.UpperBounds, (nAgents, 1))
        bRangeMatrix = ubMatrix - lbMatrix
        # Random initialization of the population
        self.Population = lbMatrix + np.random.rand(nAgents,nVars) * bRangeMatrix
#
#        f = open("population.txt", "a")
#        print(self.Population, file=f)
#        print("next", file=f)
#        f.close()
        
        # Initial fitness
        self.CurrentPopulationFitness = np.zeros(nAgents)
        self.__evaluatePopulation()
        
        # Initial best-so-far individuals and global best
        self.PreviousBestPosition = self.Population.copy()
        self.PreviousBestFitness = self.CurrentPopulationFitness.copy()
        
        bInd = self.CurrentPopulationFitness.argmin()
        self.GlobalBestFitness = self.CurrentPopulationFitness[bInd].copy()
        self.GlobalBestPosition = self.PreviousBestPosition[bInd, :].copy()
                # improvement stopping criterion.

        # iteration counter starts at -1, meaning initial population
        self.Iteration = -1;
        
        # Neighborhood radius 
        self.NeighborhoodRadius = max(1, int(np.floor(nAgents * self.NeighborhoodFraction)));       

        # Keep the global best of each iteration as an array initialized with NaNs. First element is for initial population,
        # so it has self.MaxIterations+1 elements. Useful for output functions, but is also used for the insignificant
        self.GlobalBestSoFarFitnesses = np.zeros(self.MaxIterations+1)
        self.GlobalBestSoFarFitnesses.fill(np.nan)
        self.GlobalBestSoFarFitnesses[0] = self.GlobalBestFitness
        
        self.LocalMutation = lbMatrix + np.zeros((nAgents,nVars)) * bRangeMatrix
        self.GlobalMutation = lbMatrix + np.zeros((nAgents,nVars)) * bRangeMatrix
        self.CombinedMutation = lbMatrix + np.zeros((nAgents,nVars)) * bRangeMatrix
        
        # call output function, but neglect the returned stop flag
        if self.OutputFcn:
            self.OutputFcn(self)
    
    
    def __evaluatePopulation(self):
        """ Helper private member function that evaluates the population, by calling ObjectiveFcn either in serial or
            parallel mode, depending on the UseParallel option during initialization.
        """
        nAgents = self.PopulationSize
        if self.UseParallel:
            nCores = multiprocessing.cpu_count()
            self.CurrentPopulationFitness[:] = Parallel(n_jobs=nCores)( 
                    delayed(self.ObjectiveFcn)(self.Population[i,:]) for i in range(nAgents) )
        else:
            self.CurrentPopulationFitness[:] = [self.ObjectiveFcn(self.Population[i,:]) for i in range(nAgents)]
    
        
    def optimize( self ):
        """ Runs the iterative process on the initialized population. """
                
        nAgents = self.PopulationSize
        nVars = self.nVars
        NeighborhoodRadius = self.NeighborhoodRadius
        NeighborhoodSize = 2*NeighborhoodRadius+1;
        a = self.aScaleFactor
        b = self.bScaleFactor
        minWeight = self.minWeight
        maxWeight = self.maxWeight
                
        # start the iteration
        doStop = False
        
        while not doStop:
            self.Iteration += 1
            
            W = minWeight + (maxWeight - minWeight)*(self.Iteration - 1)/(self.MaxIterations - 1)
            
            for i in range(nAgents):              

                # COMPUTE LOCAL MUTATION
                # 1. find neighbors indices              
                             
                neighborsIndices = list(range(i-NeighborhoodRadius, i+NeighborhoodRadius+1))
                
                neighborsIndices = [index%nAgents for index in neighborsIndices]
                
                neighbors = self.Population[neighborsIndices].copy()    

#                f = open("neighborsIndices.txt", "a")
#                print("these are the indices", neighborsIndices ,file=f)
#                f.close()                
                                       
                # 2. find best local, using GLOBAL indexing!!!   
                bLocalInd = self.PreviousBestFitness[neighborsIndices].argmin()
                bestNeighbor = self.Population[bLocalInd]
                                
                # 3. select two random indices from [i-NeighborhoodRadius, i+NeighborhoodRadius]                
                indicesToChooseFrom = np.arange(NeighborhoodSize)
                # TODO: check numbers!!!
                indicesToChooseFrom = list(filter(lambda x : x!= NeighborhoodRadius, indicesToChooseFrom)) # remove i
                p = np.random.choice(indicesToChooseFrom, 1)
                indicesToChooseFrom = list(filter(lambda x : x!= p, indicesToChooseFrom))
                q = np.random.choice(indicesToChooseFrom, 1)

                # 4. compute the mutation
                self.LocalMutation[i,:] = self.Population[i,:] + a*(bestNeighbor-self.Population[i,:]) + b*(neighbors[p,:]-neighbors[q,:])
                
                # COMPUTE GLOBAL MUTATION
                # 1. select two random indices from the population
                indicesToChooseFrom = np.arange(nAgents)
                indicesToChooseFrom = list(filter(lambda x : x!= i, indicesToChooseFrom))
                r1 = np.random.choice(nAgents, 1)
                indicesToChooseFrom = list(filter(lambda x : x!= r1, indicesToChooseFrom))
                r2 = np.random.choice(nAgents, 1)

                # 2. compute the mutation
                self.GlobalMutation[i,:] = self.Population[i,:] + a*(self.GlobalBestPosition-self.Population[i,:]) + b*(self.Population[r1,:]-self.Population[r2,:])
                
                # COMPUTE COMBINED MUTATION
                self.CombinedMutation[i,:] = W*self.GlobalMutation[i,:] + (1-W)*self.LocalMutation[i,:]
              
                # COMPUTE CROSSOVER
                crossoverPositions = np.random.rand(nVars)
                for j in range(nVars):
                    if (crossoverPositions[j]<self.CR):
                        self.Population[i,j] = self.CombinedMutation[i,j]
                
                # check bounds violation
                posInvalid = self.Population[i,:] < self.LowerBounds
                self.Population[i,posInvalid] = self.LowerBounds[posInvalid]
                
                posInvalid = self.Population[i,:] > self.UpperBounds
                self.Population[i,posInvalid] = self.UpperBounds[posInvalid]
                
#                print("This is the ", i, " agent", self.Population[i])
            
            # calculate new fitness & update best
            self.__evaluatePopulation()
            agentsProgressed = self.CurrentPopulationFitness < self.PreviousBestFitness
            self.PreviousBestPosition[agentsProgressed, :] = self.Population[agentsProgressed, :]
            self.PreviousBestFitness[agentsProgressed] = self.CurrentPopulationFitness[agentsProgressed]
            
            newBestInd = self.CurrentPopulationFitness.argmin()
            newBestFit = self.CurrentPopulationFitness[newBestInd]
            
            if newBestFit < self.GlobalBestFitness:
                self.GlobalBestFitness = newBestFit
                self.GlobalBestPosition = self.Population[newBestInd, :].copy()
  
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
        
            
