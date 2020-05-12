# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:23:25 2020

@author: Anna
"""

import numpy as np
import matplotlib.pyplot as plt
import shapely
import shapely.ops
import itertools
import collections.abc
# import math 

from shapely.ops import cascaded_union

from DynNeighborPSO import DynNeighborPSO
from demo_shapely import plotShapelyPoly
from WoodProblemDefinition import Stock, Order1, Order2, Order3

joinStyle = shapely.geometry.JOIN_STYLE.mitre
capStyle = shapely.geometry.CAP_STYLE.square

###############################################################################
def smoothness(poly):
    
    a = 1.11
     
    if (poly.area==0):
        smoothness = 1
    else:
        criterium = (poly.convex_hull.area/poly.area) - 1.0        
        smoothness = 1.0/ (1.0 + a*criterium)
    
    return smoothness

###############################################################################

def ObjectiveFcn(particle):

    orderItems = myOrder.copy()
    currentStockUnion = myStock
    initialStock = myStock
       
    xs = [particle[3*i] for i in range(len(orderItems))]
    ys = [particle[3*i+1] for i in range(len(orderItems))]
    thetas = [90*(round(particle[3*i+2])%2) for i in range(len(orderItems))] # 0, 90
       
    orderItems = [shapely.affinity.translate(shapely.affinity.rotate(orderItems[i], thetas[i], origin='centroid'),\
                                             xs[i], ys[i]) for i in range(len(orderItems))] 
    
    #----------------------------- OBJECTIVE 1--------------------------------#
    # upologismos emvadou alliloepikalupsis metaksu twn sximatwn
    orderItemsWithBuffer = [item.buffer(0.1, join_style=joinStyle, cap_style = capStyle) for item in orderItems]
    collection = [P for P in orderItemsWithBuffer]
    
    intersectingArea = 0
    for pol in  itertools.combinations(collection, 2):
        if (pol[0].intersects(pol[1])):
            currentIntersection = pol[0].intersection(pol[1])
            intersectingArea += currentIntersection.area
          
    #----------------------------- OBJECTIVE 2--------------------------------#
    # upologismos emvadou sximatwn pou den topo8eti8ike mesa sto stock
    sumPenaltyArea = 0
    for i in range(len(orderItems)):    
        itemPenaltyArea=0    
        if (orderItems[i].convex_hull.within(currentStockUnion)==False):
            itemWithin = orderItems[i].intersection(currentStockUnion)
            itemPenaltyArea = orderItems[i].area - itemWithin.area
        sumPenaltyArea += itemPenaltyArea
    
    #----------------------------- OBJECTIVE 3--------------------------------#
    #---------------------------- Utilization --------------------------------#
    
    initialStockPolygons = list(initialStock)  
    
    #------------------------ Absolute Waste ---------------------------------#
#    binsWasteArea = []
#    sumWaste = 0
#    
#    for binItem in initialStockPolygons:        
#        wasteForCurrentBin = 0
#        initialBinArea = binItem.area      
#        for item in orderItems:
#            if binItem.contains(item):
#                binItem = binItem.difference(item)       
#        finalBinArea = binItem.area        
#        if (finalBinArea!=initialBinArea):
#            wasteForCurrentBin = finalBinArea
#            binsWasteArea.append(wasteForCurrentBin)
#            
#    for waste in binsWasteArea:
#        sumWaste += waste
        
    #-------------------------- Percent Watse --------------------------------#   
    sumWaste = 0 
    
    for binItem in initialStockPolygons:
        wasteForCurrentBin = 0
        initialBinArea = binItem.area
        for item in orderItems:
            if binItem.contains(item):
                binItem = binItem.difference(item)
        finalBinArea = binItem.area 
        wasteForCurrentBin = finalBinArea/initialBinArea
        sumWaste += wasteForCurrentBin  
        
    #--------------------------- FITNESS FUNCTION ----------------------------#   
    #---------------------SUMMING OF THE OBJECTIVES---------------------------#
    
    objectiveValue = 0.300*intersectingArea + 0.695*sumPenaltyArea + \
                        + 0.005*sumWaste
                        
    return objectiveValue
          
###############################################################################          

class FigureObjects:
    
    def __init__(self, LowerBound, UpperBound):
        """ Creates the figure that will be updated by the update member function.
            
        All line objects (best solution, swarm, global fitness line) are initialized with NaN values, as we only 
        setup the style. Best-so-far fitness 
        
        The input arguments LowerBound & UpperBound must be scalars, otherwise an assertion will fail.
        """
                
        assert np.isscalar(LowerBound), "The input argument LowerBound must be scalar."
        assert np.isscalar(UpperBound), "The input argument LowerBound must be scalar."
        
        # figure
        self.fig = plt.figure()
        
        # 2D axis: the original stock & global best placement of order items
        self.axFirst = self.fig.add_subplot(221)
        self.axFirst.set_xlim(LowerBound, UpperBound)
        self.axFirst.set_ylim(LowerBound, UpperBound)
        self.axFirst.set_title('Stock and Order')
        self.StockAndOrderPlot = self.axFirst
        
        self.axFirst.relim
        self.axFirst.autoscale_view()
        self.axFirst.set_aspect('equal')
        self.axFirst.set_title(f'[{np.NaN},{np.NaN}]') # title is best-so-far position as [x,y]
        
        # 2D axis: the remaining stock
        self.axSecond = self.fig.add_subplot(222)
        self.axSecond.set_xlim(LowerBound, UpperBound)
        self.axSecond.set_ylim(LowerBound, UpperBound)
        self.axSecond.set_title('Remaining Stock')
        self.RemainingStock = self.axSecond
        self.axSecond.relim
        self.axSecond.autoscale_view()
        self.axSecond.set_aspect('equal')
        
        # global best fitness line
        self.axBestFit = plt.subplot(212)
        self.axBestFit.set_title('Best-so-far global best fitness:')
        
        self.lineBestFit, = self.axBestFit.plot([], [])
        
        # auto-arrange subplots to avoid overlappings and show the plot
        self.fig.tight_layout()
    
    def update(self, pso):
        """ Updates the figure in each iteration provided a PSODynNeighborPSO object. """
        # pso.Iteration is the PSO initialization; setup the best-so-far fitness line xdata and ydata, now that 
        # we know MaxIterations
        if pso.Iteration == -1:
            xdata = np.arange(pso.MaxIterations+1)-1
            self.lineBestFit.set_xdata(xdata)
            self.lineBestFit.set_ydata(pso.GlobalBestSoFarFitnesses)          
        
        #---------------------------------------------------------------------#
        # Computing new placement of order items
        orderItems = myOrder.copy()
             
        xx = [pso.GlobalBestPosition[3*i] for i in range(len(orderItems))]
        yy = [pso.GlobalBestPosition[(3*i)+1] for i in range(len(orderItems))]
        zz = [90*(round(pso.GlobalBestPosition[(3*i)+2])%2) for i in range(len(orderItems))]
        
        orderItems = [ shapely.affinity.translate(shapely.affinity.rotate(orderItems[i], zz[i], origin='centroid'), \
                                                  xx[i], yy[i]) for i in range(len(orderItems))]
        
        # Printing new placement of order items
        self.axFirst.clear()
        
        self.axFirst.title.set_text('Stock and Order')
        self.StockAndOrderPlot = plotShapelyPoly(self.axFirst, myStock)
        self.StockAndOrderPlot = plotShapelyPoly(self.axFirst, orderItems)   
        self.axFirst.relim
        self.axFirst.autoscale_view()
        self.axFirst.set_aspect('equal')
        
        #---------------------------------------------------------------------#
        # Computing remaining stock pieces
        remainingStock = myStock
        for item in orderItems:
            item = item.buffer(0.1, join_style=joinStyle, cap_style = capStyle)
            remainingStock = remainingStock.difference(item)
        
        remainingStock = remainingStock.buffer(-0.3, join_style=joinStyle, cap_style = capStyle).buffer(0.3, join_style=joinStyle, cap_style = capStyle)       
        remainingPolygons = list(remainingStock)
        
        # Printing remaining stock pieces
        self.axSecond.clear()
        
        self.axSecond.title.set_text('Remaining Stock Pieces')
        self.RemainingStock = plotShapelyPoly(self.axSecond, remainingPolygons)  
        self.axSecond.relim
        self.axSecond.autoscale_view()
        self.axSecond.set_aspect('equal')
        
        #---------------------------------------------------------------------#
        # update the global best fitness line (remember, -1 is for initialization == iteration 0)
        self.lineBestFit.set_ydata(pso.GlobalBestSoFarFitnesses)
        self.axBestFit.relim()
        self.axBestFit.autoscale_view()
        self.axBestFit.title.set_text('Best-so-far global best fitness: {:g}'.format(pso.GlobalBestFitness))
        
        # because of title and particles positions changing, we cannot update specific artists only (the figure
        # background needs updating); redrawing the whole figure canvas is expensive but we have to
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def OutputFcn(pso, figObj):
    """ Our output function: updates the figure object and prints best fitness on terminal.
        Always returns False (== don't stop the iterative process)
    """
    
    if pso.Iteration == -1:
        print('Iter.    Global best')
    print('{0:5d}    {1:.6f}'.format(pso.Iteration, pso.GlobalBestFitness))   

    
    figObj.update(pso)
    
    return False

###############################################################################
###############################################################################
    
def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def createRemainingStockPolygonsList(remainingStockPolygonsList):
    
    remainingStockPolygonsList = [remainingStockPolygonsList] # to remaining cut pieces mporei na einai multipolygon
    remainingStockPolygonsList = list(flatten(remainingStockPolygonsList))
    
    # from smaller area to larger
    remainingStockPolygonsList.sort(key=lambda piece: piece.area, reverse=0)
    
    remainingStockUnion = cascaded_union(remainingStockPolygonsList)
    
    return remainingStockUnion

###############################################################################

def fitItemUpdateBin(item, currentBin):
    
    currentBin = currentBin.difference(item.buffer(0.1, join_style=joinStyle, cap_style = capStyle))
    currentBin = currentBin.buffer(-0.3, join_style=joinStyle, cap_style = capStyle).buffer(0.3, join_style=joinStyle, cap_style = capStyle)                                       
    
    currentBin = createRemainingStockPolygonsList(currentBin)
        
    return currentBin
    
###############################################################################

############################################################################### 
# creates the list of vertices of the input polygon

def createVerticesList(remainingPolygon):
    
    allVerticesList = []
    
    if (type(remainingPolygon) == shapely.geometry.multipolygon.MultiPolygon):
       # remainingPolygon = shapely.ops.cascaded_union(remainingPolygon)
       remainingPolygons = list(remainingPolygon)
       remainingPolygons.sort(key=lambda piece: piece.area, reverse=0)
       for p in remainingPolygons:             
           if (p.convex_hull.exterior.coords):
               verticesList = p.convex_hull.exterior.coords
               verticesList = sorted(verticesList , key=lambda k: [k[1], k[0]])
               allVerticesList += verticesList
           elif (p.interior.coords):
               verticesList = p.interior.coords
               verticesList = sorted(verticesList , key=lambda k: [k[1], k[0]])
               allVerticesList += verticesList
    else: 
        if (remainingPolygon.convex_hull.exterior.coords):
            verticesList = remainingPolygon.convex_hull.exterior.coords
            verticesList = sorted(verticesList , key=lambda k: [k[1], k[0]])
            allVerticesList += verticesList
        elif (remainingPolygon.interior.coords):
            verticesList = remainingPolygon.interior.coords
            verticesList = sorted(verticesList , key=lambda k: [k[1], k[0]])
            allVerticesList += verticesList
    
    return allVerticesList

###############################################################################

if __name__ == "__main__":
    """ Executed only when the file is run as a script. """   
    
    # in case somebody tries to run it from the command line directly...
    plt.ion()
    
    plt.close("all")
    
    # np.random.seed(1987) # 

#-----------------------------------------------------------------------------#
    # stockBins = Stock    
    # store all stock items in a MultiPolygon, stockBinsUnion.bounds = (0.0, 0.0, 18.5, 21.0)
    
#    Stock.sort(key=lambda piece: piece.area, reverse=0)
#    
#    for i in range(0,len(Stock)):
#        if (Stock[i].bounds[2]>Stock[i].bounds[3]):
#            Stock[i] = shapely.affinity.translate(shapely.affinity.rotate(Stock[i], 90, origin=(0,0)), Stock[i].bounds[3], 0)
    
    stockBins = Stock
    stockBinsUnion = stockBins[0]
    
    # arrange stock in an approximately square space 20x20, one item after the other
    for i in range(1, len(stockBins)):
        testPolygon = shapely.affinity.translate(stockBins[i], stockBins[i-1].bounds[2]+1.5, 0)
        if testPolygon.bounds[2]>20:
            stockBins[i] = shapely.affinity.translate(stockBins[i], 0 ,stockBinsUnion.bounds[3]+1.5)
            stockBinsUnion = stockBinsUnion.union(stockBins[i])
        else:
            stockBins[i] = shapely.affinity.translate(stockBins[i], stockBins[i-1].bounds[2]+1.5, stockBins[i-1].bounds[1])
            stockBinsUnion = stockBinsUnion.union(stockBins[i])

    myOrders = Order1, Order2, Order3
    cutOrders = []
    initialstockBinsUnion = stockBinsUnion
#-----------------------------------------------------------------------------#

    for order_num in range(0, len(myOrders)):
        
        myOrder = myOrders[order_num]
           
        myStock = stockBinsUnion         
    
        nVars = 3*len(myOrder) # 3*len(orderItems) gia na kanoun kai rotate ta polygwna
    
        SwarmSize = 80 # SwarmSize = 90
        # peaks is defined typically defined from -3 to 3, but we set -5 to 5 here to make the problem a bit harder
        LowerBounds = 0 * np.ones(nVars)
        UpperBounds = 20 * np.ones(nVars)
        
        figObj = FigureObjects(LowerBounds[0], UpperBounds[0])
        
        # lambda functor (unnamed function) so that the output function appears to accept one argument only, the 
        # DynNeighborPSO object; behind the scenes, the local object figObj is stored within the lambda   
        outFun = lambda x: OutputFcn(x, figObj)
        
        # UseParallel=True is actually slower for simple objective functions such as this, but may be useful for more 
        # demanding objective functions. Requires the joblib package to be installed.
        # MaxStallIterations=20 is the default. Check how the algorithms performs for larger MaxStallIterations 
        # (e.g., 100 or 200).
        pso = DynNeighborPSO(ObjectiveFcn, nVars,  LowerBounds=LowerBounds, UpperBounds=UpperBounds, SwarmSize=SwarmSize*(order_num+1),
                             OutputFcn=outFun, UseParallel=True, MaxStallIterations=200)                                                              
        pso.optimize() 
    
        print("\nThese are the best positions achieved: ", pso.GlobalBestPosition)
        print("\nThis is the best fitness achieved: ", pso.GlobalBestFitness)        
        
        ## Placing myOrder according to PSO global best position       
        
        best_xs = [pso.GlobalBestPosition[3*i] for i in range(len(myOrder))]
        best_ys = [pso.GlobalBestPosition[3*i+1] for i in range(len(myOrder))]
        best_thetas = [90*(round(pso.GlobalBestPosition[3*i+2])%2) for i in range(len(myOrder))]
    
        # print(best_thetas)
 
        orderItems = myOrder.copy()
        orderItems = [shapely.affinity.translate(shapely.affinity.rotate(orderItems[i], best_thetas[i], origin='centroid'), \
                                                  best_xs[i], best_ys[i]) for i in range(len(orderItems))]

###############################################################################           
#### Finetuning Order Placement with Bottom Left Fill Heuristic Placement #####
###############################################################################
                
        stockBins = list(stockBinsUnion)     
        listOfItemsAssignedToBin = []        
        currentOrderFinetuned = []       

        for currentBin in stockBins:                                  
            # mapping of items of the order to the corresponding bin
            listOfItemsAssignedToBin = []
            indices= [i for i in range(len(orderItems)) if orderItems[i].intersects(currentBin)==True]
            itemsAssignedToBinByPSO = [orderItems[i] for i in indices]            
            # make (bounds[0], bounds[1]) = (0,0) for all the items                     
            itemsAssignedToBin = [shapely.affinity.translate(item, -item.bounds[0], -item.bounds[1]) \
                                        for item in itemsAssignedToBinByPSO]
            # sort item in descending order of size
            itemsAssignedToBin.sort(key=lambda piece: piece.area, reverse=1)
            
            numOfItemsToBeFitted = len(itemsAssignedToBin)
            numOfItemsFittedSoFar = 0
            itemsInsertedIntoBin = []
            
            tempCurrentBin = currentBin
            
            for item in itemsAssignedToBin:
                
                insertionPoints = createVerticesList(tempCurrentBin)
                
                for insertionPoint in insertionPoints:
            
                    tempItem = shapely.affinity.translate(item, insertionPoint[0], insertionPoint[1])    
                    
                    tempItemRotated90CW = shapely.affinity.translate(shapely.affinity.rotate(item, -90, origin = (0,0)), 0, item.bounds[2])
                    tempItemRotated90CW = shapely.affinity.translate(tempItemRotated90CW, insertionPoint[0], insertionPoint[1])
                
                    if (tempItem.within(tempCurrentBin)==True and tempItemRotated90CW.within(tempCurrentBin)==True):                
                        remainingBinTempItem = fitItemUpdateBin(tempItem, tempCurrentBin)
                        remainingBinTempItemRotated90CW = fitItemUpdateBin(tempItemRotated90CW, tempCurrentBin)
                    
                        # if both rotations can be fitted at current inserted point
                        if (smoothness(remainingBinTempItem)==smoothness(remainingBinTempItemRotated90CW)):
                            if (remainingBinTempItem.bounds[3]<=remainingBinTempItemRotated90CW.bounds[3]):
                                tempCurrentBin = fitItemUpdateBin(tempItem, tempCurrentBin)
                                itemsInsertedIntoBin.append(tempItem)
                                numOfItemsFittedSoFar +=1
                                break
                            else: 
                                tempCurrentBin = fitItemUpdateBin(tempItemRotated90CW, tempCurrentBin)
                                itemsInsertedIntoBin.append(tempItemRotated90CW)
                                numOfItemsFittedSoFar +=1
                                break
                        elif (smoothness(remainingBinTempItem)>smoothness(remainingBinTempItemRotated90CW)):        
                            tempCurrentBin = fitItemUpdateBin(tempItem, tempCurrentBin)
                            itemsInsertedIntoBin.append(tempItem)
                            numOfItemsFittedSoFar +=1
                            break
                        else:
                            tempCurrentBin = fitItemUpdateBin(tempItemRotated90CW, tempCurrentBin)
                            itemsInsertedIntoBin.append(tempItemRotated90CW)
                            numOfItemsFittedSoFar +=1
                            break
                
                    # if only one of the rotations can be fitted
                    elif (tempItem.within(tempCurrentBin)==True and tempItemRotated90CW.within(tempCurrentBin)==False):
                        tempCurrentBin = fitItemUpdateBin(tempItem, tempCurrentBin)
                        itemsInsertedIntoBin.append(tempItem)
                        numOfItemsFittedSoFar +=1
                        break
                    elif (tempItem.within(tempCurrentBin)==False and tempItemRotated90CW.within(tempCurrentBin)==True):
                        tempCurrentBin = fitItemUpdateBin(tempItemRotated90CW, tempCurrentBin)
                        itemsInsertedIntoBin.append(tempItemRotated90CW)
                        numOfItemsFittedSoFar +=1
                        break
                    # if none of the rotations can be fitted
                    else:
#                        print("Not fitted!")
#                        print("The insertion point is ", insertionPoint)
                        ()
                                            
            # estimating how good of a result is the current permutation
            if (numOfItemsFittedSoFar == numOfItemsToBeFitted): 
                placementForCurrentBin = itemsInsertedIntoBin
            else:
                placementForCurrentBin = itemsAssignedToBinByPSO
                
            # "save" the fitted items of the current bin, i.e. add them to a list of all the items fitted in previous 
            currentOrderFinetuned = currentOrderFinetuned + placementForCurrentBin

###############################################################################
########################## End of BLF Heuristic ###############################
###############################################################################                         
              
        remainingStockBins = stockBinsUnion
        
        for item in currentOrderFinetuned:
            remainingStockBins = remainingStockBins.difference(item.buffer(0.1, join_style=joinStyle, cap_style = capStyle))
            
        remainingStockBins = remainingStockBins.buffer(-0.3, join_style=joinStyle, cap_style = capStyle).buffer(0.3, join_style=joinStyle, cap_style = capStyle)
        
        fig, ax = plt.subplots(ncols=2)
        fig.canvas.set_window_title('Order after BLF finetuning & Morphological Opening applied')
        pp = plotShapelyPoly(ax[0], stockBins+currentOrderFinetuned)
        pp[0].set_facecolor([1,1,1,1])
        plotShapelyPoly(ax[1], remainingStockBins)
        ax[0].set_title('Order items after BLF')
        ax[1].set_title('Remaining stock after BLF')
        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()
        for a in ax:
            a.set_aspect('equal') 
    
        # FINAL ASSIGGNMENTS OF THIS ORDER
        myOrder = currentOrderFinetuned                
        cutOrders.append(myOrder)     
        stockBinsUnion = remainingStockBins
    
###############################################################################
###############################################################################
### excecuted only once in the end  
    
    listOfInitialstockBins = list(initialstockBinsUnion)
    cutOrders = [item for sublist in cutOrders for item in sublist]
    
    # PLOT ALL ODRERS TOGETHER
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('All Orders Together on Stock')
    plotShapelyPoly(ax, initialstockBinsUnion)
    plotShapelyPoly(ax, cutOrders)
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect('equal')
    
    # PROOF OF FEASIBILITY OF SOLUTION
    collection = [P for P in cutOrders]
    
    intersectingArea = 0
    for pol in  itertools.combinations(collection, 2):
        if (pol[0].intersects(pol[1])):
            currentIntersection = pol[0].intersection(pol[1])
            intersectingArea += currentIntersection.area
            
        sumPenaltyArea = 0
    for i in range(len(cutOrders)):    
        itemPenaltyArea=0    
        if (cutOrders[i].within(initialstockBinsUnion)==False):
            itemWithin = cutOrders[i].intersection(initialstockBinsUnion)
            itemPenaltyArea = cutOrders[i].area - itemWithin.area
        sumPenaltyArea += itemPenaltyArea
 
    print("\n intersectingArea = ", intersectingArea)
    print("\n sumPenaltyArea = ", sumPenaltyArea) 

###############################################################################

            
