import matplotlib.pyplot as plt
from mytools.mykerastools import getKerasModelData
import numpy as np
import time

#plt.rcParams.update({'font.size': 20, 'figure.figsize': (10, 8)})

# My Plot Tools

# Disticnt color list

colorList = [(0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),\
             (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),\
             (210, 245, 60), (0, 128, 128), (230, 190, 255), (170, 110, 40),\
             (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0),\
             (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255)]

colorsArrayNormalized = np.array(colorList)/255 

# Getting fixed and variable hyperparameters of the pool

def getFixedAndVariable(parameterPool):

    fixed = dict()
    variable = set()

    for key,value in parameterPool.items():

        if len(value) > 1:

            variable.add(key)

        else:

            fixed[key] = value[0]

    return fixed, variable

# Plotting Results

def plotResultsCompilation(resultsCompilation, parameterPool, mode = 'allInOne', plotMetric = 'acc', show = True,\
                           save = False, fileName = None, verbose = 0, colorsArray = colorsArrayNormalized):

    # Detect fixed and variable hyperparameters
    fixed, variable = getFixedAndVariable(parameterPool)

    if 'hiddenLayers' in fixed:
        modelData = getKerasModelData(resultsCompilation[0].get('model'))
        fixed['model'] = (modelData['nnStructure'], modelData['nnActivations'])
    
    figureText = f'Fixed Hyperparameters: {fixed}'#, loss: {resultsCompilation[0].get("model").loss}'

    if verbose >= 1:
        print(figureText)
        print(f'Fixed Hyperparameters: {variable}')

    # Text box properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    if mode == 'allInOne':

        fig, ax = plt.subplots()
        plt.title(figureText)
        idx = 0

        for combinationResults in resultsCompilation:

            color = colorsArray[idx] if idx < len(colorsArray) else np.random.rand(3)
            combinationList = [f'{key}: {combinationResults["parameterCombination"][key]}' for key in variable]
            labelString = f'{combinationList}'\
                          f'\nmax Acc: {float(combinationResults["history"][plotMetric].max()):.4f} ({combinationResults["history"][plotMetric].argmax()})'\
                          f'\nmin Loss: {float(combinationResults["history"]["loss"].min()):.4f} ({combinationResults["history"]["loss"].argmin()})'\
                          f'\ntime: {combinationResults["time"]:.2f} min'
            ax.plot(combinationResults['history'][plotMetric], label = labelString, color = color)
            ax.plot(combinationResults['history']['loss'], color = color)
            idx += 1

        plt.xlabel('Epochs')
        plt.ylabel('Loss & '+plotMetric)

        ax.legend(shadow=True, framealpha =0.7, loc = 0)# loc = 7: center right
        
        ax.grid(True)

    elif mode == 'many':

        nCombinations = len(resultsCompilation)

        fig, axs = plt.subplots(nCombinations)
        fig.suptitle(figureText)

        for index in range(nCombinations):

            axs[index].plot(resultsCompilation[index]['history']['loss'], label='loss')
            axs[index].plot(resultsCompilation[index]['history'][plotMetric], label=plotMetric)

            combinationList = [f'{key}: {resultsCompilation[index]["parameterCombination"][key]}' for key in variable]
            
            textString = f'{combinationList}'\
                         f'\nmax Acc: {float(resultsCompilation[index]["history"][plotMetric].max()):.4f} '\
                         f'on Epoch: {resultsCompilation[index]["history"][plotMetric].argmax()}'\
                         f'\nmin Loss: {float(resultsCompilation[index]["history"]["loss"].min()):.4f} '\
                         f'on Epoch: {resultsCompilation[index]["history"]["loss"].argmin()}'\
                         f'\ntime: {resultsCompilation[index]["time"]:.2f} min'
            #axs[index].set_title()
            axs[index].text(0.5, 0.4, textString, transform=axs[index].transAxes, bbox = props)
            axs[index].legend(shadow=True, fancybox=True, framealpha =0.7)
            axs[index].grid(True)

    else:

        raise ValueError("graph mode not valid. (Ex. 'allInOne or 'many')")

    fig.set_size_inches(20,12)
    
    if save:
        dateString = time.strftime('%Y%m%d%H%M',time.localtime())
        fileName = f'Varying {variable}.png' if fileName is None else fileName
        plt.savefig(dateString+fileName)

    if show: plt.show(block = False)
