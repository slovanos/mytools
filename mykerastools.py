#import keras
from keras.models import Sequential
from keras.layers import Dense#, InputLayer
from keras import optimizers
from keras.callbacks import Callback
import keras.backend as K
import numpy as np
# Grid Search form scikit-learn
from sklearn.model_selection import ParameterGrid
from mytools.myutils import ceil, is_integer
import time

# My Keras Utils Functions

# Getting Model Info (maybe integrate to visualize function)
# (add lr and loss used)

# Callback to stop trainig when reaching a certain value

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='acc', baseline=1.0):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print(f'Epoch {epoch}: Reached baseline, terminating training')
                self.model.stop_training = True
                

def getKerasModelData(model, returnValue = 'dict'):

    '''Takes a Keras model and returns:
        List of the nnStructure: one element per layer indicating number of units
        List of the nnActivations: Names of the Activation Functions for each layer. 
        List of Weights: weights of each layer as numpy array
        TotalNumber of weights as integer'''

    nLayers = len(model.layers)
    nnInput = model.input_shape[1]
    nnStructure = [nnInput]
    nnActivations = []
    nnWeights = []

    for n in range(nLayers):
        nnStructure.append(model.layers[n].units)
        nnWeights.append(model.layers[n].get_weights()[0]) # on [1] are the bias weights
        nnActivations.append(model.layers[n].activation.__name__)
        
    nTotalWeights =  model.count_params()

    if returnValue == 'dict':

        data = {'nnStructure': nnStructure, 'nnActivations': nnActivations, \
                'nnWeights': nnWeights, 'nTotalWeights': nTotalWeights}

        return data

    elif returnValue == 'list': # for backward compatibility (cuando se adapten
        # borrar nomás)

        return nnStructure, nnActivations, nnWeights, nTotalWeights


# Make a Dense keras model with arbitrary layers

def makeDenseModel(X ,Y ,hiddenLayers , defaultActivation = 'relu', optimizer = 'sgd', loss = None, metrics = None, lr = None, verbose = 1):

    '''Make a Keras Sequential Dense Model. Arguments:
    X: input data, dim: (m x n) with m: #samples, n: #features
    Y: target values, dim: (m x nL)
    hiddenLayers: list of integers, tuples or combinations of both containing number of units
    (and activation functions if desired) for each of the the hidden layers.
    Where no activactions are specified the defaultActivation will be used ('relu' if not specified)

    defaultActivation: activation used when no specification or inference available.

    optimizer: desired optimizer. 'sgd' default.

    loss: Loss function will be infered automatically from data unless explicited.

    lr: learning rate. If not provided the optimizer's default will be used.

    Example: hiddenLayers = [(128,'relu'), 64, (32,'tanh'), ...]'''

    # Input Layer

    m,nInputFeatures = X.shape

    # First Layer Data
    
    l1 = hiddenLayers[0]

    if isinstance(l1,tuple):
        
        n1 = l1[0]
        a1 = l1[1]

    elif isinstance(l1, int):

        n1 = l1

        if np.array_equal(np.array([-1,0,1]), np.unique(X)):

            if verbose >=1: print(f'[-1,0,1] inputs interval detected')

            a1 = 'tanh'

        elif np.array_equal(np.array([0,1]), np.unique(X)):

            a1 = 'sigmoid'

        else:

            a1 = defaultActivation

    else:

        raise ValueError(f'Invalid data type first element of hiddenLayers: {l1}')

    # Last layer and loss

    if verbose >=1: print(f'\nInfering parameters for model...')

    nL = Y.shape[1]

    if np.array_equal(np.array([-1,0,1]), np.unique(Y)):

        if verbose >=1: print(f'Multivalues output detected ([-1,0,1])')

        aL = 'tanh'
        loss = loss or 'mse'
        
    elif np.array_equal(np.array([0,1]), np.unique(Y)):
        
        if verbose >=1: print(f'Hot-Encoded output detected')
        
        # output should be:
        # sigmoid, in case of binary classification (one output unit, two classes (0,1))
        # or multi-label classification (more than two non exclusive targets)
        if nL == 1 or np.any(np.count_nonzero(Y, axis = 1) > 1): 

            if verbose >=1:
                if nL == 1:
                    print(f'Binary output detected')
                else:
                    print(f'Multilabel-classification detected')

            aL = 'sigmoid'
            loss = loss or 'binary_crossentropy'
            
        # OR softmax, in case multi-class classification (more than two exclusive targets)
        else: # np.all(np.count_nonzero(test, axis = 1) == 1)

            if verbose >= 1: print(f'Multiclass-classification detected')
            
            aL = 'softmax'
            loss = loss or 'categorical_crossentropy'

    else:

        aL = 'linear'
        loss = loss or 'mse'

    # ++++++++ Making model ++++++++++++

    model = Sequential()
    model.add(Dense(n1, activation=a1, input_shape=(nInputFeatures,)))

    for l in hiddenLayers[1:]:

        if isinstance(l,tuple):

            nl = l[0]
            al = l[1]

        elif isinstance(l,int):

            nl = l
            al = defaultActivation

        else:
            
            raise ValueError(f'Invalid data type in hiddenLayers: {l}')

        model.add(Dense(nl, activation=al))

    # Last Layer
    model.add(Dense(nL, activation=aL))
    
    metrics = metrics or ['acc']
    # Compile
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if verbose >=1:
        print(f'\nOutput Layer Activation: {aL}, Loss Function: {loss}, metrics: {model.metrics}')

    if lr:

        K.set_value(model.optimizer.lr, lr)

    return model


def trainMany(X, Y, parameterPool, adaptEpochs = True, callbacks = None , metrics = None, verbose = 1):

    '''Trains many models on the training data X, Y, combining the parameters
    passed in the dictionary parameterPool.
    If parameters are not passed, default values are taken or infered
    adapteEpochs: if True, Epochs are adapted Proportionaly to batchSize'''

    nTrainSamples = X.shape[0]

    nInputFeatures = X.shape[1]
    nOutputs = Y.shape[1]

    if 'nEpochs' not in parameterPool:
        parameterPool['nEpochs'] = [100]
        
    if 'batchSize' not in parameterPool:
        parameterPool['batchSize'] = [32]

    if 'hiddenLayers' not in parameterPool:
        parameterPool['hiddenLayers'] = [[32]]

    if 'optimizer' not in parameterPool:
        parameterPool['optimizer'] = ['sgd']

    if 'loss' not in parameterPool:
        parameterPool['loss'] = [None]

    #if 'metrics' not in parameterPool:
    #    parameterPool['metrics'] = [['accuracy']]

    if 'learningRate' not in parameterPool:
        parameterPool['learningRate'] = [None]

    parameterGrid = ParameterGrid(parameterPool) # every element is a a dict with a parameter combination

    resultsCompilation = []

    iteration = 1

    for parameterCombination in parameterGrid:

        if adaptEpochs and len(parameterPool['nEpochs']) == 1:

            if len(parameterPool['batchSize']) > 1:# proportionally according batchSize

                nEpochs = parameterPool['nEpochs'][0]*ceil(parameterCombination['batchSize']/parameterPool['batchSize'][0])

            elif len(parameterPool['hiddenLayers']) > 1: #proportionally to NN size. # Think to change to adapt by number of weights

                nEpochs = ceil(parameterPool['nEpochs'][0]*np.prod(parameterCombination['hiddenLayers'])/np.prod(parameterPool['hiddenLayers'][0]))

            elif len(parameterPool['learningRate']) > 1: #proportionally according learningRate

                nEpochs = ceil(parameterPool['nEpochs'][0]*ceil(parameterPool['learningRate'][0]/parameterCombination['learningRate']))

            else:

                if verbose >= 1:
                    print('No variable hyperparameters to adapt Epochs to (batchSize or hiddenLayers) or several Epochs given')
                    nEpochs = parameterCombination['nEpochs']
        else:

            nEpochs = parameterCombination['nEpochs']

        batchSize = parameterCombination['batchSize']
    
        # es = EarlyStopping(monitor='acc', mode = 'max', patience = patience, verbose = 2)
        # tb = TerminateOnBaseline(monitor='acc', baseline=1.0)

        if verbose >= 1:

            print(f'\n++++++++++ Training iteration {iteration} from {len(parameterGrid)} ++++++++')
            print(f'Parameter Combination: {parameterCombination}')
            print(f'nEpochs: {nEpochs}')

        # Creating Model
            
        model = makeDenseModel(X, Y, parameterCombination['hiddenLayers'],\
                               optimizer = parameterCombination['optimizer'],\
                               loss = parameterCombination['loss'],\
                               metrics = metrics,\
                               lr = parameterCombination['learningRate'], verbose = verbose)

        # Training   
        start = time.process_time()

        history = model.fit(X, Y, epochs = nEpochs, batch_size = batchSize, verbose = 0, callbacks = callbacks)

        end = time.process_time()

        if verbose >= 1:
            
            print(f'Time to fit {(end-start)/60:.2f} minutes')

            print(model.metrics_names)
            print(model.evaluate(X, Y, verbose = 0))

            #print(f'\nloss: {loss}, accuracy: {accuracy}')

        npHist = history2Numpy(history.history)

        combinationResults = {'model': model,\
                              'parameterCombination':parameterCombination,\
                              'history': npHist,\
                              'time':(end-start)/60}
        # it only uses first metric. Fix
        resultsCompilation.append(combinationResults)
        
        iteration += 1   

    return resultsCompilation


def history2Numpy(historyDict):

    histDictNumpy = dict()

    for key,lista in historyDict.items():

        histDictNumpy[key] = np.array(lista)

    return histDictNumpy
