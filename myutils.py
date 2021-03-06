import sys, os, time
import numpy as np
import pickle
import requests

# My Utils Functions

# +++++++++++++++++++++++++++++++++ Path ++++++++++++++++++++++++++++++++++++++

def setSysPath(path):
    if path not in sys.path:
        print(f'Inserting {path} to system path')
        sys.path.insert(0, path)

def setCWD():
    """Checking path, if necessary, change to current working directory"""
    if os.getcwd() != os.path.dirname(__file__):
        print('Changing Path to Current Working Directory')
        os.chdir(os.path.dirname(__file__))

        # or sys.path[0] instead of os.path.dirname(__file__)
"""As initialized upon program startup, the first item of this list, path[0], 
is the directory containing the script that was used to invoke the Python interpreter
"""

# +++++++++++++++++++++++++++++++++ Files +++++++++++++++++++++++++++++++++++++

# Load / Save Files to / from variables

def saveObj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def loadObj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Files List

def listFiles(path):
    """Returns a List of strings with the file names (no directories) on the given path"""
    filesList = [f for f in os.listdir(path) if os.path.isfile(path+f)]
    return filesList


# Downloading file 

def downloadFile(url, fileName=None):
    """Downloads file from url and stores it as 'fileName' if given"""

    try:
        r = requests.get(url)
        #r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(e)
        #raise Systemexit(e)
    else:
        if fileName is None:
            fileName = os.path.basename(url)

        with open(fileName, 'wb') as f:
            f.write(r.content)


# Update File if Necessary

def updateFile(fileUrl, useFileName=None, mtime=1, force=False, verbose=False):
    """Checks if a file exists on path, if not or if older than mtime
    (days, fraction possible), it downloads it from the url.
    If no fileName is given it uses the name on url.
    """
    
    if useFileName is None:
        print(f'\nNo file Name given, using url basename. And returning its value')
        fileName = os.path.basename(fileUrl)
    else:
        fileName = useFileName

    if os.path.exists(fileName):
        if verbose: print(f'\nFile already exists')

        timeDiff = (time.time() - os.path.getmtime(fileName)) / (3600*24)

        if timeDiff >= mtime or force:
            if verbose:
                print(f'\nLocal version older than required. Downloading from url...')
 
            downloadFile(fileUrl, fileName)
                
        else:

            print(f'\nThe date of local file {fileName} is within required timespan of {mtime} days.'\
                  ' Not downloading')
    else:

        downloadFile(fileUrl, fileName)

    if useFileName is None:
        return fileName

# ++++++++++++++++++++++ Command line Input +++++++++++++++++++++++++++++++++++

def inputInteger(default=0, message='Enter an option (integer) [q to quit]:'):
    """Enter integers until quiting is desired.
    If only enter is pressed the default value is used"""
    while True:
        
        n = input(f'\n{message}')
        if n == 'q':
            sys.exit(0)

        elif n == '':
            print(f'No pick, using default value {default}')            
            return default

        try:
            n = int(n)
            return n
        
        except ValueError:
            print('Not a valid input')


def inputWithTimeout(timeout, promptMsg = None, raiseException = False):
    '''Input with timeout. In Linux execute it directly on terminal.
     (It may not work on Windows)
     
     timeout: timeout in seconds
     promptMsg: Message to prompt (optional)
     raiseException: if True sets TimeoutExpired after timeout.
     Otherwise continues (default)
    '''
    if promptMsg:
        sys.stdout.write(promptMsg)
        sys.stdout.flush()

    ready, _, _ = select.select([sys.stdin], [],[], timeout)
    
    if ready:
        return sys.stdin.readline().rstrip('\n') # expect stdin to be line-buffered

    elif raiseException:
        sys.stdout.write('\nTimeout Expired\n')
        sys.stdout.flush()
        raise TimeoutError

# +++++++++++++++++++++++++++++++++ Time ++++++++++++++++++++++++++++++++++++++

def countDown(s=3):
    """Simple countdown to delay the start of a process
    countDown(s): s: seconds to delay. Default=3
    """
    for n in range(s,0,-1):
        print(n)
        time.sleep(1)

# +++++++++++++++++++++++++++++++++ Math ++++++++++++++++++++++++++++++++++++++

def ceil(x):
    """Simple ceil implementation"""
    y = int(x) if (x==int(x) or x<0) else int(x)+1
    return y

# Normalize numpy array

# Improve names, make more descriptive

def normalize(x):
    """Normalize numpy array to range [0,1]"""
    xNorm = (x-np.min(x)) / np.ptp(x)
    return xNorm

def normalize2(x):
    """Normalize numpy array to range [-1,1]"""
    xNorm = 2.*(x-np.min(x)) / np.ptp(x) - 1
    return xNorm

# Alternative to given range:
# np.interp(a, (a.min(), a.max()), (-1, +1))

# +++++++++++++++++++++++++++++++++ Numpy +++++++++++++++++++++++++++++++++++++

def isInteger(x):
    """Check if the elements of the numpy array x are integers"""
    return np.equal(np.mod(x, 1), 0)

# +++++++++++++++++++++++++++++++++ Others ++++++++++++++++++++++++++++++++++++

# Toggle values (used for turns)
def toggleValue(currentValue, value1, value2):
    if currentValue == value1:
        return value2
    elif currentValue == value2:
        return value1

