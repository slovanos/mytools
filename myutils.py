import sys, os, time
import select
import numpy as np
import pickle
import requests
import timeit
from functools import wraps
import json


# ++++++++++++++++++++++++++ Timing / Time ++++++++++++++++++++++++++++++++++++
def tic():
    """
    Time code snippet. Tribute to Matlab tic-toc functions.
    Usage:

    t = tic()
    #code snippet to time
    toc(t)
    """
    return timeit.default_timer()


def toc(t, verbose=True):
    """
    Time code snippet. Tribute to Matlab tic-toc C functions.
    Usage:

    t = tic()
    #code snippet to time
    toc(t)
    """
    elapsed = timeit.default_timer()-t

    if verbose:
        if elapsed < 1:
            time_unit = 'ms'
            elapsed *= 1000
        elif 1 <= elapsed < 60:
            time_unit = 's'
        else:
            elapsed /= 60
            time_unit = 'm'

        elapsed = round(elapsed, 2)
        print('Time elapsed:', elapsed, time_unit)

    return elapsed


def timefunc(f):
    """
    Decorator to measure function excecution time.
    Usage:

    @timefunc
    def function_to_time(arg1, arg2):
       pass
    """
    @wraps(f)
    def wrap(*args, **kw):

        ts = timeit.default_timer()
        result = f(*args, **kw)
        elapsed = timeit.default_timer() - ts

        if elapsed < 1:
            time_unit = 'ms'
            elapsed *= 1000
        elif 1 <= elapsed < 60:
            time_unit = 's'
        else:
            elapsed /= 60
            time_unit = 'm'

        elapsed = round(elapsed, 2)

        #  print(f.__name__, args, kw)
        print('Time:', elapsed, time_unit)

        return result
    return wrap


def countdown(s=3):
    """Simple countdown to delay the start of a process
    countdown(s): s: seconds to delay. Default=3
    """
    for n in range(s, 0, -1):
        print(n)
        time.sleep(1)


# +++++++++++++++++++++++++++++++++ Path ++++++++++++++++++++++++++++++++++++++
def setsyspath(path):
    if path not in sys.path:
        print(f'Inserting {path} to system path')
        sys.path.insert(0, path)


def setcwd():
    """Checking path, if necessary, change to current working directory"""
    if os.getcwd() != os.path.dirname(__file__):
        print('Changing Path to Current Working Directory')
        os.chdir(os.path.dirname(__file__))
        # or sys.path[0] instead of os.path.dirname(__file__)


"""As initialized upon program startup, the first item of this list, path[0],
is the directory containing the script that was used to invoke the Python interpreter
"""


# +++++++++++++++++++++++++++++++++ Files +++++++++++++++++++++++++++++++++++++

# Load / Save Files to / from variables / List / Download
def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def list_files(path, extensions=''):
    """
    Returns a sorted List of strings with the file names for the given path
    extensions: string or tuple with strings.
    Examples: list_files('exampledir, ('.jpg', '.png'))
    """

    files_list = [f for f in os.listdir(path)
    if os.path.isfile(os.path.join(path, f)) and f.endswith(extensions)]

    return sorted(files_list)


def download_file(url, file_name=None):
    """Downloads file from url and stores it as 'file_name' if given"""
    try:
        r = requests.get(url)
        # r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(e)
        # raise Systemexit(e)
    else:
        if file_name is None:
            file_name = os.path.basename(url)

        with open(file_name, 'wb') as f:
            f.write(r.content)


# Update File if Necessary
def update_file(file_url, use_file_name=None, mtime=1, force=False, verbose=False):
    """Checks if a file exists on path, if not or if older than mtime
    (days, fraction possible), it downloads it from the url.
    If no file_name is given it uses the name on url.
    """
    if use_file_name is None:
        print('\nNo file Name given, using url basename. And returning its value')
        file_name = os.path.basename(file_url)
    else:
        file_name = use_file_name

    if os.path.exists(file_name):
        if verbose:
            print('\nFile already exists')

        time_diff = (time.time() - os.path.getmtime(file_name)) / (3600*24)

        if time_diff >= mtime or force:
            if verbose:
                print('\nLocal version older than required. Downloading from url...')

            download_file(file_url, file_name)
        else:
            print(f'\nThe date of local file {file_name} is within required timespan'
                  f'of {mtime} days. Not downloading')
    else:
        download_file(file_url, file_name)

    if use_file_name is None:
        return file_name


# ++++++++++++++++++++++ Command line Input +++++++++++++++++++++++++++++++++++
def input_integer(default=0, message='Enter an option (integer) [q to quit]:'):
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


def input_w_timeout(timeout, prompt_msg=None, raise_exception=False):
    '''Input with timeout. In Linux execute it directly on terminal.
     (It may not work on Windows)

     timeout: timeout in seconds
     prompt_msg: Message to prompt (optional)
     raise_exception: if True sets TimeoutExpired after timeout.
     Otherwise continues (default)
    '''
    if prompt_msg:
        sys.stdout.write(prompt_msg)
        sys.stdout.flush()

    ready, _, _ = select.select([sys.stdin], [], [], timeout)

    if ready:
        return sys.stdin.readline().rstrip('\n')  # expect stdin to be line-buffered

    elif raise_exception:
        sys.stdout.write('\nTimeout Expired\n')
        sys.stdout.flush()
        raise TimeoutError


def sep(c='-', l=80):
    """Simple separator for the command line output"""
    if len(c) < 2:  # only separator
        out = c * l
    else:  # text
        if c[0] in '+-#=':  # input: separator followed by text
            s = c[0]
            c = c[1:]
        else:  # input: just text
            s = '='
        ls = l-2 - len(c)
        n1 = int(ls / 2)
        n2 = ls - n1

        out = s*n1 + ' ' + c + ' ' + s*n2

    out = '\n' + out + '\n'
    print(out)


# +++++++++++++++++++++++++++++++++ Math ++++++++++++++++++++++++++++++++++++++
def ceil(x):
    """Simple ceil implementation"""
    y = int(x) if (x == int(x) or x < 0) else int(x)+1
    return y


# +++++++++++++++++++++++++++++++++ Numpy +++++++++++++++++++++++++++++++++++++

def normalize(x):
    """Normalize numpy array to the unit interval. Range [0,1]"""
    return (x-np.min(x)) / np.ptp(x)


def normalizesigned(x):
    """Normalize numpy array to range [-1,1]"""
    return 2.*(x-np.min(x)) / np.ptp(x) - 1


def normalize2range(x, minx=0, maxx=1):
    """Normalize numpy array to arbitrary range [minx,maxx]"""
    return (x-x.min()) * (maxx-minx) / x.ptp() + minx

# Alternative to given range:
# np.interp(a, (a.min(), a.max()), (-1, +1))


def is_integer(x):
    """Check if the elements of the numpy array x are integers"""
    return np.equal(np.mod(x, 1), 0)


def cos_sim(u, v):
    """Calculates cosine similarity function between vectors u and v"""
    return np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))


def argsort_k_th(v, k, th):
    """Return indexes for the top k elements of v greater than th in descending order"""
    idx, = np.where(v > th)
    return idx[v[idx].argsort()[::-1][:k]]


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# +++++++++++++++++++++++++++++++++ Others ++++++++++++++++++++++++++++++++++++
# Toggle values (used for turns)
def toggle_value(current_value, value1, value2):
    if current_value == value1:
        return value2
    elif current_value == value2:
        return value1
