import os
os.environ['MKL_NUM_THREADS']="1"
os.environ['NUMEXPR_NUM_THREADS']="1"
os.environ['OMP_NUM_THREADS']="1"

import pickle
import numpy
import scipy.sparse as sp
import mumpspy
import pymumps
from mumps import mumps
# import pypardiso 
# import time

from funcy import print_durations

import cProfile

def profileit(prof_fname, sort_field='cumtime'):
    """
    Parameters
    ----------
    prof_fname
        profile output file name
    sort_field
        "calls"     : (((1,-1),              ), "call count"),
        "ncalls"    : (((1,-1),              ), "call count"),
        "cumtime"   : (((3,-1),              ), "cumulative time"),
        "cumulative": (((3,-1),              ), "cumulative time"),
        "file"      : (((4, 1),              ), "file name"),
        "filename"  : (((4, 1),              ), "file name"),
        "line"      : (((5, 1),              ), "line number"),
        "module"    : (((4, 1),              ), "file name"),
        "name"      : (((6, 1),              ), "function name"),
        "nfl"       : (((6, 1),(4, 1),(5, 1),), "name/file/line"),
        "pcalls"    : (((0,-1),              ), "primitive call count"),
        "stdname"   : (((7, 1),              ), "standard name"),
        "time"      : (((2,-1),              ), "internal time"),
        "tottime"   : (((2,-1),              ), "internal time"),
    Returns
    -------
    None

    """
    def actual_profileit(func):
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            stat_fname = '{}.stat'.format(prof_fname)
            prof.dump_stats(prof_fname)
            print_profiler(prof_fname, stat_fname, sort_field)
            print('dump stat in {}'.format(stat_fname))
            return retval
        return wrapper
    return actual_profileit


def print_profiler(profile_input_fname, profile_output_fname, sort_field='cumtime'):
    import pstats
    with open(profile_output_fname, 'w') as f:
        stats = pstats.Stats(profile_input_fname, stream=f)
        stats.sort_stats(sort_field)
        stats.print_stats()


with open('dataMatTest.pck','rb') as f:
    data = pickle.load(f)

A = data['A']
b = data['b']

print('Matrix size {}'.format(A.shape))

## test scipy
@print_durations()
@profileit("scipy_profile")
def run_scipy():
    sol = sp.linalg.spsolve(A,b)



## test mumpspy
solver = mumpspy.MumpsSolver(system='real64')
@print_durations()
@profileit("mumpspy_profile")
def run_mumpspy():
    sol = solver.solve(A,b)


## test pymumps
@print_durations()
@profileit("pymumps_profile")
def run_pymumps():
    sol = pymumps.spsolve(A,b)

## test python-mumps
inst = mumps.Context()
@print_durations()
@profileit("python-mumps_profile")
def run_pythonmumps():    
    inst.factor(A)
    sol = inst.solve(b)

# ## test pardiso
# @print_durations()
# @profileit("pardiso_profile")
# def run_pardiso():  
#     sol = pypardiso.spsolve(A,b)


run_scipy()
run_mumpspy()
run_pymumps()
run_pythonmumps()
# run_pardiso()

nb = 1e4

A = sp.rand(nb,nb,density=0.25, format='csr')
b = numpy.random.rand(int(nb))

print('Matrix size {}'.format(A.shape))

run_scipy()
run_mumpspy()
run_pymumps()
run_pythonmumps()
# run_pardiso()
