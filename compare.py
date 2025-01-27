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
import pandas as pd
# import pypardiso 
import time

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

def build_pb(size=1000, band_size=10):
    nb = size
    # build diagnonals
    diagonals = list()
    for i in range(band_size):
        diagonals.append(numpy.random.rand(int(nb-i)))
    # assemble matrix
    Amat = sp.diags(diagonals,range(band_size),format='csr')
    # generate symmetric matrix
    Amat = Amat + Amat.T - sp.diags(Amat.diagonal())
    # generate rhs
    bvec = numpy.random.rand(int(nb))
    # triangular lower part of A
    Amattril = sp.tril(Amat)
    return Amat, bvec, Amattril


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
    return sol

class scipy_solver:
    @print_durations()
    @profileit("scipy_factorize_profile")
    def __init__(self, A):
        self.factor = sp.linalg.factorized(A) #sp.linalg.splu(A)

    @print_durations()
    @profileit("scipy_solve_profile")
    def solve(self, b):
        return self.factor(b) #self.factor.solve(b)


## test mumpspy
solver = mumpspy.MumpsSolver(system='real64')
@print_durations()
@profileit("mumpspy_profile")
def run_mumpspy():
    sol = solver.solve(A,b)
    return sol


## test pymumps
@print_durations()
@profileit("pymumps_profile")
def run_pymumps():
    sol = pymumps.spsolve(A,b)    
    return sol

## test pymumps with symmetric matrix
@print_durations()
@profileit("pymumps_sym_profile")
def run_pymumps_sym():
    ins = pymumps.DMumpsContext(par = 1, sym=1)
    ins.set_centralized_sparse(Atril)
    sol = b.copy()
    ins.set_rhs(sol)
    ins.set_silent()
    ins.run(job=6)
    del ins
    return sol

class pymumps_solver_sym:
    @print_durations()
    @profileit("pymumps_sym_factorize_profile")
    def __init__(self, Amat):
        self.ctx = pymumps.DMumpsContext(par = 1, sym=1)
        self.ctx.set_centralized_sparse(Amat)
        self.ctx.set_silent()
        self.ctx.run(job=4)

    @print_durations()
    @profileit("pymumps_sym_solve_profile")
    def solve(self, bvec):
        sol = bvec.copy()
        self.ctx.set_rhs(sol)
        self.ctx.run(job=3)
        return sol

## test python-mumps
inst = mumps.Context()
@print_durations()
@profileit("python-mumps_profile")
def run_pythonmumps():    
    inst.factor(A)
    sol = inst.solve(b)
    return sol

# ## test pardiso
# @print_durations()
# @profileit("pardiso_profile")
# def run_pardiso():  
#     sol = pypardiso.spsolve(A,b)


solA = run_scipy()
solB = run_mumpspy()
solC = run_pymumps()
solD = run_pythonmumps()
# run_pardiso()

__TOL = 1e-10
print("solution A-B: {}".format(numpy.all(numpy.abs(solA-solB)<__TOL)))
print("solution A-C: {}".format(numpy.all(numpy.abs(solA-solC)<__TOL)))
print("solution A-D: {}".format(numpy.all(numpy.abs(solA-solD)<__TOL)))
print("solution B-C: {}".format(numpy.all(numpy.abs(solB-solC)<__TOL)))

nb = 1e4

A = sp.rand(nb,nb,density=0.25, format='csr')
b = numpy.random.rand(int(nb))

print('Matrix size {}'.format(A.shape))

# solA = run_scipy()
# solB = run_mumpspy()
# solC = run_pymumps()
# solD = run_pythonmumps()
# # run_pardiso()

# __TOL = 1e-10
# print("solution B: {}".format(numpy.all(solA-solB<__TOL)))
# print("solution C: {}".format(numpy.all(solA-solC<__TOL)))
# print("solution D: {}".format(numpy.all(solA-solD<__TOL)))

################################################
################################################
################################################

nb = 1e3
band_size = 100

A, b, Atril = build_pb(nb, band_size)

print('symmetric matrix size {}'.format(A.shape))

solA = run_scipy()
solB = run_pymumps()
solC = run_pymumps_sym()


__TOL = 1e-9
print("solution A-B: {}".format(numpy.all(numpy.abs(solA-solB)<__TOL)))
print("solution A-C: {}".format(numpy.all(numpy.abs(solA-solC)<__TOL)))
print("solution B-C: {}".format(numpy.all(numpy.abs(solB-solC)<__TOL)))

print('with factorization')
solver = scipy_solver(A)
solA = solver.solve(b)

solverp = pymumps_solver_sym(Atril)
solB = solverp.solve(b)

__TOL = 1e-9
print("solution A-B: {}".format(numpy.all(numpy.abs(solA-solB)<__TOL)))


################################################
################################################
################################################

df = pd.DataFrame(columns=['nb','scipy','pymumps','pymumps_sym'])


nb = 1e5
for band in numpy.logspace(1,4,10):
    print('band size {}'.format(int(band)))
    # build pb
    A, b, Atril = build_pb(nb, int(band))
    #
    ticA = time.process_time()
    solA = run_scipy()
    tocA = time.process_time()
    ticB = time.process_time()
    solB = run_pymumps()
    tocB = time.process_time()
    ticC = time.process_time()
    solC = run_pymumps_sym()
    tocC = time.process_time()

    __TOL = 1e-9
    print("solution A-B: {}".format(numpy.all(numpy.abs(solA-solB)<__TOL)))
    print("solution A-C: {}".format(numpy.all(numpy.abs(solA-solC)<__TOL)))
    print("solution B-C: {}".format(numpy.all(numpy.abs(solB-solC)<__TOL)))

    dftmp = pd.DataFrame({'band':[band],'scipy':[tocA-ticA],'pymumps':[tocB-ticB],'pymumps_sym':[tocC-ticC]})
    df = pd.concat([df,dftmp], ignore_index=True)

print(df)