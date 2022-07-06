"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. It should take about a cpu-minute to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
from configparser import ConfigParser
import sys
from pathlib import Path
logger = logging.getLogger(__name__)

# Parsing .cfg passed
config_file = Path(sys.argv[-1])
logger.info("Running with config file {}".format(str(config_file)))

# Setting global params
runconfig = ConfigParser()
runconfig.read(str(config_file))

# Parameters
timestepper = d3.RK222
params = runconfig['params']
Nx = params.getint('nx')
Nz = params.getint('nz')
Lx = params.getfloat('Lx')
Lz = params.getfloat('Lz')
Rayleigh = params.getfloat('Rayleigh')
Prandtl = params.getfloat('Prandtl')
dealias = params.getfloat('dealias')

dtype = np.float64

run_params = runconfig['run']
restart = run_params.get('restart_file')
stop_wall_time = run_params.getfloat('stop_wall_time')
stop_sim_time = run_params.getfloat('stop_sim_time')
stop_iteration = run_params.getint('stop_iteration')
max_timestep = run_params.getfloat('max_timestep')

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

integ = lambda A: d3.Integrate(d3.Integrate(A, 'x'), 'z')

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
problem.add_equation("b(z=0) = Lz")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b['g'] += Lz - z # Add linear background

# define kinetic energy
KE = 0.5 * u@u 
KE.name = 'KE'

## Analysis

# creating output directory, automatically named for the config file
datadir = Path("rbc_runs") / config_file.stem

if dist.comm.rank == 0:
    if not datadir.exists():
        datadir.mkdir(parents=True)


# create separate tasks for different dimensional data

# checkpoints (state vector)
checkpoints = solver.evaluator.add_file_handler(datadir/Path('checkpoints'), wall_dt=3300, max_writes=1, parallel='gather')
checkpoints.add_tasks(solver.state) # save the entire state vector


# snapshots (usually 2D even in 3D simulations; quantities nice for vizualization or analysis)
snapshots = solver.evaluator.add_file_handler(datadir/Path('snapshots'), sim_dt=0.25, max_writes=50, parallel='gather')
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# EXERCISE: add profiles, 1D outputs written out at some time interval
# Could be Nusselt number as a function of z (integrated in x, for example)...
# these would make spacetime plots in post processing


# timeseries (scalar functions of time)
timeseries = solver.evaluator.add_file_handler(datadir/Path('timeseries'), iter=10)
timeseries.add_task(integ(KE), name='KE')
## could add volume averaged Nusselt # here...

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties: for *real time* solution monitoring
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

