

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

from memtowel import MemoryTowel
towel = MemoryTowel()
towel.write_comm_memory(reset=True)

from global_params import *
import sys, os
sys.path.insert(0, os.getcwd())
from test_params import *


# Create bases and domain
start_init_time = time.time()
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=mesh)
towel.write_comm_memory()

# 3D Boussinesq
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','vz','wz'], ncc_cutoff=0, max_ncc_terms=max_ncc_terms)
problem.parameters['P0'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R0'] = (Rayleigh / Prandtl)**(-1/2)
problem.substitutions['P'] = "P0*exp(-z)"
problem.substitutions['R'] = "R0*exp(-z)"
problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation("dt(b) - P*(dx(dx(b)) + dy(dy(b)) + dz(bz)) - w         = - u*dx(b) - v*dy(b) - w*bz")
problem.add_equation("dt(u) - R*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + dx(p)     = - u*dx(u) - v*dy(u) - w*uz")
problem.add_equation("dt(v) - R*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + dy(p)     = - u*dx(v) - v*dy(v) - w*vz")
problem.add_equation("dt(w) - R*(dx(dx(w)) + dy(dy(w)) + dz(wz)) + dz(p) - b = - u*dx(w) - v*dy(w) - w*wz")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(b) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(v) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("integ_z(p) = 0", condition="(nx == 0) and (ny == 0)")
towel.write_comm_memory()

# Build solver
solver = problem.build_solver(timestepper, matsolver=de.matsolvers.matsolvers[matsolver.lower()])
solver.stop_iteration = iterations
logger.info('Solver built')
towel.write_comm_memory()

# Initial conditions
z = domain.grid(2)
b = solver.state['b']
bz = solver.state['bz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

# Perturbations damped at walls
zb, zt = z_basis.interval
b['g'] = 1e-3 * noise * (zt - z) * (z - zb) / (Lz / 2)**2
b.differentiate('z', out=bz)
towel.write_comm_memory()

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    dt = initial_dt
    while solver.ok:
        if (solver.iteration) % change_dt_cadence == 0:
            dt = dt * 0.9
        solver.step(dt)
        towel.write_comm_memory()
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

