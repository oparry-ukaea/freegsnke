# load_ext autoreload
# autoreload 2

import os
import matplotlib.pyplot as plt
import freegs4e
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial

from copy import deepcopy
from IPython.display import display, clear_output
import time
from timeit import default_timer as timer


jax.config.update("jax_enable_x64", False)

# # set paths
# os.environ["ACTIVE_COILS_PATH"] = f"../machine_configs/MAST-U/MAST-U_like_active_coils.pickle"
# os.environ["PASSIVE_COILS_PATH"] = f"../machine_configs/MAST-U/MAST-U_like_passive_coils.pickle"
# os.environ["WALL_PATH"] = f"../machine_configs/MAST-U/MAST-U_like_wall.pickle"
# os.environ["LIMITER_PATH"] = f"../machine_configs/MAST-U/MAST-U_like_limiter.pickle"

# build machine
from freegsnke import build_machine
tokamak = build_machine.tokamak(
    active_coils_path=f"../machine_configs/MAST-U/MAST-U_like_active_coils.pickle",
    passive_coils_path=f"../machine_configs/MAST-U/MAST-U_like_passive_coils.pickle",
    limiter_path=f"../machine_configs/MAST-U/MAST-U_like_limiter.pickle",
    wall_path=f"../machine_configs/MAST-U/MAST-U_like_wall.pickle",
)

from freegsnke import equilibrium_update
eq = equilibrium_update.Equilibrium(    tokamak=tokamak,
    Rmin=0.1, Rmax=2.0,   # Radial range
    Zmin=-2.2, Zmax=2.2,  # Vertical range
    nx=65,   # Number of grid points in the radial direction
    ny=129,  # Number of grid points in the vertical direction
    # psi=plasma_psi
)  

from freegsnke.jtor_update import ConstrainPaxisIp

# Diverted plasma
# profiles = ConstrainPaxisIp(
#     eq=eq,
#     paxis=8e3,
#     Ip=6e5,
#     fvac=0.5,
#     alpha_m=1.8,
#     alpha_n=1.2
# )
# Limited plasma 
profiles = ConstrainPaxisIp(
    eq=eq,
    paxis=6e3,
    Ip=4e5,
    fvac=0.5,
    alpha_m=1.8,
    alpha_n=1.2
)

from freegsnke import GSstaticsolver
GSStaticSolver = GSstaticsolver.NKGSsolver(eq)

# load the coil currents
import pickle
# with open('simple_diverted_currents_PaxisIp.pk', 'rb') as f:
with open('simple_limited_currents_PaxisIp.pk', 'rb') as f:

    currents_dict = pickle.load(f)
    
# assign currents to the eq object
for key in currents_dict.keys():
    eq.tokamak[key].current = currents_dict[key]
    
eq.tokamak.getCurrents()
eq1=deepcopy(eq)

#call the solver
t1=timer()
GSStaticSolver.solve(
    eq=eq,
    profiles=profiles,
    constrain=None,
    target_relative_tolerance=1e-9,
    max_solving_iterations=50,
    verbose=True
    )
print("time GS solve=",timer()-t1)

# Do it using JAX solver now
from freegsnke import j_GSstaticsolver, j_limiter_func

jLimiter = j_limiter_func.Limiter_handler(eq, eq.tokamak.limiter)

from freegsnke.j_jtor import JConstrainPaxisIp, JLao85

# jProfile = JConstrainPaxisIp(
#     paxis=8e3,
#     Ip=6e5,
#     fvac=0.5,
#     alpha_m=1.8,
#     alpha_n=1.2
# )

jProfile = JConstrainPaxisIp(
    paxis=6e3,
    Ip=4e5,
    fvac=0.5,
    alpha_m=1.8,
    alpha_n=1.2
)

# alpha, beta = profiles.Lao_parameters(4,4)
# jProfile = JLao85(Ip=6e5,fvac=0.5,alpha=alpha,beta=beta)

# Initialise the solver
jGS = j_GSstaticsolver.NKGSsolver(eq, jProfile, jLimiter)

# fig1, ax1 = plt.subplots(1, 1, figsize=(4, 8), dpi=80)
# ax1.grid(True, which='both')
# eq.plot(axis=ax1, show=False)
# eq.tokamak.plot(axis=ax1, show=False)
# ax1.set_xlim(0.1, 2.15)
# ax1.set_ylim(-2.25, 2.25)
# plt.tight_layout()

# Allocate current vector and tuple of profile parameters (Ip, (alpha_m, alpha_n, p_axis))
currentlist=eq.tokamak.getCurrents()
jcurr=jnp.array([1.0*currentlist[key] for key in currentlist.keys()])
jProfilePars=jProfile.init_params

t1=timer()
# First JAX solve - will take longer because it is compiling code,
# subsequent calls should be much faster
psi_j=jGS.solve(
    eq.psi(),
    jProfilePars,
    jcurr,
    target_relative_tolerance=1e-4,
    use_newton=True,
    verbose=True)
print("time Jax GS solve=",timer()-t1)

# check psi field
print("Difference between Jax and default solver:",jnp.linalg.norm(psi_j-eq.psi()))

fig1, ax1 = plt.subplots(1, 1, figsize=(5, 8), dpi=80)
#ax1.grid(True, which='both')
#plt.contourf(eq.R, eq.Z, np.abs(eq.psi()-psi_j),50)
plt.contour(eq.R,eq.Z,eq.psi(),20,colors='black')
plt.contourf(eq.R, eq.Z, np.abs((eq.psi()-psi_j)),50,cmap='hot_r')
eq.tokamak.plot(axis=ax1, show=False)
plt.plot(eq.tokamak.wall.R, eq.tokamak.wall.Z, 'k', 3.0)
ax1.set_xlim(0.1, 2.15)
ax1.set_ylim(-2.25, 2.25)
plt.tight_layout()
plt.colorbar(); plt.savefig('fig_newton_limiter_fp32.png',format='png')


# Test cost function as a function of Profile Parameters and current vec
def j_func(p,j):
    psi = jGS.solve(eq.psi(),p,j,target_relative_tolerance=1e-7,use_newton=False,verbose=False)
    
    oo, xx = jGS.critpoints(psi)

    j=oo[0][0] # X coordinate of primary o-point

    return j

# tpsi = jnp.dot(jcurr, jGS.pgreen)
# ppsi = psi_j.reshape(-1) - tpsi

# def r_func(p):
#     r = jGS.F_function(ppsi,tpsi,p)

#     return jnp.linalg.norm(r)

# j0 = j_func(jProfilePars,jcurr)
# t1=timer()
# djdprof, djdcurr = jax.jacfwd(j_func,argnums=(0,1,))(jProfilePars,jcurr) # Forward mode will take longer
# print("time Jax GS Jacfwd=",timer()-t1)

# t1=timer()
# djdprof, djdcurr = jax.grad(j_func,argnums=(0,1,))(jProfilePars,jcurr) # Reverse-mode
# print("time Jax GS Grad=",timer()-t1)

# print("Profile parameter sensitivity: ", djdprof)
# print("Coil current sensitivity: ", djdcurr)

# Manually check gradients by finite difference
# for j in range(0,5):
#     dj = 1e-4*jnp.pow(0.5,j)
#     p1=(jnp.array(profiles.Ip),jnp.array([profiles.alpha_m + dj,profiles.alpha_n,profiles.paxis]))
#     # c1 = jcurr.at[1].set(jcurr[1]+dj)
#     j1 = j_func(p1,jcurr)
#     print(dj,(j1-j0)/dj)

# Automatically check gradients by finite difference
# from jax.test_util import check_grads

# check_grads(j_func,(jProfilePars,jcurr,),order=1,modes='fwd,rev')