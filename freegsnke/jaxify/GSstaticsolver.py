import numpy as np
import jax
import jax.numpy as jnp
import freegs4e
from . import nk_solver
import interpax as ix
import jax.scipy as jsp
import equinox as eqx
from timeit import default_timer as timer
from functools import partial
import warnings
from jax.experimental import sparse as jexsp

# Physical constants
mu0 = 4e-7 * jnp.pi

class NKGSsolver(eqx.Module):

    """Solver for the non-linear forward Grad Shafranov (GS) 
    static problem. Here, the GS problem is written as a root
    problem in the plasma flux psi. This root problem is 
    passed to and solved by the NewtonKrylov solver itself,
    class nk_solver.

    The solution domain is set at instantiation time, through the 
    input freeGS equilibrium object.

    The non-linear solver itself is called using the 'solve' method.
    """
    R: jax.Array
    Z: jax.Array
    dRdZ: float
    pgreen: jax.Array
    bndry_indices: jax.Array
    greenfunc: jax.Array
    Ainv: jax.Array
    profile: eqx.Module
    limiter: eqx.Module
     
    def __init__(self, eq, profile, limiter_func):

        """Instantiates the solver object.
        Based on the domain grid of the input equilibrium object, it prepares
        - the linear solver 'self.linear_GS_solver'
        - the response matrix of boundary grid points 'self.greens_boundary'


        Parameters
        ----------
        eq : a freeGS equilibrium object.
             The domain grid defined by (eq.R, eq.Z) is the solution domain 
             adopted for the GS problems. Calls to the nonlinear solver will
             use the grid domain set at instantiation time. Re-instantiation 
             is necessary in order to change the propertes of either grid or
             domain.

        """
     
   
        #eq is an Equilibrium instance, it has to have the same domain and grid as 
        #the ones the solver will be called on
        
        R = jnp.asarray(eq.R)
        Z = jnp.asarray(eq.Z)
        self.R = R
        self.Z = Z
        R_1D = R[:,0]
        Z_1D = Z[0,:]
        
        #for reshaping
        nx,ny = np.shape(R)
        
        #for integration
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        self.dRdZ = dR*dZ

        greenlist={}
        currentlist={}
        ii=0
        for item in eq._pgreen:
            greenitem=0.0
            if (type(eq._pgreen[item]) is dict):
                jj=0
                for obj in eq._pgreen[item]:
                    greenitem=greenitem+eq._pgreen[item][obj]*eq.tokamak.coils[ii][1].coils[jj][2]
                    jj=jj+1
            else:
                greenitem=eq._pgreen[item]
            greenlist[item]=jnp.asarray(greenitem)
            currentlist[item]=jnp.asarray(eq.tokamak.getCurrents()[item])
            ii=ii+1
        
        self.pgreen=jnp.array([greenlist[key].reshape(-1) for key in greenlist.keys()])

        #linear solver for del*Psi=RHS
        generator=freegs4e.gradshafranov.GSsparse4thOrder(eq.R[0,0],eq.R[-1,0],eq.Z[0,0],eq.Z[0,-1])
         
        self.Ainv = jnp.linalg.inv(jexsp.BCSR.from_scipy_sparse(generator(nx,ny)).todense())

        # List of indices on the boundary
        bndry_indices = np.concatenate(
            [
                [(x, 0) for x in range(nx)],
                [(x, ny - 1) for x in range(nx)],
                [(0, y) for y in np.arange(1,ny-1)],
                [(nx - 1, y) for y in np.arange(1,ny-1)],
            ]
        )
        self.bndry_indices = jnp.asarray(bndry_indices)
        
        # matrices of responses of boundary locations to each grid positions
        greenfunc = Greens(R[jnp.newaxis,:,:], 
                           Z[jnp.newaxis,:,:], 
                           R_1D[self.bndry_indices[:,0]][:,jnp.newaxis,jnp.newaxis], 
                           Z_1D[self.bndry_indices[:,1]][:,jnp.newaxis,jnp.newaxis])
        # Prevent infinity/nan by removing Greens(x,y;x,y) 
        zeros = jnp.ones_like(greenfunc)
        zeros=zeros.at[jnp.arange(len(self.bndry_indices)), self.bndry_indices[:,0], self.bndry_indices[:,1]].set(0.0)
        self.greenfunc = greenfunc*zeros*self.dRdZ

        self.profile = profile
        self.limiter = limiter_func
        # zeromach = jnp.asarray(jnp.pi)
		# while (1.0+zeromach/2.0 > 1.0):
		# 	zeromach = zeromach/2.0

		# self.minprec = 2.*zeromach

    @jax.jit
    def gs_oper(self, psi):
        """
        Apply the full elliptic operator to 2D field L*psi_plasma
        """

        nx,ny = psi.shape
        dR = self.R[1, 0] - self.R[0, 0]
        dZ = self.Z[0, 1] - self.Z[0, 0]

        b = jnp.zeros_like(psi)

        invdR2 = 1.0 / dR ** 2
        invdZ2 = 1.0 / dZ ** 2

        iR = 1.0/(2.0*self.R*dR)

        # set 2ndorder accurate derivatives first (at 1 to N-1)
        d2R=invdR2*(psi[2:,1:-1]+psi[:-2,1:-1]-2.0*psi[1:-1,1:-1])
        d2Z=invdZ2*(psi[1:-1,2:]+psi[1:-1,:-2]-2.0*psi[1:-1,1:-1])
        iRdR=iR[1:-1,1:-1]*(psi[2:,1:-1]-psi[:-2,1:-1])

        b=b.at[1:-1,1:-1].set(d2R+d2Z-iRdR)

        # set fourth order derivatives first (at 2 to N-2)
        d2R=invdR2*(-1/12*psi[4:,2:-2]+4/3*psi[3:-1,2:-2]-2.5*psi[2:-2,2:-2]+4/3*psi[1:-3,2:-2]-1/12*psi[:-4,2:-2])
        d2Z=invdZ2*(-1/12*psi[2:-2,4:]+4/3*psi[2:-2,3:-1]-2.5*psi[2:-2,2:-2]+4/3*psi[2:-2,1:-3]-1/12*psi[2:-2,:-4])
        iRdR=2.0*iR[2:-2,2:-2]*(-1/12*psi[4:,2:-2]+2/3*psi[3:-1,2:-2]-2/3*psi[1:-3,2:-2]+1/12*psi[:-4,2:-2])

        b=b.at[2:-2,2:-2].set(d2R+d2Z-iRdR)

        return b
    
    @jax.jit
    def critpoints(self, psi):
        dR = self.R[1, 0] - self.R[0, 0]
        dZ = self.Z[0, 1] - self.Z[0, 0]
        r1d = self.R[:,0]
        z1d = self.Z[0,:]
        f_psi = ix.Interpolator2D(r1d,z1d,psi)

        Bp2=jnp.zeros_like(psi)
        psiR=Bp2.copy()
        psiZ=Bp2.copy()
        psiR=psiR.at[1:-1,1:-1].set(0.5*(psi[2:,1:-1]-psi[:-2,1:-1])/dR)
        psiZ=psiZ.at[1:-1,1:-1].set(0.5*(psi[1:-1,2:]-psi[1:-1,:-2])/dZ)

        Bp2=Bp2.at[:,:].set((psiR**2 + psiZ**2))

        nx, ny = Bp2.shape
        # start off by finding coarse values of Bp2 closest to 0.0 as in Ben Dudsons's routine
        A=Bp2
        A2=jnp.zeros_like(A)

        A2=A2.at[1:-1,1:-1].set(jnp.where(
            (A[1:-1,1:-1]<A[:-2,1:-1]) &
            (A[1:-1,1:-1]<A[2:,1:-1]) &
            (A[1:-1,1:-1]<A[1:-1:,:-2]) &
            (A[1:-1,1:-1]<A[1:-1:,2:]) &
            (A[1:-1,1:-1]<A[2:,2:]) &
            (A[1:-1,1:-1]<A[2:,:-2]) &
            (A[1:-1,1:-1]<A[:-2,2:]) &
            (A[1:-1,1:-1]<A[:-2,:-2]),1,0))

        ir, iz = jnp.nonzero(A2,size=30)

        @jax.jit
        def _calc_point(i,j,psi,psiR,psiZ,dR,dZ):
            fR , fZ = psiR[i,j], psiZ[i,j]
            fRR = (psi[i+1,j]-2*psi[i,j]+psi[i-1,j])/dR**2
            fZZ = (psi[i,j+1]-2*psi[i,j]+psi[i,j-1])/dZ**2
            fRZ = 0.25*(psi[i+1,j+1] + psi[i-1,j-1] -psi[i-1,j+1] - psi[i+1,j-1] )/(dR*dZ)
            det = fRR*fZZ-fRZ**2
            delta_R = -(fR*fZZ-fRZ*fZ)/det
            delta_Z = -(fZ*fRR-fRZ*fR)/det

            return det, delta_R, delta_Z

        _calc_point_vmap=jax.jit(jax.vmap(_calc_point,in_axes=(0,0,None,None,None,None,None)))

        det, deltaR, deltaZ = _calc_point_vmap(ir,iz,psi,psiR,psiZ,dR,dZ)
        est_psi = psi[ir,iz]+0.5*(psiR[ir,iz]*deltaR + psiZ[ir,iz]*deltaZ)
        est_R, est_Z = self.R[ir,iz] + deltaR, self.Z[ir,iz] + deltaZ
        optmsk = jnp.where((jnp.abs(deltaR)<1.5*dR) & (jnp.abs(deltaZ)<1.5*dZ) & (det>0.0) & (ir>0),1,-1000)
        xptmsk = jnp.where((jnp.abs(deltaR)<1.5*dR) & (jnp.abs(deltaZ)<1.5*dZ) & (det<0.0) & (ir>0),1,-1000)

        opoints = jnp.vstack((optmsk*est_R,optmsk*est_Z,optmsk*est_psi)).T
        xpoints = jnp.vstack((xptmsk*est_R,xptmsk*est_Z,xptmsk*est_psi)).T
        
        # Find primary O-point by sorting by distance from middle of domain
        Rmid = 0.5 * (self.R[-1, 0] + self.R[0, 0])
        Zmid = 0.5 * (self.Z[0, -1] + self.Z[0, 0])
        Dmid = (opoints[:,0] - Rmid) ** 2 + (opoints[:,1]-Zmid) **2
        isort = jnp.argsort(Dmid)

        opoints=opoints[isort,:]

        def _check_xpoint(opt,fpsi,xpt):
            [r0,z0,psi0]=opt
            [rx,zx,psix]=xpt
            rl=jnp.linspace(r0,rx,num=50)
            zl=jnp.linspace(z0,zx,num=50)
            psil=jnp.sign(psix-psi0)*fpsi(rl,zl)
            psimax=jnp.amax(psil)
            idmin = jnp.argmin(psil)
            xcheck = (psimax-psil[-1])/(psimax-psil[0])<0.001
            ocheck = ((rl[idmin]-r0)**2 + (zl[idmin]-z0)**2)<1e-4

            return jnp.logical_and(xcheck,ocheck)

        # Check xpoints to ensure they are valid using monotinicity principle
        _check_xpoint_vmap=(jax.vmap(_check_xpoint,in_axes=(None,None,0)))
        xocheck = _check_xpoint_vmap(opoints[0,:],f_psi,xpoints)
        xptfilt = jnp.where(xocheck,1,1000)
        xpoints = xpoints.at[:,2].set(xpoints[:,2]*xptfilt)

        # Sort X-points by distance to primary O-point in psi space
        psi_axis = opoints[0,2]
        Daxis = (xpoints[:,2] - psi_axis) ** 2
        xsort = jnp.argsort(Daxis)

        xpoints=xpoints[xsort,:]

        return opoints, xpoints

    @jax.jit
    def mask(self, psi, opoint, xpoint):

        mask=jnp.zeros(psi.shape)

        Ro, Zo, psio = opoint[0]
        Rx, Zx, psix = xpoint[0]

        # Normalise psi
        psin = (psi - psio) / (psix - psio)

        # Condition that 0 < psin < 1
        s1=jnp.where(psin>=0,1,0)*jnp.where(psin<=1,1,0)

        # Condition that (R-Rx)*(R0-Rx) + (Z-Zx)*(Z0-ZX) > 0
        m2=(self.R-Rx)*(Ro-Rx) + (self.Z-Zx)*(Zo-Zx)
        s2=jnp.where(m2>0,1,0)

        m3=(self.R-xpoint[1,0])*(Ro-xpoint[1,0]) + (self.Z-xpoint[1,1])*(Zo-xpoint[1,1])
        s3=jnp.where(m3>0,1,0)

        mask=s1*s2*s3

        return mask

    @jax.jit
    def jtor(self, profilePars, psi):

        opts, xpts = self.critpoints(psi)
        diverted_mask = self.mask(psi, opts, xpts)
        psib = xpts[0,2]
        psi_bound, limiter_mask = self.limiter.core_mask_limiter(self.R, self.Z, psi, psib, diverted_mask, self.limiter.limiter_mask_out)
        psi_axis = opts[0,2]
        jtor = self.profile.jtor(self, profilePars, psi, psi_axis, psi_bound, limiter_mask)

        return jtor

    @jax.jit
    def freeboundary(self, profilePars, psi):
        """Imposes boundary conditions on set of boundary points. 

        Parameters
        ----------
        plasma_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the plasma
        tokamak_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the tokamak alone, including all metal currents,
            in both active coils and passive structures
        profiles : freeGS profile object
            profile object describing target plasma properties, 
            used to calculate current density jtor
        """
      
        #jtor and RHS given tokamak_psi above and the input plasma_psi
        nx,ny=self.R.shape
        ppsi = psi.reshape(nx,ny)
        jtor = self.jtor(profilePars, ppsi)    
        rhs = -mu0*self.R*jtor
        
        #calculates and assignes boundary conditions
        psi_boundary = jnp.zeros_like(self.R)
        psi_bnd = jnp.sum(self.greenfunc*jtor[jnp.newaxis,:,:], axis=(-1,-2))
        psi_boundary=psi_boundary.at[:, 0].set(psi_bnd[:nx])
        psi_boundary=psi_boundary.at[:, -1].set(psi_bnd[nx:2*nx])
        psi_boundary=psi_boundary.at[0, 1:ny-1].set(psi_bnd[2*nx:2*nx+ny-2])
        psi_boundary=psi_boundary.at[-1, 1:ny-1].set(psi_bnd[2*nx+ny-2:])

        rhs=rhs.at[0, 1:ny-1].set(psi_boundary[0, 1:ny-1])
        rhs=rhs.at[:, 0].set(psi_boundary[:, 0])
        rhs=rhs.at[-1, 1:ny-1].set(psi_boundary[-1, 1:ny-1])
        rhs=rhs.at[:, -1].set(psi_boundary[:, -1])

        return rhs

    @jax.jit
    def F_function2(self, plasma_psi, tokamak_psi, profilePars):

        nx,ny = self.R.shape
        psi = plasma_psi + tokamak_psi
        rhs = self.freeboundary(profilePars, psi)
        ppsi = plasma_psi.reshape(nx,ny)
        resid = self.gs_oper(ppsi) - rhs

        # Apply boundary condition to residual
        resid = resid.at[0,:].set(ppsi[0,:]-rhs[0,:])
        resid = resid.at[:,0].set(ppsi[:,0]-rhs[:,0])
        resid = resid.at[-1,:].set(ppsi[-1,:]-rhs[-1,:])
        resid = resid.at[:,-1].set(ppsi[:,-1]-rhs[:,-1])

        return resid.reshape(-1)
    
    @jax.jit
    def F_function(self, plasma_psi, tokamak_psi, profilePars): 
        """Nonlinear Grad Shafranov equation written as a root problem
        F(plasma_psi) \equiv [\delta* - J](plasma_psi)
        The plasma_psi that solves the Grad Shafranov problem satisfies
        F(plasma_psi) = [\delta* - J](plasma_psi) = 0

        
        Parameters
        ----------
        plasma_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the plasma
        tokamak_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the tokamak alone, including all metal currents,
            in both active coils and passive structures
        profiles : freeGS profile object
            profile object describing target plasma properties, 
            used to calculate current density jtor
        
        Returns
        -------
        residual : np.array of size eq.nx*eq.ny
            residual of the GS equation
        """ 
        psi = plasma_psi + tokamak_psi
        rhs = self.freeboundary(profilePars, psi)
        residual = plasma_psi - jnp.dot(self.Ainv,rhs.reshape(-1))

        return residual

    @jax.jit
    def relative_norm_residual(self, res, psi):
        """Calculates a normalised relative residual, based on linalg.norm

        Parameters
        ----------
        res : ndarray
            Residual
        psi : ndarray
            plasma_psi

        Returns
        -------
        float
            Relative normalised residual
        """
        return jnp.linalg.norm(res) / jnp.linalg.norm(psi)

    @jax.jit 
    def relative_del_residual(self, res, psi):
        """Calculates a normalised relative residual, based on the norm max(.) - min(.)

        Parameters
        ----------
        res : ndarray
            Residual
        psi : ndarray
            plasma_psi

        Returns
        -------
        float, float
            Relative normalised residual, norm(plasma_psi)
        """
        del_psi = jnp.amax(psi) - jnp.amin(psi)
        del_res = jnp.amax(res) - jnp.amin(res)
        return del_res / del_psi, del_psi

    def solve(self,     
        init_psi,
        profilePars,
        currentvec,
        target_relative_tolerance,
        use_newton=False,
        lag_Jacobian=1, 
        max_solving_iterations=50,
        Picard_handover=0.15,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.2,  
        max_n_directions=8,
        clip=10,
        verbose=False,
        max_rel_update_size=0.2,
    ):
        
        """The method that actually solves the GS problem.
        The problem is specified by the 2 freeGS objects eq and profiles.
        The first specifies the metal currents (throught eq.tokamak)
        and the second specifies the desired plasma properties 
        (i.e. plasma current and profile functions).
        
        The plasma_psi which solves the given GS problem is assigned to 
        the input eq, and can be found at eq.plasma_psi.

        Parameters
        ----------
        profiles : freeGS profile object
            Specifies the target properties of the plasma.
            These are used to calculate Jtor(psi)
        target_relative_tolerance : float
            NK iterations are interrupted when this criterion is 
            satisfied. Relative convergence
        max_solving_iterations : int
            NK iterations are interrupted when this limit is surpassed
        Picard_handover : float
            Value of relative tolerance above which a Picard iteration
            is performed instead of a full NK call
        step_size : float
            l2 norm of proposed step
        scaling_with_n : float
            allows to further scale dx candidate steps by factor
            (1 + self.n_it)**scaling_with_n
        target_relative_explained_residual : float between 0 and 1
            terminates iteration when exploration can explain this 
            fraction of the initial residual R0
        max_n_directions : int
            terminates iteration even though condition on 
            explained residual is not met
        max_Arnoldi_iterations : int
            terminates iteration after attempting to explore
            this number of directions
        max_collinearity : float between 0 and 1
            rejects a candidate direction if resulting residual 
            is collinear to any of those stored previously
        clip : float
            maximum step size for each explored direction, in units 
            of exploratory step dx_i
        threshold : float 
            catches cases of untreated (partial) collinearity 
        clip_hard : float
            maximum step size for cases of untreated (partial) collinearity
        
        """
        
        nx,ny=init_psi.shape
        tokamak_psi = jnp.dot(currentvec,self.pgreen)
        trial_plasma_psi = jnp.asarray(init_psi).reshape(-1) - tokamak_psi

        # update solution
        if (use_newton):
            solver_params = (
                            target_relative_tolerance, 
                            max_solving_iterations,
                            Picard_handover,
                            lag_Jacobian,
                            verbose,
                            )
            plasma_psi, _ = nsolve(self,
                            solver_params,
                            trial_plasma_psi,
                            tokamak_psi,
                            profilePars,
                            )
        else:
            solver_params = (
                            target_relative_tolerance, 
                            max_solving_iterations,
                            Picard_handover,
                            step_size,
                            scaling_with_n,
                            target_relative_unexplained_residual,  
                            max_n_directions,
                            clip,
                            verbose,
                            max_rel_update_size,
                            )
            plasma_psi, _ = nksolve(self,
                            solver_params,
                            trial_plasma_psi,
                            tokamak_psi,
                            profilePars,
                            )
    
        # return new solution
        return (plasma_psi+tokamak_psi).reshape(nx,ny)

@partial(jax.custom_jvp, nondiff_argnums=(0,1))
def nksolve(solver,
            solver_params,
            trial_plasma_psi,
            tokamak_psi, 
            profilePars, 
            ):

    trial_plasma_psi = jax.lax.stop_gradient(trial_plasma_psi)

    # Unpack solver parameters
    (target_relative_tolerance, 
    max_solving_iterations,
    Picard_handover,
    step_size,
    scaling_with_n,
    target_relative_unexplained_residual,  
    max_n_directions,
    clip,
    verbose,
    max_rel_update_size) = solver_params 

    res0 = solver.F_function(trial_plasma_psi, tokamak_psi, profilePars)
    norm_rel_change = solver.relative_norm_residual(res0, trial_plasma_psi)
    rel_change, del_psi = solver.relative_del_residual(res0, trial_plasma_psi)
    relative_change = 1.0 * rel_change
    history_norm_rel_change = [norm_rel_change]
    starting_direction = res0
    log = []
    picard_flag = 0
    nx,ny = solver.R.shape

    log.append(f"Initial relative error = {rel_change:.2e}")
    if verbose:
        for x in log:
            print(x)
    
    iter = 0
        
    def Ffunc(x):
        return solver.F_function(x, tokamak_psi, profilePars)

    def dFfunc(x, dx):
        r, dr = jax.jvp(Ffunc, (x,), (dx,))
        return dr

    def condfun(rel_change, iter):
        return jnp.logical_and(
                    rel_change > target_relative_tolerance, 
                    (iter < max_solving_iterations)
                    )
    rel_change = jnp.maximum(rel_change,2*target_relative_tolerance)
    while (condfun(rel_change, iter)):
        if rel_change > Picard_handover:
            log.append("-----")
            log.append("Picard iteration: " + str(iter))
            # using Picard instead of NK
            # make picard update to the flux up-down symmetric
            # this combats the instability of picard iterations
            res0_2d = res0.reshape(nx,ny)
            update_sym = -0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
            update_nonsym = -1.0 * res0
            alpha = jnp.array((picard_flag < 3)).astype('float32')
            update = alpha*update_sym + (1-alpha)*update_nonsym
            picard_flag  = picard_flag + 1
        else:
            log.append("-----")
            log.append("Newton-Krylov iteration: " + str(iter))
            update, Abasis = nk_solver.Arnoldi_iteration(x0=trial_plasma_psi, #trial_current expansion point
                                                dx=starting_direction, #first vector for current basis
                                                R0=res0, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                                F_function=lambda u: Ffunc(u),
                                                step_size=step_size,
                                                scaling_with_n=scaling_with_n,
                                                target_relative_unexplained_residual=target_relative_unexplained_residual,  
                                                max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
                                                clip=clip)
            log.append(
                    f"...number of Krylov vectors used =  {(Abasis[1].shape[1])}"
                )
        
        del_update = jnp.amax(update) - jnp.amin(update)
        if del_update / del_psi > max_rel_update_size:
            # Reduce the size of the update as found too large
            update *= jnp.abs(max_rel_update_size * del_psi / del_update)
        
        trial_plasma_psi = trial_plasma_psi + update
        res0 = solver.F_function(trial_plasma_psi, tokamak_psi, profilePars)
        starting_direction = res0
        norm_rel_change = solver.relative_norm_residual(res0, trial_plasma_psi)
        rel_change, del_psi = solver.relative_del_residual(res0, trial_plasma_psi)
        relative_change = 1.0 * rel_change
        history_norm_rel_change.append(norm_rel_change)
        log.append(f"...relative error =  {rel_change:.2e}")
        log.append("-----")
        if verbose:
            for x in log:
                print(x)

        log = []
        iter +=1

    return (trial_plasma_psi, Abasis)

@nksolve.defjvp
def nksolve_jvp(solver, solver_params, primals, tangents):

    trial_plasma_psi, tokamak_psi, profilePars, = primals
    dppsi, dtpsi, dprofile, = tangents

    # Unpack solver parameters
    (target_relative_tolerance, 
    max_solving_iterations,
    Picard_handover,
    step_size,
    scaling_with_n,
    target_relative_unexplained_residual,  
    max_n_directions,
    clip,
    verbose,
    max_rel_update_size) = solver_params 

    opsi, Abasis = nksolve(solver, solver_params, trial_plasma_psi, tokamak_psi, profilePars)
    psi0, Gloc, Qloc = Abasis

    def Ffunc(x):
        return solver.F_function(x, tokamak_psi, profilePars)

    def Floc(t, p):
        return solver.F_function(psi0, t ,p)

    def dFfunc(dx):
        r, dr = jax.jvp(Ffunc, (psi0,), (dx,))
        return dr

    def solve_with_gmres(A,b):
        return jax.scipy.sparse.linalg.gmres(A,b,x0=b,restart=10,solve_method='incremental',atol=1e-6)[0]

    res0, jvp_res0 = jax.jvp(Floc,(tokamak_psi, profilePars), (dtpsi, dprofile,))
    tangent_out = jax.lax.custom_linear_solve(dFfunc, -jvp_res0, solve=solve_with_gmres, transpose_solve=solve_with_gmres)
    
    primal_out = (opsi, Abasis)

    return (primal_out, (tangent_out,(jnp.zeros_like(psi0), jnp.zeros_like(Gloc), jnp.zeros_like(Qloc))) )

@partial(jax.custom_jvp,nondiff_argnums=(0,1))
def nsolve(solver,
            solver_params,
            trial_plasma_psi,
            tokamak_psi, 
            profilePars, 
            ):

    trial_plasma_psi = jax.lax.stop_gradient(trial_plasma_psi)

    # Unpack solver parameters
    (target_relative_tolerance, 
    max_solving_iterations,
    Picard_handover,
    lag_Jacobian,
    verbose) = solver_params 

    res0 = solver.F_function2(trial_plasma_psi, tokamak_psi, profilePars)
    norm_rel_change = solver.relative_norm_residual(res0, trial_plasma_psi)
    rel_change, del_psi = solver.relative_del_residual(res0, trial_plasma_psi)
    relative_change = 1.0 * rel_change
    history_norm_rel_change = [norm_rel_change]
    log = []
    picard_flag = 0
    nx,ny = solver.R.shape

    log.append(f"Initial relative error = {rel_change:.2e}")
    if verbose:
        for x in log:
            print(x)
    
    iter = 0
        
    def Ffunc(x):
        return solver.F_function2(x, tokamak_psi, profilePars)

    def dFfunc(x, dx):
        r, dr = jax.jvp(Ffunc, (x,), (dx,))
        return dr

    def condfun(rel_change, iter):
        return jnp.logical_and(
                    rel_change > target_relative_tolerance, 
                    (iter < max_solving_iterations)
                    )
    rel_change = jnp.maximum(rel_change,2*target_relative_tolerance)

    while (condfun(rel_change, iter)):
        log.append("-----")
        log.append("Newton iteration: " + str(iter))
        psi0 = trial_plasma_psi
        if(jnp.mod(iter,lag_Jacobian)==0):
            Jmat = jax.jacfwd(solver.F_function2,argnums=0)(trial_plasma_psi, tokamak_psi, profilePars)
        update = jnp.linalg.solve(Jmat,res0)
        err=jnp.linalg.norm(update)
        
        trial_plasma_psi = trial_plasma_psi - update
        res0 = solver.F_function2(trial_plasma_psi, tokamak_psi, profilePars)
        norm_rel_change = solver.relative_norm_residual(update, trial_plasma_psi)
        rel_change, del_psi = solver.relative_del_residual(update, trial_plasma_psi)
        relative_change = 1.0 * rel_change
        history_norm_rel_change.append(norm_rel_change)
        log.append(f"...relative error =  {rel_change:.2e}")
        log.append(f"...norm error =  {err:.2e}")
        log.append("-----")

        if verbose:
            for x in log:
                print(x)

        log = []
        iter +=1

    return (trial_plasma_psi, (psi0, Jmat))

@nsolve.defjvp
def nsolve_jvp(solver, solver_params, primals, tangents):

    trial_plasma_psi, tokamak_psi, profilePars, = primals
    dppsi, dtpsi, dprofile, = tangents

    opsi, basis = nsolve(solver, solver_params, trial_plasma_psi, tokamak_psi, profilePars)
    psi0, Jmat = basis

    def Ffunc(x):
        return solver.F_function2(x, tokamak_psi, profilePars)

    def Floc(t, p):
        return solver.F_function2(psi0, t ,p)

    res0, jvp_res0 = jax.jvp(Floc,(tokamak_psi, profilePars), (dtpsi, dprofile,))
    primal_out = (opsi, basis)
    tangent_out = jnp.linalg.solve(Jmat, -jvp_res0)

    return (primal_out, (tangent_out, (jnp.zeros_like(psi0), jnp.zeros_like(Jmat))))

# def nksolve_fwd(solver, solver_params, trial_plasma_psi, tokamak_psi, profilePars):

#     psi_guess, basis = nksolve(solver, solver_params, trial_plasma_psi, tokamak_psi, profilePars)

#     return (psi_guess, basis), (psi_guess,tokamak_psi, profilePars, basis)

# def nksolve_bwd(solver, solver_params, res, v):
    
#     psi_guess, tokamak_psi, profilePars, basis = res
#     nx,ny = solver.R.shape
#     psi0, Gloc, Qloc = basis

    
#     _, vjp_params = jax.vjp(lambda c,p : solver.F_function(psi_guess, c, p), tokamak_psi, profilePars)

#     res1 = vjp_params(u)
    
#     return jnp.zeros_like(psi_guess), res1[0], res1[1]

# nksolve.defvjp(nksolve_fwd, nksolve_bwd)

@jax.jit
def Greens(Rc, Zc, R, Z):
    """
    Calculate poloidal flux at (R,Z) due to a unit current
    at (Rc,Zc) using Greens function

    """

    # Calculate k^2
    k2 = 4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)

    # Clip to between 0 and 1 to avoid nans e.g. when coil is on grid point
    k2 = jnp.clip(k2, 2e-7, 1.0 - 2e-7)
    k = jnp.sqrt(k2)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)
    return (
        (mu0 / (2.0 * jnp.pi))
        * jnp.sqrt(R * Rc)
        * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))
        / k
    )

@jax.jit
def ellipk(m):
    A=jnp.array([1.37982864606273237150E-4,
                2.28025724005875567385E-3,
                7.97404013220415179367E-3,
                9.85821379021226008714E-3,
                6.87489687449949877925E-3,
                6.18901033637687613229E-3,
                8.79078273952743772254E-3,
                1.49380448916805252718E-2,
                3.08851465246711995998E-2,
                9.65735902811690126535E-2,
                1.38629436111989062502E0
                ])
    B=jnp.array([2.94078955048598507511E-5,
                9.14184723865917226571E-4,
                5.94058303753167793257E-3,
                1.54850516649762399335E-2,
                2.39089602715924892727E-2,
                3.01204715227604046988E-2,
                3.73774314173823228969E-2,
                4.88280347570998239232E-2,
                7.03124996963957469739E-2,
                1.24999999999870820058E-1,
                4.99999999999999999821E-1
                ])
    return jnp.polyval(A,1-m) - jnp.log(1-m)*jnp.polyval(B,1-m)

@jax.jit
def ellipe(m):
    A=jnp.array([1.53552577301013293365E-4,
                2.50888492163602060990E-3,
                8.68786816565889628429E-3,
                1.07350949056076193403E-2,
                7.77395492516787092951E-3,
                7.58395289413514708519E-3,
                1.15688436810574127319E-2,
                2.18317996015557253103E-2,
                5.68051945617860553470E-2,
                4.43147180560990850618E-1,
                1.00000000000000000299E0
                ])
    B=jnp.array([3.27954898576485872656E-5,
                1.00962792679356715133E-3,
                6.50609489976927491433E-3,
                1.68862163993311317300E-2,
                2.61769742454493659583E-2,
                3.34833904888224918614E-2,
                4.27180926518931511717E-2,
                5.85936634471101055642E-2,
                9.37499997197644278445E-2,
                2.49999999999888314361E-1
                ])

    return jnp.polyval(A,1-m) - jnp.log(1-m)*((1-m)*jnp.polyval(B,1-m))

ellipk = jax.custom_jvp(ellipk)

@ellipk.defjvp
@jax.jit
def _ellipk_jvp(primals, tangents):
    m, = primals
    m_dot, = tangents
    dKdk = m_dot*((ellipe(m)/((2*m)*(1-m))) - (ellipk(m)/(2*m)))
    return ellipk(m), dKdk

ellipe = jax.custom_jvp(ellipe)

@ellipe.defjvp
@jax.jit
def _ellipe_jvp(primals, tangents):
    m, = primals
    m_dot, = tangents
    dEdk = m_dot*(ellipe(m)-ellipk(m))/(2.0*m)
    return ellipe(m), dEdk