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
from . import full_mask

# Physical constants
mu0 = 4e-7 * jnp.pi

class AbstractLinearSolver(eqx.Module):
    A: jax.Array

    def __init__(self, A):
        self.A = A


class SparseLinearSolver(AbstractLinearSolver):

    def __init__(self, A):
        super().__init__(A)
        
    def __call__(self, rhs):
        return jexsp.linalg.spsolve(self.A.data,
                                    self.A.indices,
                                    self.A.indptr,
                                    rhs)

class DenseLinearSolver(AbstractLinearSolver):

    def __init__(self, A):
        super().__init__(A)

    def __call__(self, rhs):
        return jnp.dot(self.A, rhs)

class NKGSsolver(eqx.Module):

    """Solver for the non-linear forward Grad Shafranov (GS) 
    static problem. Here, the GS problem is written as a root
    problem in the plasma flux psi. This root problem is 
    solved by a Newton method or a Newton-Krylov method.

    The solution domain is set at instantiation time, through the 
    input freeGS equilibrium object.

    The non-linear solver itself is called using the 'solve' method.
    """
    R: jax.Array
    Z: jax.Array
    dRdZ: float
    coil_green: jax.Array
    bndry_indices: jax.Array
    bndry_green: jax.Array
    profile: eqx.Module
    limiter: eqx.Module
    linear_GS_solver: eqx.Module
     
    def __init__(self, eq, profile, limiter_func, use_sparse_solver=False, precompute_boundary_greens=True):

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
        profile: a jaxify-ed profile object as defined in jaxify.jtor that
                defines the toroidal current profile parametrization
        limiter_func: a jaxify-ed limiter object as defined in 
                     jaxify.limiter_func that defines the limtier domain
        use_sparse_solver: Boolean to define whether a sparse linear solver
                          is used. If False, dense matrix inverse is 
                          calculated, stored, and used in every iteration.
        precompute_boundary_greens: Boolean to define whether the Green's
                        functions for the boundary condition is
                        calculated and stored or computed on-the-fly.
                        If True, self.greenfunc is stored and used.        
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
        
        ncoils = eq._vgreen.shape[0]
        self.coil_green=jnp.array(eq._vgreen.reshape((ncoils,nx*ny)))

        #linear solver for del*Psi=RHS
        generator=freegs4e.gradshafranov.GSsparse4thOrder(eq.R[0,0],eq.R[-1,0],eq.Z[0,0],eq.Z[0,-1])
        
        if (use_sparse_solver):
            A = jexsp.BCSR.from_scipy_sparse(generator(nx,ny))
            self.linear_GS_solver = SparseLinearSolver(A)
        else:
            A = jnp.linalg.inv(jexsp.BCSR.from_scipy_sparse(generator(nx,ny)).todense())
            self.linear_GS_solver = DenseLinearSolver(A)
        
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

        if (precompute_boundary_greens):
            # matrices of responses of boundary locations to each grid positions
            greenfunc = Greens(R[jnp.newaxis,:,:], 
                            Z[jnp.newaxis,:,:], 
                            R_1D[self.bndry_indices[:,0]][:,jnp.newaxis,jnp.newaxis], 
                            Z_1D[self.bndry_indices[:,1]][:,jnp.newaxis,jnp.newaxis])
            # Prevent infinity/nan by removing Greens(x,y;x,y) 
            zeros = jnp.ones_like(greenfunc)
            zeros=zeros.at[jnp.arange(len(self.bndry_indices)), self.bndry_indices[:,0], self.bndry_indices[:,1]].set(0.0)
            self.bndry_green = greenfunc*zeros*self.dRdZ
        else:
            self.bndry_green = None
        
        self.profile = profile
        self.limiter = limiter_func

        # Test run
        init_params=profile.init_params
        ppsi=eq.plasma_psi.reshape(-1)
        tpsi=eq.tokamak.getPsitokamak(vgreen=eq._vgreen).reshape(-1)
        self.F_function(ppsi,tpsi,init_params)
        self.F_function2(ppsi,tpsi,init_params)
        # zeromach = jnp.asarray(jnp.pi)
		# while (1.0+zeromach/2.0 > 1.0):
		# 	zeromach = zeromach/2.0

		# self.minprec = 2.*zeromach

    @jax.jit
    def gs_oper(self, psi):
        """
        Apply the full elliptic operator to 2D field L*psi_plasma

        Arguments:
        ----------
        psi: jax.Array of size (nx,ny)

        Returns:
        --------
        b: jax.Array of size (nx,ny)
           d2(psi)dR2 + d2(psi)/dZ2 + 1.0/R*d(psi)/dR
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
        """
        Calculate the critical (O-,X-) points 

        Arguments:
        ----------
        psi: jax.Array of size (nx,ny)
             Total Psi (plasma + metal)
        Returns:
        --------
        oo: jax.Array of size (critsize,3)
            O-Points in an array with static size
            Each row has (R_o, Z_o, Psi_o)
        xx: jax.Array of size (critsize,3)
            O-Points in an array with static size
            Each row has (R_x, Z_x, Psi_x)

        critsize is needed to ensure that the output
        of the function is always the same size so that
        JAX can trace the function. This means that not 
        all the values in the array are physical, and
        we only use the top few as needed.
        """

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

        ir, iz = jnp.nonzero(A2,size=200)

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
        """
        Calculate the physical plasma domain 

        Arguments:
        ----------
        psi: jax.Array of size (nx,ny)
            Total Psi (plasma + metal)
        oo: jax.Array of size (critsize,3)
            O-Points in an array with static size
            Each row has (R_o, Z_o, Psi_o)

        xx: jax.Array of size (critsize,3)
            O-Points in an array with static size
            Each row has (R_x, Z_x, Psi_x)

        Returns:
        --------
        mask: jax.Array of size (nx,ny)
            Integer array defining 1 in plasma domain
            and 0 outside plasma domain.
        """

        # mask=jnp.zeros(psi.shape)

        # Ro, Zo, psio = opoint[0]
        # Rx, Zx, psix = xpoint[0]

        # # Normalise psi
        # psin = (psi - psio) / (psix - psio)

        # # Condition that 0 < psin < 1
        # s1=(psin>=0)*(psin<=1)

        # # Condition that (R-Rx)*(R0-Rx) + (Z-Zx)*(Z0-ZX) > 0
        # m2=(self.R-Rx)*(Ro-Rx) + (self.Z-Zx)*(Zo-Zx)
        # s2=(m2>0)

        # m3=(self.R-xpoint[1,0])*(Ro-xpoint[1,0]) + (self.Z-xpoint[1,1])*(Zo-xpoint[1,1])
        # s3=(m3>0)

        # mask=s1*s2*s3

        mask = full_mask.inside_mask(self.R, self.Z,
                                   psi, opoint, xpoint)

        return mask

    @jax.jit
    def jtor(self, profilePars, psi, check_limited=True):
        """
        Calculate the toroidal plasma current 

        Arguments:
        ----------
        psi: jax.Array of size (nx,ny)
            Total Psi (plasma + metal)
        profilePars: Tuple containing (Ip, (plasma_pars)) 
            defining toroidal current profile

        Returns:
        --------
        jtor: jax.Array of size (nx,ny)
            Toroidal plasma current density
        """

        opts, xpts = self.critpoints(psi)
        diverted_mask = self.mask(psi, opts, xpts)
        psib = xpts[0,2]

        if (check_limited):
            dmask_inside_limiter = diverted_mask*self.limiter.mask_inside_limiter 
            psi_bound, limiter_mask = self.limiter.core_mask_limiter(
                                        self.R, self.Z, 
                                        psi, psib, 
                                        dmask_inside_limiter, 
                                        self.limiter.limiter_mask_out)
            lmask_sum = jnp.sum(limiter_mask * self.limiter.mask_inside_limiter)

            # Quantities to calculate Jtor inside plasma core
            plasma_domain_mask = jnp.where(lmask_sum==0,
                                    dmask_inside_limiter,
                                    limiter_mask)
            psi_bndry = jnp.where(lmask_sum==0,psib,psi_bound)
        else:
            plasma_domain_mask = diverted_mask
            psi_bndry = psib

        psi_axis = opts[0,2]
        jtor = self.profile.jtor(self, profilePars, psi, 
                    psi_axis, psi_bndry, plasma_domain_mask)

        return jtor

    @jax.jit
    def freeboundary(self, profilePars, psi):
        """
        Calculate the RHS of GS problem and impose boundary
        conditions. 

        Arguments
        ----------
        psi : jax.Array of size (nx,ny)
            Total psi (metal + plasma)
        profilePars: Tuple containing (Ip, (plasma_pars)) 
            defining toroidal current profile

        Returns
        -------
        rhs: jax.Array of size (nx,ny)
        """
      
        #jtor and RHS given tokamak_psi above and the input plasma_psi
        nx,ny=self.R.shape
        ppsi = psi.reshape(nx,ny)
        jtor = self.jtor(profilePars, ppsi)    
        rhs = -mu0*self.R*jtor
        zeroprec = self.R[0,0]-self.R[0,0]

        def _psibound(x,y):
            greenfunc = Greens(self.R, self.Z, self.R[x, y], self.Z[x, y])
            # Prevent infinity/nan by removing (x,y) point
            greenfunc = greenfunc.at[x, y].set(0.0)
            # Integrate over the domain
            psival = jnp.sum(jnp.sum(greenfunc * jtor))

            return psival

        _psibound_vmap=jax.vmap(_psibound,in_axes=(0,0))
        
        #calculates and assignes boundary conditions
        psi_boundary = jnp.zeros_like(self.R)
        if self.bndry_green is None:
            # calculate greenfunctions on-the-fly using vmapped function
            xb, yb = self.bndry_indices[:,0], self.bndry_indices[:,1]
            psi_bnd = _psibound_vmap(xb, yb)
            psi_boundary = psi_boundary.at[xb,yb].set(psi_bnd*self.dRdZ)
        else:
            # weighted sum over the last two axes.
            # "contract" axis 1 of greenfunc with axis 0 of jtor
            # contract axis 2 of greenfunc with axis 1 of jtor
            psi_bnd = jnp.tensordot(self.bndry_green, jtor, axes=([1, 2], [0, 1]))

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
        """
        Nonlinear Grad Shafranov equation written as a root problem
        F(plasma_psi) \equiv \delta* (plasma_psi) - J(plasma_psi + tokamak_psi)

        The plasma_psi that solves the Grad Shafranov problem satisfies
        F(plasma_psi) = 0

        Arguments
        ----------
        plasma_psi : 1-D jax.Array of size (nx*ny)
            Magnetic flux contribution from plasma
        tokamak_psi : 1-D jax.Array of size (nx*ny)
            Magnetic flux contribution from metal objects
            (coils + passive structures)
        profilePars: Tuple containing (Ip, (plasma_pars)) 
            defining toroidal current profile

        Returns
        -------
        resid: 1-D jax.Array of size (nx*ny)
            Residual of GS problem 
        """

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
        F(plasma_psi) \equiv plasma_psi - \delta*^{-1}( J(plasma_psi + tokamak_psi))

        The plasma_psi that solves the Grad Shafranov problem satisfies
        F(plasma_psi) = 0

        
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
        residual = plasma_psi - self.linear_GS_solver(rhs.reshape(-1))

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
        max_solving_iterations=100,
        Picard_handover=0.11,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.2,  
        max_n_directions=16,
        clip=10,
        verbose=False,
        max_rel_update_size=0.2,
    ):
        
        """The method that actually solves the GS problem.
        The problem is specified by the profile parameters and currents.       

        Arguments:
        ----------
        init_psi : jax.Array of size (nx,ny)
            Initial guess for the nonlinear solver
        profilePars: Tuple of Profile Parameters
        currentvec: jax.Array of 1-D currents vector 
        target_relative_tolerance : float
            NK/NR iterations are interrupted when this criterion is 
            satisfied. Relative convergence
        use_newton: boolean 
            Choose whether Newton-Krylov or Newton-Raphson method is used
        lag_Jacobian: int
            How often to recalculate the Jacobian in the NR method
            1 means every iteration, 2 means every 2 iterations, etc
        max_solving_iterations : int
            NK iterations are interrupted when this limit is surpassed
        Picard_handover : float
            Value of relative tolerance above which a Picard iteration
            is performed instead of a full NK call
        step_size : float
            l2 norm of proposed step, in units of the size of the residual R0
        scaling_with_n : float
            allows to further scale the proposed steps as a function of the
            number of previous steps already attempted
            (1 + n_it)**scaling_with_n
        target_relative_unexplained_residual : float between 0 and 1
            terminates internal iterations when the considered directions
            can (linearly) explain such a fraction of the initial residual R0
        max_n_directions : int
            terminates iteration even though condition on
            explained residual is not met
        max_rel_update_size : float
            maximum relative update, in norm, to plasma_psi. If larger than this,
            the norm of the update is reduced
        clip : float
            maximum size of the update due to each explored direction, in units
            of exploratory step used to calculate the finite difference derivative
        verbose : bool
            flag to allow progress printouts
        """
        
        nx,ny=init_psi.shape
        tokamak_psi = jnp.dot(currentvec,self.coil_green)
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
            plasma_psi, _ = _nsolve(self,
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
            plasma_psi, _ = _nksolve(self,
                            solver_params,
                            trial_plasma_psi,
                            tokamak_psi,
                            profilePars,
                            )
    
        # return new solution
        return (plasma_psi+tokamak_psi).reshape(nx,ny)

@partial(jax.custom_jvp, nondiff_argnums=(0,1))
def _nksolve(solver,
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

    log.append("Initial relative error ="+str(rel_change))
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
            if picard_flag < min(max_solving_iterations - 1, 3):
                    # make picard update to the flux up-down symmetric
                    # this combats the instability of picard iterations
                    res0_2d = res0.reshape(nx, ny)
                    res0 = 0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
                    picard_flag += 1
            else:
                    # update = -1.0 * res0
                    picard_flag = 1
            update = -1.0 * res0
            Abasis = (trial_plasma_psi, None, None)
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
            log.append("Update too large, resized.")
        
        update0 = update
        check_resid = True
        while (check_resid):
            new_trial_plasma_psi = trial_plasma_psi + update
            new_res0 = solver.F_function(new_trial_plasma_psi, tokamak_psi, profilePars)
            new_norm_rel_change = solver.relative_norm_residual(
                        new_res0, new_trial_plasma_psi
                    )
            nan_resid = jnp.isnan(new_norm_rel_change)
            norm_increase = (new_norm_rel_change > 1.2 * history_norm_rel_change[-1])
            check_resid = jnp.logical_or(nan_resid,norm_increase)
            if (check_resid):
                log.append(
                        "Update resizing triggered due to residual increase or NaN..."
                    )
                update = update*0.75

        trial_plasma_psi = trial_plasma_psi + update
        res0 = solver.F_function(trial_plasma_psi, tokamak_psi, profilePars)
        norm_rel_change = solver.relative_norm_residual(res0, trial_plasma_psi)
        rel_change, del_psi = solver.relative_del_residual(res0, trial_plasma_psi)
        starting_direction = res0
        relative_change = 1.0 * rel_change
        history_norm_rel_change.append(norm_rel_change)
        log.append("...relative error ="+str(rel_change))
        log.append("-----")
        if verbose:
            for x in log:
                print(x)

        log = []
        iter +=1

    return (trial_plasma_psi, Abasis)

@_nksolve.defjvp
def _nksolve_jvp(solver, solver_params, primals, tangents):

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

    opsi, Abasis = _nksolve(solver, solver_params, trial_plasma_psi, tokamak_psi, profilePars)
    psi0, Gloc, Qloc = Abasis

    def Ffunc(x):
        return solver.F_function(x, tokamak_psi, profilePars)

    def Floc(t, p):
        return solver.F_function(psi0, t ,p)

    def dFfunc(dx):
        r, dr = jax.jvp(Ffunc, (psi0,), (dx,))
        return dr

    def solve_with_gmres(A,b):
        return jax.scipy.sparse.linalg.gmres(A,b,x0=b,restart=10,solve_method='incremental',atol=1e-9)[0]

    res0, jvp_res0 = jax.jvp(Floc,(tokamak_psi, profilePars), (dtpsi, dprofile,))
    tangent_out = jax.lax.custom_linear_solve(dFfunc, -jvp_res0, solve=solve_with_gmres, transpose_solve=solve_with_gmres)
    
    primal_out = (opsi, Abasis)

    return (primal_out, (tangent_out,(jnp.zeros_like(opsi), jnp.zeros_like(Gloc), jnp.zeros_like(Qloc))) )

@partial(jax.custom_jvp,nondiff_argnums=(0,1))
def _nsolve(solver,
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
            log.append("...Assembling Jacobian...")
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

@_nsolve.defjvp
def _nsolve_jvp(solver, solver_params, primals, tangents):

    trial_plasma_psi, tokamak_psi, profilePars, = primals
    dppsi, dtpsi, dprofile, = tangents

    opsi, basis = _nsolve(solver, solver_params, trial_plasma_psi, tokamak_psi, profilePars)
    psi0, Jmat = basis

    def Ffunc(x):
        return solver.F_function2(x, tokamak_psi, profilePars)

    def Floc(t, p):
        return solver.F_function2(psi0, t ,p)

    res0, jvp_res0 = jax.jvp(Floc,(tokamak_psi, profilePars), (dtpsi, dprofile,))
    primal_out = (opsi, basis)
    tangent_out = jnp.linalg.solve(Jmat, -jvp_res0)

    return (primal_out, (tangent_out, (jnp.zeros_like(psi0), jnp.zeros_like(Jmat))))

@jax.jit
def Greens(Rc, Zc, R, Z):
    """
    Calculate poloidal flux at (R,Z) due to a unit current
    at (Rc,Zc) using Greens function

    """

    # Calculate k^2
    k2 = 4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)

    # Clip to between 0 and 1 to avoid nans e.g. when coil is on grid point
    k2 = jnp.clip(k2, 1e-10, 1.0 - 1e-10)
    k = jnp.sqrt(k2)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)
    return (
        (mu0 / (2.0 * jnp.pi))
        * jnp.sqrt(R * Rc)
        * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))
        / k
    )

# Elliptical functions
# Polynomial expressions taken from
# Methods and Programs for Mathematical Functions
# by Stephen L. Moshier, E. Horwood, 1989
# https://www.moshier.net/methprog.pdf
# pages 387 and 392

@jax.custom_jvp
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

@ellipk.defjvp
def _ellipk_jvp(primals, tangents):
    m, = primals
    m_dot, = tangents
    dKdk = m_dot*((ellipe(m)/((2*m)*(1-m))) - (ellipk(m)/(2*m)))
    return ellipk(m), dKdk

@jax.custom_jvp
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

@ellipe.defjvp
def _ellipe_jvp(primals, tangents):
    m, = primals
    m_dot, = tangents
    dEdk = m_dot*(ellipe(m)-ellipk(m))/(2.0*m)
    return ellipe(m), dEdk