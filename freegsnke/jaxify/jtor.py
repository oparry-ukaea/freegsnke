import numpy as np
import jax.numpy as jnp
import jax
import jax.scipy as jsp
import equinox as eqx

# Physical constants
mu0 = 4e-7 * jnp.pi

class JConstrainPaxisIp(eqx.Module):

    init_params: jax.Array

    def __init__(self, paxis, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):

        self.init_params = (jnp.array(Ip),jnp.array([alpha_m, alpha_n, paxis, Raxis]))

    @jax.jit    
    def jtor(self, solver, profilePars, psi, psia, psib, plasmadomain):

        # Extract profile information
        Ip=profilePars[0]
        alpha_m=profilePars[1][0]
        alpha_n=profilePars[1][1]
        paxis=profilePars[1][2]
        Raxis=profilePars[1][3]

        # Normalised psi
        psi_norm = (psi - psia) / (psib - psia)

        # Current profile shape
        jtorshape = (1.0 - jnp.clip(psi_norm, 0.0, 1.0) ** alpha_m) ** alpha_n
        jtorshape = jtorshape * plasmadomain

        # Now apply constraints to define constants

        # Need integral of jtorshape to calculate paxis
        # Note factor to convert from normalised psi integral
        a, b = 1.0/alpha_m , 1.0+alpha_n
        shapefac = jsp.special.gamma(a)*jsp.special.gamma(b)/(alpha_m*jsp.special.gamma(a+b))
        shapeintegral = shapefac * (psib - psia)

        # Pressure on axis is
        # paxis = - (L*Beta0/Raxis) * shapeintegral

        # Integrate current components
        IR = jnp.sum(jnp.sum(jtorshape * solver.R / Raxis)) * solver.dRdZ
        I_R = jnp.sum(jnp.sum(jtorshape * Raxis / solver.R)) * solver.dRdZ

        # Toroidal plasma current Ip is
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        LBeta0 = -paxis * Raxis / shapeintegral

        L = Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        # Toroidal current
        Jtor = L * (Beta0 * solver.R / Raxis 
                + (1 - Beta0) * Raxis / solver.R) * jtorshape

        return Jtor

class JFiesta_Topeol(eqx.Module):

    init_params: jax.Array

    def __init__(self, Beta0, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):

        self.init_params = (jnp.array(Ip),jnp.array([alpha_m, alpha_n, Beta0, Raxis]))

    @jax.jit    
    def jtor(self, solver, profilePars, psi, psia, psib, plasmadomain):

        # Extract profile information
        Ip=profilePars[0]
        alpha_m=profilePars[1][0]
        alpha_n=profilePars[1][1]
        Beta0=profilePars[1][2]
        Raxis=profilePars[1][3]

        # Normalised psi
        psi_norm = (psi - psia) / (psib - psia)

        # Current profile shape
        jtorshape = (1.0 - jnp.clip(psi_norm, 0.0, 1.0) ** alpha_m) ** alpha_n
        jtorshape = jtorshape * plasmadomain

        # Toroidal current
        Jtor = (Beta0 * solver.R / Raxis 
                + (1 - Beta0) * Raxis / solver.R) * jtorshape
        L = Ip / jnp.maximum(jnp.sum(Jtor)*solver.dRdZ, 1e-9)

        return L*Jtor

class JLao85(eqx.Module):

    init_params: jax.Array
    Ip_logic: bool

    def __init__(self,
        Ip,
        fvac,
        alpha,
        beta,
        alpha_logic=True,
        beta_logic=True,
        Ip_logic=True,
    ):

        par_alpha = jnp.array(np.array(alpha))
        par_beta = jnp.array(np.array(beta))

        if alpha_logic:
            par_alpha = jnp.concatenate((par_alpha,jnp.array([-jnp.sum(par_alpha)])))
        if beta_logic:
            par_beta = jnp.concatenate((par_beta,jnp.array([-jnp.sum(par_beta)])))

        self.init_params = (jnp.array(Ip), par_alpha, par_beta)
        self.Ip_logic = Ip_logic

    @jax.jit
    def jtor(self, solver, profilePars, psi, psia, psib, plasmadomain):

        # Extract profile information
        (Ip, alpha, beta) = profilePars
        alpha_exp = jnp.arange(0,len(alpha))
        beta_exp = jnp.arange(0,len(beta))

        # Normalised psi
        psi_norm = (psi - psia) / (psib - psia)
        psi_norm = jnp.clip(psi_norm,0.0,1.0)

        # calculate the p' and FF' profiles
        pprime_term = (
            psi_norm[jnp.newaxis, :, :]
            ** alpha_exp[:, jnp.newaxis, jnp.newaxis]
        )
        pprime_term *= alpha[:, jnp.newaxis, jnp.newaxis]
        pprime_term = jnp.sum(pprime_term, axis=0)
        pprime_term *= solver.R

        ffprime_term = (
            psi_norm[jnp.newaxis, :, :]
            ** beta_exp[:, jnp.newaxis, jnp.newaxis]
        )
        ffprime_term *= beta[:, jnp.newaxis, jnp.newaxis]
        ffprime_term = jnp.sum(ffprime_term, axis=0)
        ffprime_term /= solver.R
        ffprime_term /= mu0

        # sum together
        Jtor = pprime_term + ffprime_term

        # put to zero all current outside the LCFS
        Jtor *= psi > psib

        Jtor *= Ip * Jtor > 0

        Jtor *= plasmadomain

        # if Ip normalisation is required, do it
        jtorIp = jnp.sum(Jtor)
        L = jnp.where(self.Ip_logic,Ip/(jtorIp*solver.dRdZ),1.0)
        Jtor = L * Jtor

        return Jtor
    
class JPprimeFFprime(eqx.Module):

    init_params: jax.Array
    Ip_logic: bool

    def __init__(self, Ip, pprime_data, ffprime_data, Ip_logic=True):

        npoints = pprime_data.shape[0]
        psin = jnp.linspace(0,1,npoints)
        self.init_params = (jnp.array(Ip),(psin, jnp.array(pprime_data), 
                                                 jnp.array(ffprime_data)))
        
        self.Ip_logic = Ip_logic

    @jax.jit    
    def jtor(self, solver, profilePars, psi, psia, psib, plasmadomain):

        # Extract profile information
        Ip=profilePars[0]
        psin=profilePars[1][0]
        pprime=profilePars[1][1]
        ffprime=profilePars[1][2]

        # Normalised psi
        psi_norm = (psi - psia) / (psib - psia)

        # calculate normalised psi
        psi_norm = jnp.clip(psi_norm, 0.0, 1.0)

        # calculate the p' and FF' profiles
        pprime_term = jnp.interp(psi_norm, psin, pprime)

        ffprime_term = jnp.interp(psi_norm, psin, ffprime)

        # sum together
        Jtor = solver.R*pprime_term + (1.0/solver.R/mu0)*ffprime_term

        # put to zero all current outside the LCFS
        Jtor *= plasmadomain

        # if Ip normalisation is required, do it
        jtorIp = jnp.sum(Jtor)
        L = jnp.where(self.Ip_logic,Ip/(jtorIp*solver.dRdZ),1.0)
        Jtor = L * Jtor
        
        return Jtor