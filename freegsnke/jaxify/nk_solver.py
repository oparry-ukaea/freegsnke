import jax.numpy as np

"""Implementation of Newton Krylow algorithm for solving
a generic root problem of the type
F(x, other args) = 0
in the variable x. 
Problem must be formulated so that x is a 1d np.array.

In practice, given a guess x_0 and F(x_0) = R_0
it aims to find the best step dx such that 
F(x_0 + dx) is minimum.
"""       

def Arnoldi_iteration(x0, #trial_current expansion point
							dx, #first vector for current basis
							R0, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
							F_function,
							step_size,
							scaling_with_n,
							target_relative_unexplained_residual,  
							max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
							clip,
							):
	
	"""Performs the iteration of the NK solution method:
	1) explores direction dx
	2) computes and stores new residual
	3) builds new candidate direction to continue exploring
	Calculates best candidate step, stored at self.dx

	Parameters
	----------
	x0 : 1d np.array, np.shape(x0) = self.problem_dimension
		The expansion point x_0
	dx : 1d np.array, np.shape(dx) = self.problem_dimension
		The first direction to be explored. 
	R0 : 1d np.array, np.shape(R0) = self.problem_dimension
		Residual at expansion point x_0
	F_function : 1d np.array, np.shape(x0) = self.problem_dimension
		Function representing the root problem at hand
	args : list 
		Additional arguments for using function F
		F(x_0 + dx, *args)
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
	problem_dimension = len(x0)
	nR0 = np.linalg.norm(R0)

	#basis in x space
	Q = np.zeros((problem_dimension, max_n_directions+1))
	#orthonormal basis in x space
	Qn = np.zeros((problem_dimension, max_n_directions+1))
	#basis in residual space
	G = np.zeros((problem_dimension, max_n_directions+1))
	#orthonormal basis in residual space
	Gn = np.zeros((problem_dimension, max_n_directions+1))
	
	n_it = 0
	n_it_tot = 0
	adjusted_step_size = step_size*nR0

	explore = 1
	while explore:
		this_step_size = adjusted_step_size*((1 + n_it)**scaling_with_n)
		candidate_step = this_step_size*dx/np.linalg.norm(dx)
		candidate_x = x0 + candidate_step
		R_dx = F_function(candidate_x)
		useful_residual = R_dx - R0

		Q=Q.at[:, n_it].set(candidate_step)
		Qn=Qn.at[:, n_it].set(candidate_step / np.linalg.norm(candidate_step))
		
		G=G.at[:, n_it].set(useful_residual)
		Gn=Gn.at[:, n_it].set(useful_residual / np.linalg.norm(useful_residual))

		#orthogonalize with respect to previously attemped directions 
		useful_residual -= np.sum(np.sum(Qn[:,:]*useful_residual[:,np.newaxis], axis=0, keepdims=True)*Qn[:,:], axis=1)
		dx = useful_residual

		n_it += 1
		Gloc = G[:,:n_it]
		Qloc = Q[:,:n_it]
		coeffs = np.matmul( np.matmul( np.linalg.inv( np.matmul(Gloc.T, Gloc)), Gloc.T), -R0)                            
		coeffs = np.clip(coeffs, -clip, clip)
		explained_residual = np.sum(Gloc*coeffs[np.newaxis,:], axis=1) 
		relative_unexplained_residual = np.linalg.norm(explained_residual + R0)/nR0
		explained_residual_check = (relative_unexplained_residual > target_relative_unexplained_residual)
		explore = explained_residual_check
		explore *= (n_it < max_n_directions)

	best_dx = np.sum(Qloc*coeffs[np.newaxis,:], axis=1)

	return (best_dx, (x0, Gloc, Qloc))