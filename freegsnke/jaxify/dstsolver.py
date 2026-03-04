import jax.numpy as jnp
import equinox as eqx
import jax

import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp
from typing import Callable


# ---------------- built-in tridiagonal_solve ----------------
@jax.jit
def tri_default(dl, diag, du, rhs):
	"""
	Tridiagonal solver using jax.lax.linalg.tridiagonal_solve.
	Shapes:
	  - rhs: (B, N) or (N,)
	  - dl, diag, du: same leading shape as rhs.
	Requires:
	  dl[..., 0] == 0 and du[..., -1] == 0
	"""

	x = jax.lax.linalg.tridiagonal_solve(dl, diag, du, rhs[..., None])
	return jnp.squeeze(x, axis=-1)


@jax.jit
def thomas_solve_fused_batched(dl, diag, du, rhs):
	"""
	Batched differentiable Thomas solver with fused memory pattern (corrected).

	Parameters
	----------
	dl   : (B, N)  lower diagonal; dl[:, 0] must be 0
	diag : (B, N)  main diagonal
	du   : (B, N)  upper diagonal; du[:, -1] must be 0
	rhs  : (B, N)  right-hand side

	Returns
	-------
	x    : (B, N)  solution to A x = rhs for each batch row

	Notes
	-----
	Forward elimination (for i = 1..N-1):
		w       = dl[i] / diag[i-1]
		diag[i] = diag[i] - w * du[i-1]
		rhs[i]  = rhs[i]  - w * rhs[i-1]

	Back substitution:
		x[N-1]  = rhs2[N-1] / diag2[N-1]
		x[i]    = (rhs2[i] - du[i]*x[i+1]) / diag2[i]
	"""
	B, N = diag.shape

	def _n1_case():
		# N == 1
		return rhs / diag

	def _general_case():
		# Seed (i=0)
		diag0 = diag[:, 0]          # (B,)
		rhs0  = rhs[:,  0]          # (B,)

		# Prepare sequences for i=1..N-1
		dl_i    = dl[:, 1:]         # (B, N-1)   dl[i]
		du_im1  = du[:, :-1]        # (B, N-1)   du[i-1]
		diag_i  = diag[:, 1:]       # (B, N-1)   diag[i]
		rhs_i   = rhs[:,  1:]       # (B, N-1)   rhs[i]

		def fwd_step(carry, inp):
			diag_prev, rhs_prev = carry                 # (B,), (B,)
			dl_i_t, du_im1_t, diag_i_t, rhs_i_t = inp   # (B,)*4

			w       = dl_i_t / diag_prev               # (B,)
			diag_i2 = diag_i_t - w * du_im1_t
			rhs_i2  = rhs_i_t  - w * rhs_prev

			return (diag_i2, rhs_i2), (diag_i2, rhs_i2)

		(diag_rhs_last, diag_rhs_emits) = jax.lax.scan(
			fwd_step,
			(diag0, rhs0),
			(dl_i.T, du_im1.T, diag_i.T, rhs_i.T)       # leading axis = time (N-1,)
		)
		diag2_tail_T, rhs2_tail_T = diag_rhs_emits      # each (N-1, B)

		# Reconstruct full arrays (B, N)
		diag2 = jnp.concatenate([diag0[:, None], diag2_tail_T.T], axis=1)
		rhs2  = jnp.concatenate([rhs0[:,  None], rhs2_tail_T.T ], axis=1)

		# Back substitution
		x_last = rhs2[:, -1] / diag2[:, -1]             # (B,)

		du_i      = du[:, :-1]                          # (B, N-1)    du[i] for i=0..N-2
		rhs2_head = rhs2[:, :-1]
		diag2_head= diag2[:, :-1]

		# Traverse i = N-2 .. 0  (reverse)
		def bwd_step(x_next, inp):
			du_i_t, rhs2_i_t, diag2_i_t = inp
			x_i = (rhs2_i_t - du_i_t * x_next) / diag2_i_t
			return x_i, x_i  # carry x_i, emit x_i

		x_head_rev_last, x_head_rev = jax.lax.scan(
			bwd_step,
			x_last,
			(du_i[:, ::-1].T, rhs2_head[:, ::-1].T, diag2_head[:, ::-1].T)
		)
		# x_head_rev: shape (N-1, B) with [x[N-2], x[N-3], ..., x[0]]
		x_head = x_head_rev[::-1].T  # (B, N-1) → [x[0], x[1], ..., x[N-2]]

		x = jnp.concatenate([x_head, x_last[:, None]], axis=1)  # (B, N)
		return x

	return jax.lax.cond(N == 1, _n1_case, _general_case)


@jax.jit
def thomas_solve_fused(dl, diag, du, rhs):
	"""
	Convenience wrapper that accepts 1D or 2D and returns same shape.
	"""
	# Promote to (B, N)
	if rhs.ndim == 1:
		dl2   = dl[None, :]
		diag2 = diag[None, :]
		du2   = du[None, :]
		rhs2  = rhs[None, :]
		x     = thomas_solve_fused_batched(dl2, diag2, du2, rhs2)
		return x[0]
	else:
		return thomas_solve_fused_batched(dl, diag, du, rhs)


@jax.jit
def thomas_solve_fused_multi_rhs(dl, diag, du, rhs):
	"""
	Multi-RHS version.

	Parameters
	----------
	dl, diag, du : (B, N)
	rhs          : (B, N, K)

	Returns
	-------
	x            : (B, N, K)
	"""
	# vmap over the last dimension of rhs (K RHS per system)
	solve_one_rhs = jax.vmap(
		lambda r: thomas_solve_fused_batched(dl, diag, du, r),
		in_axes=2,
		out_axes=2
	)
	return solve_one_rhs(rhs)

# DST method
class DSTSolver(eqx.Module):

	Rmin: float
	Rmax: float
	Zmin: float
	Zmax: float
	R: jax.Array
	Z: jax.Array
	dR: float
	dZ: float
	dl_batch: jax.Array
	du_batch: jax.Array
	diag_batch: jax.Array
	_solve_tridiag: Callable = eqx.field(static=True)

	def __init__(self,R,Z):
		self.Rmin = R[0,0]
		self.Rmax = R[-1,0]
		self.Zmin = Z[0,0]
		self.Zmax = Z[0,-1]
		self.R = R
		self.Z = Z

		self.dR = R[1,0]-R[0,0]
		self.dZ = Z[0,1]-Z[0,0]

		self.init_matrix()

	# -------------------------------
	# DST-I implementation (orthonormal, self-inverse per your check)
	# @jax.jit
	# def dstI1D(self, x, norm="ortho"):
	# 	"""1D type-I discrete sine transform along the last axis."""
	# 	num_dims = x.ndim
	# 	N = x.shape
	# 	padding = ((0, 0),) * (num_dims - 1) + ((1, 1),)
	# 	x = jnp.pad(x, pad_width=padding, mode="constant", constant_values=0.0)
	# 	x = jnp.fft.irfft(-1j * x, axis=-1, norm=norm)
	# 	x = jax.lax.slice_in_dim(x, 1, N[-1] + 1, axis=-1)
	# 	return x

	@jax.jit
	def dstI1D(self, x):
		"""
		Orthonormal DST-I along the last axis using rFFT and odd extension.
		Self-inverse: dstI1D_rfft(dstI1D_rfft(x)) == x (up to numerical error).

		x: (..., N)
		returns: (..., N)
		"""
		N = x.shape[-1]
		# Odd extension: y = [0, x, 0, -x[::-1]]  -> length L = 2*(N+1)
		y = jnp.concatenate(
			[jnp.zeros_like(x[..., :1]), x, jnp.zeros_like(x[..., :1]), -x[..., ::-1]],
			axis=-1
		)

		# Real FFT on the extended signal
		Y = jnp.fft.rfft(y, axis=-1)  # default 'backward' norm

		# DST-I coefficients are proportional to the imaginary part at bins 1..N
		# For L = 2*(N+1), Im(Y[..., k]) = 2 * sum_n x_n * sin(pi*k*n/(N+1)), k=1..N
		# Orthonormal scaling factor:
		scale = 0.5 * jnp.sqrt(2.0 / (N + 1))

		S = -jnp.imag(Y[..., 1:N+1]) * scale

		return S

	def init_matrix(self):
		nr,nz = self.R.shape

		Nint = nz - 2
		m = jnp.arange(1, Nint + 1)

		# Discrete eigenvalues of the 1D FD Laplacian D_ZZ (Dirichlet, interior)
		# D_ZZ eigenvalues are negative: lambda_m = -(2/dZ^2) * (1 - cos(m*pi/(Nint+1)))
		# We'll store mu_m = +(2/dZ^2)*(1 - cos(...)) and subtract it later.
		mu = (2.0 / self.dZ**2) * (1.0 - jnp.cos(m * jnp.pi / (Nint + 1)))  # mu_m > 0

		# R-direction FD operator for GS operator (-1/R*D_R+D_RR) with Dirichlet at R=0, R=L_R
		Rvec=self.R[:,0]
		Rvecsub=-Rvec[1:]
		Rvecsup=-Rvec[:-1]
		sub = -1.0/(2.0*Rvecsub*self.dR) + jnp.full(nr - 1, 1.0 / self.dR**2)       # lower diagonal
		sup = 1.0/(2.0*Rvecsup*self.dR) + jnp.full(nr - 1, 1.0 / self.dR**2)       # upper diagonal
		main_base = jnp.full(nr, -2.0 / self.dR**2)    # diagonal of D_RR

		# Batch diagonals for all modes
		diag_batch = jnp.tile(main_base, (mu.shape[0], 1))
		diag_batch = diag_batch.at[:, 1:-1].add(-mu[:, None])  # subtract for Δ* psi = f
		diag_batch = diag_batch.at[:, 0].set(1.0)
		diag_batch = diag_batch.at[:, -1].set(1.0)
		self.diag_batch = diag_batch

		sub = sub.at[-1].set(0.0)
		sup = sup.at[0].set(0.0)
		sub1 = jnp.append(0.0,sub)
		sup1 = jnp.append(sup,0.0)
		self.dl_batch = jnp.tile(sub1, (mu.shape[0], 1))
		self.du_batch = jnp.tile(sup1, (mu.shape[0], 1))


	   # Decide backend ONCE
		devices = jax.devices()
		has_gpu = any(d.platform == "gpu" for d in devices)

		# Place arrays and select solver ONCE
		# if has_gpu:
		# 	self._solve_tridiag = tri_default
		# else:
		self._solve_tridiag = thomas_solve_fused_batched 


	@jax.jit
	def __call__(self, rhs):

		# prepare rhs
		# -------------------------------
		# Decomposition: psi = g + w, w|_{Z=0,L_Z} = 0
		phi0 = rhs[:, 0]      # psi at Z=0
		phiL = rhs[:, -1]     # psi at Z=L_Z

		# Choose g to match Z-boundaries (linear in Z is sufficient)
		# g(R,Z) = phi0(R) + (phiL(R) - phi0(R)) * (Z / L_Z)
		g = phi0[:, None] + (phiL - phi0)[:, None] * ((self.Z - self.Zmin)/(self.Zmax - self.Zmin))

		# Compute discrete Laplacian of g (consistent FD):
		# g_RR: second difference in R
		g_RR = jnp.zeros_like(g)
		g_RR = g_RR.at[1:-1, :].set((g[:-2, :] - 2.0 * g[1:-1, :] + g[2:, :]) / self.dR**2)

		# g_ZZ: second difference in Z (will be ~0 for linear Z, but compute for generality)
		g_ZZ = jnp.zeros_like(g)
		g_ZZ = g_ZZ.at[:, 1:-1].set((g[:, :-2] - 2.0 * g[:, 1:-1] + g[:, 2:]) / self.dZ**2)

		# iR_g_R: 1/R*dgdr
		iR = 1.0/(2.0*self.R*self.dR)
		iRgR = jnp.zeros_like(g)
		iRgR = iRgR.at[1:-1, :].set(iR[1:-1,:]*(g[2:, :]-g[:-2, :]))

		Delta_g = g_RR + g_ZZ - iRgR

		# Modified RHS for w: Laplacian(w) = f - Laplacian(g)
		F = rhs - Delta_g

		# Z interior
		F_int = F[:, 1:-1]

		# -------------------------------
		# DST in Z (interior only)
		F_hat = self.dstI1D(F_int)  # shape (NR, NZ-2)

		# Prepare boundary values of w on R edges and transform them too:
		w_b0 = (rhs[0, 1:-1] - g[0, 1:-1])        # w at R=0, Z interior
		w_bL = (rhs[-1, 1:-1] - g[-1, 1:-1])      # w at R=L_R, Z interior
		w_b0_hat = self.dstI1D(w_b0)                   # shape (Nint,)
		w_bL_hat = self.dstI1D(w_bL)                   # shape (Nint,)

		rhs_batch = F_hat.T
		rhs_batch = rhs_batch.at[:, 0].set(w_b0_hat)
		rhs_batch = rhs_batch.at[:, -1].set(w_bL_hat)

		# Solve all modes using jax.lax.tridiagonal_solve
		# psi_modes = jax.lax.linalg.tridiagonal_solve(self.dl_batch, self.diag_batch, self.du_batch, rhs_batch[:,:,None])
		psi_modes = self._solve_tridiag(self.dl_batch, self.diag_batch, self.du_batch, rhs_batch)
		# psi_modes = thomas_solve_fused_batched(self.dl_batch, self.diag_batch, self.du_batch, rhs_batch)
		w_hat_int = psi_modes.T  # shape (NR, Nint)

		# Inverse DST to reconstruct w in physical space on Z interior
		w_int = self.dstI1D(w_hat_int)  # shape (NR, Nint)

		# Assemble full w with zero Z boundaries
		w = jnp.zeros_like(rhs)
		w = w.at[:, 1:-1].set(w_int)
		# Z=0, Z=L_Z remain zero by construction

		# Recover psi = g + w
		psi = g + w

		return psi.reshape(-1)