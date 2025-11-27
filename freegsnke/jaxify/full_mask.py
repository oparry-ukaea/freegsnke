import jax
import jax.numpy as jnp
from jax import jit, lax

def bitmask_region_growing(psin, seed_i, seed_j, threshold=1.0, max_iters=1000):
    nx, ny = psin.shape
    mask = jnp.zeros_like(psin, dtype=bool)
    mask = mask.at[seed_i, seed_j].set(True)

    def cond_fn(state):
        mask, prev_mask, iter_count = state
        return jnp.any(mask != prev_mask) & (iter_count < max_iters)

    def body_fn(state):
        mask, prev_mask, iter_count = state
        up    = jnp.pad(mask[:-1, :], ((1, 0), (0, 0)))
        down  = jnp.pad(mask[1:, :], ((0, 1), (0, 0)))
        left  = jnp.pad(mask[:, :-1], ((0, 0), (1, 0)))
        right = jnp.pad(mask[:, 1:], ((0, 0), (0, 1)))
        neighbors = up | down | left | right
        new_mask = (psin < threshold) & neighbors
        updated_mask = mask | new_mask
        return updated_mask, mask, iter_count + 1

    final_mask, _, _ = lax.while_loop(cond_fn, body_fn, (mask, jnp.zeros_like(psin, dtype=bool), 0))
    return final_mask

@jax.jit
def inside_mask_(R, Z, psi, opoint, xpoint=[], mask_outside_limiter=None, psi_bndry=None):
    nx, ny = psi.shape
    mask = jnp.zeros_like(psi)

    Ro, Zo, psi_axis = opoint[0]
    if psi_bndry is None:
        _, _, psi_bndry = xpoint[0]

    psin = (psi - psi_axis) / (psi_bndry - psi_axis)

    # Block X-point regions
    rx = jnp.array([pt[0] for pt in xpoint])
    zx = jnp.array([pt[1] for pt in xpoint])
    ix = jnp.argmin(jnp.abs(R[:, 0][:, None] - rx[None, :]), axis=0)
    jx = jnp.argmin(jnp.abs(Z[0, :][None, :] - zx[:, None]), axis=1)

    offsets = jnp.array([-1, 0, 1])
    di, dj = jnp.meshgrid(offsets, offsets, indexing='ij')
    di = di.flatten()
    dj = dj.flatten()

    def block_fn(i0, j0):
        ii = jnp.clip(i0 + di, 0, nx - 1)
        jj = jnp.clip(j0 + dj, 0, ny - 1)
        return ii, jj

    block_vmap = jax.jit(jax.vmap(lambda i0, j0: block_fn(i0, j0), in_axes=(0, 0)))
    iis, jjs = block_vmap(ix, jx)
    mask = mask.at[iis,jjs].set(2)

    # Seed point for flood-fill
    rind = jnp.argmin(jnp.abs(R[:, 0] - Ro))
    zind = jnp.argmin(jnp.abs(Z[0, :] - Zo))

    # Bitmask flood-fill
    core_mask = bitmask_region_growing(psin, rind, zind)

    # Apply core mask to current mask
    mask = jnp.where(core_mask, 1, mask)

    # Vectorized revisit of X-point regions
    def revisit_fn(i0, j0):
        ii = jnp.clip(i0 + di, 0, nx - 1)
        jj = jnp.clip(j0 + dj, 0, ny - 1)
        psin_vals = psin[ii, jj]
        new_vals = jnp.where(psin_vals < 1.0, 1, 0)
        return ii, jj, new_vals

    revisit_vmap = jax.jit(jax.vmap(lambda i0, j0: revisit_fn(i0, j0), in_axes=(0, 0)))
    iis, jjs, vals = revisit_vmap(ix, jx)
    mask = mask.at[iis,jjs].set(vals)

    return mask == 1

@jit
def geom_inside_mask(R, Z, opoint, xpoint):
    slope = -(opoint[0, 0] - xpoint[0, 0]) / (opoint[0, 1] - xpoint[0, 1])
    interc = xpoint[0, 1] - slope * xpoint[0, 0]

    geom_mask = (
        (opoint[0, 1] - (slope * opoint[0, 0] + interc))
        * (Z - (slope * R + interc))
    ) > 0

    return geom_mask

@jit
def inside_mask(R, Z, psi, opoint, xpoint=[], mask_outside_limiter=None, psi_bndry=None, use_geom=True):
    mask = inside_mask_(R, Z, psi, opoint, xpoint, mask_outside_limiter, psi_bndry)

    if use_geom:
        mask = mask & geom_inside_mask(R, Z, opoint, xpoint)
        if len(xpoint) > 1:
            close_double_null = jnp.abs((xpoint[0, 2] - xpoint[1, 2]) / (opoint[0, 2] - xpoint[0, 2])) < 0.1
            mask = lax.cond(
                close_double_null,
                lambda m: m & geom_inside_mask(R, Z, opoint, xpoint[1:]),
                lambda m: m,
                mask
            )

    return mask
