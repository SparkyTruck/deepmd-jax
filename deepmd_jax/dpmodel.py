import jax.numpy as jnp
import jax
from jax import vmap, value_and_grad, lax
import flax.linen as nn
from .utils import *

class DPModel(nn.Module):
    params: dict
    def get_stats(self, coord_B3N, box_B33, static_args):
        r_BNM = vmap(get_relative_coord,(0,0,None))(coord_B3N, box_B33, static_args['lattice'])[1]
        sr_BNM = sr(r_BNM, self.params['rcut'])
        sr_BnM = split(sr_BNM, static_args['type_idx'], 1)
        self.params['sr_mean'] = [jnp.mean(i[i>1e-15]) for i in sr_BnM]
        self.params['sr_std'] = [jnp.std(i[i>1e-15]) for i in sr_BnM]
        self.params['norm'] = (sr_BNM > 0).sum(2).mean() + 1

    def get_idx(self, static_args, nbrs_lists):
        if nbrs_lists is not None:
            K, type_idx = static_args.get('K',1), static_args['type_idx']
            ntype_idx, type_idx_new = [0], [0]
            for i, nbrs in enumerate(nbrs_lists):
                ntype_idx.append(ntype_idx[-1]+nbrs.idx.shape[1])
                type_idx_new.append(type_idx_new[-1]-(-type_idx[i+1]+type_idx[i])//K)
            sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(K, 1)
            nbrs_lists = [lax.with_sharding_constraint(nbrs.idx, sharding) for nbrs in nbrs_lists]
            nbrs_idx = concat(nbrs_lists, axis=1)
            mask = get_mask_by_device(type_idx, K)
            return type_idx_new, ntype_idx, nbrs_lists, nbrs_idx, K, mask
        else:
            ntype_idx = tuple(static_args['lattice']['lattice_max']*idx for idx in static_args['type_idx'])
            return static_args['type_idx'], ntype_idx, None, None, 1, jnp.ones(1, dtype=jnp.float32)
            
    @nn.compact
    def __call__(self, coord_3N, box_33, static_args, nbrs_lists=None):
        type_idx, ntype_idx, nbrs_lists, nbrs_idx, K, mask = self.get_idx(static_args, nbrs_lists)
        # compute relative coordinates x_3NM, distance r_NM, s(r) and normalized s(r)
        x_3NM, r_NM = get_relative_coord(coord_3N, box_33, static_args.get('lattice',None), nbrs_idx)
        (N, M), A = r_NM.shape, self.params['axis'] 
        sr_nM = split(sr(r_NM, self.params['rcut']), type_idx, 0, K=K)
        sr_norm_NM = concat([x/y for x,y in zip(sr_nM, self.params['sr_std'])], K=K)
        sr_centernorm_nm = [split((x-y)/z,ntype_idx,1) for x,y,z in
                            zip(sr_nM,self.params['sr_mean'],self.params['sr_std'])]
        # environment matrix: sr_norm_NM(0th-order), R_3NM(1st-order), R2_6NM(2nd-order))
        x_norm_3NM = x_3NM / (r_NM+1e-16) 
        R_3NM = 3**0.5 * sr_norm_NM * x_norm_3NM
        R_4NM = concat([sr_norm_NM[None],R_3NM])
        R_6NM = 3 * sr_norm_NM * tensor_3to6(x_norm_3NM, axis=0, bias=1/3)
        R_XNM = concat([R_4NM, R_6NM]) if self.params['use_2nd'] else R_4NM
        # compute embedding feature and atomic features T for 0,1,2 order
        if not self.params.get('use_mp', False): # original DP without message passing
            embed_NMW = concat([concat([embedding_net(self.params['embed_widths'])(sr[:,:,None])
                                        for sr in SR], axis=1) for SR in sr_centernorm_nm], K=K)
        else: # Message Passing: Compute atomic features T; linear transform, add into F; Y=#types; B=2C, D=4C
            Y, C, E = len(type_idx)-1, self.params['embed_widths'][-1], self.params['embedMP_widths'][0]
            embedR_nmE = [[embedding_net(self.params['embed_widths']+(E,), out_linear_only=True)(
                            sr[:,:,None]) for sr in SR] for SR in sr_centernorm_nm]
            embed_NMB = concat([concat([concat([embedding_net(self.params['embed_widths'])(sr[:,:,None])
                        for _ in range(2)], axis=-1) for sr in SR], axis=1) for SR in sr_centernorm_nm], K=K)
            R_3nm = [split(r, ntype_idx, 2) for r in split(R_3NM, type_idx, 1, K=K)]
            T_N4B = (R_4NM.transpose(1,0,2) @ embed_NMB) / self.params['norm']
            T_ND = (T_N4B[:,:,None]*T_N4B[:,:,:4,None]).sum(1).reshape(N,-1)
            T_ND_n2 = [split(T,(0,4*C,8*C),-1) for T in split(T_ND, type_idx, 0, K=K)]
            T_3NC_n2 = [split(T,(0,C,2*C),-1) for T in split(T_N4B[:,1:].transpose(1,0,2), type_idx, 1, K=K)]
            if nbrs_lists is None:
                F_nmE = [[linear_norm(E)(T_ND_n2[i][0])[:,None] + jnp.tile(linear_norm(E)(T_ND_n2[j][1]),(M//N,1))
                            + (R_3nm[i][j][...,None] * (linear_norm(E)(T_3NC_n2[i][0])[:,:,None,]
                            + jnp.tile(linear_norm(E)(T_3NC_n2[j][1]),(M//N,1))[:,None])).sum(0)
                            + embedR_nmE[i][j]    for j in range(Y)] for i in range(Y)]
            else: # message passing requires broadcasting over devices
                nbrs_mn = [split(nbrs-type_idx[i]*K,type_idx,0,K=K) for i,nbrs in enumerate(nbrs_lists)]
                shard = jax.sharding.PositionalSharding(jax.devices()).replicate()
                T_ND_n2, T_3NC_n2 = lax.with_sharding_constraint([T_ND_n2, T_3NC_n2], shard)
                F_nmE = [[linear_norm(E)(T_ND_n2[i][0])[:,None]
                    + linear_norm(E)(T_ND_n2[j][1])[nbrs_mn[j][i]]
                    + (R_3nm[i][j][...,None] * (linear_norm(E)(T_3NC_n2[i][0])[:,:,None,]
                    + linear_norm(E)(T_3NC_n2[j][1])[:,nbrs_mn[j][i]])).sum(0)
                    + embedR_nmE[i][j]   for j in range(Y)] for i in range(Y)]
            embed_NMW = concat([concat([embedding_net(self.params['embedMP_widths'], in_bias_only=True,
                                                        dt_layers=range(2,len(self.params['embedMP_widths'])))(f)
                                                        for f in F], axis=1) for F in F_nmE], axis=0, K=K)
        # compute fitting net with input G = T @ T_sub; energy is sum of output; A for any axis dimension
        T_NXW = (R_XNM.transpose(1,0,2) @ embed_NMW) / self.params['norm']
        T_NW, T_N3W, T_N6W = T_NXW[:,0]+self.param('Tbias',zeros_init,T_NXW.shape[-1:]), T_NXW[:,1:4], T_NXW[:,4:] 
        G_NAW = T_NW[:,None]*T_NW[:,:A,None] + (T_N3W[:,:,None]*T_N3W[:,:,:A,None]).sum(1)
        if self.params['use_2nd']:
            G2_axis_N6A = tensor_3to6(T_N3W[:,:,A:2*A], axis=1) + T_N6W[:,:,A:2*A]
            G_NAW += (G2_axis_N6A[...,None] * T_N6W[:,:,None]).sum(1)
        fit_n1 = [fitting_net(self.params['fit_widths'])(i) for i in split(G_NAW.reshape(N,-1),type_idx,0,K=K)]
        energy = (mask * concat([f[:,0]+Eb for f,Eb in zip(fit_n1,self.params['Ebias'])], K=K)).sum()
        debug = (r_NM, T_NXW, fit_n1)
        return energy, debug

    def energy_and_force(self, variables, coord_3N, box_33, static_args):
        (energy, debug), g = value_and_grad(self.apply, argnums=1, has_aux=True)(variables, coord_3N, box_33, static_args)
        return energy, -g
    
    def get_loss_ef_fn(self):
        vmap_energy_and_force = vmap(self.energy_and_force, (None, 0, 0, None))
        def loss_ef(variables, batch_data, pref, static_args):
            e, f = vmap_energy_and_force(variables, batch_data['coord'], batch_data['box'], static_args)
            le = ((batch_data['energy'] - e)**2).mean()
            lf = ((batch_data['force'] - f)**2).mean()
            return pref['e']*le / f.shape[2] + pref['f']*lf, (le, lf)
        loss_ef_and_grad = value_and_grad(loss_ef, has_aux=True)
        return loss_ef, loss_ef_and_grad