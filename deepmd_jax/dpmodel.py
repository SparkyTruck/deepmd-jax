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
        self.params['Nnbrs'] = (sr_BNM > 0).sum(2).mean() + 1

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
        (N, M), A, Y = r_NM.shape, self.params['axis'], len(type_idx)-1
        sr_nM = split(sr(r_NM, self.params['rcut']), type_idx, K=K)
        sr_norm_NM = concat([x/y for x,y in zip(sr_nM, self.params['sr_std'])], K=K)
        sr_centernorm_nm = [split((x-y)/z,ntype_idx,1) for x,y,z in
                            zip(sr_nM,self.params['sr_mean'],self.params['sr_std'])]
        # environment matrix: sr_norm_NM(0th-order), R_3NM(1st-order), R2_6NM(2nd-order))
        x_norm_3NM = x_3NM / (r_NM+1e-16) 
        R_3NM = 3**0.5 * sr_norm_NM * x_norm_3NM
        R_3nm = [split(r, ntype_idx, 2) for r in split(R_3NM, type_idx, 1, K=K)]
        R_4NM = concat([sr_norm_NM[None],R_3NM])
        nsel = self.params['nsel'] if self.params['atomic'] else list(range(Y))
        if len(nsel) < Y: # filter type for atomic models
            sr_norm_NM = concat([sr for i,sr in enumerate(split(sr_norm_NM,type_idx,K=K)) if i in nsel])
            R_3NM = concat([sr for i,sr in enumerate(split(R_3NM,type_idx,1,K=K)) if i in nsel])
            x_norm_3NM = concat([sr for i,sr in enumerate(split(x_norm_3NM,type_idx,1,K=K)) if i in nsel])
        R_6NM = 3 * sr_norm_NM * tensor_3to6(x_norm_3NM, axis=0, bias=1/3)
        R_XNM = concat([sr_norm_NM[None],R_3NM] + ([R_6NM] if self.params['use_2nd'] else []))
        # compute embedding feature and atomic features T for 0,1,2 order
        if not self.params.get('use_mp', False): # original DP without message passing
            embed_NMW = concat([concat([embedding_net(self.params['embed_widths'])(sr[:,:,None]) for sr in SR],
                                       axis=1) for i,SR in enumerate(sr_centernorm_nm) if i in nsel], K=K)
        else: # Message Passing: Compute atomic features T; linear transform, add into F; Y=#types; B=2C, D=4C
            C, E = self.params['embed_widths'][-1], self.params['embedMP_widths'][0]
            embedR_nmE = [[embedding_net(self.params['embed_widths']+(E,), out_linear_only=True)(
                            sr[:,:,None]) for sr in SR] for i,SR in enumerate(sr_centernorm_nm) if i in nsel]
            embed_NMB = concat([concat([concat([embedding_net(self.params['embed_widths'])(sr[:,:,None])
                        for _ in range(2)], axis=-1) for sr in SR], axis=1) for SR in sr_centernorm_nm], K=K)
            T_N4B = (R_4NM.transpose(1,0,2) @ embed_NMB) / self.params['Nnbrs']
            T_ND = (T_N4B[:,:,None]*T_N4B[:,:,:4,None]).sum(1).reshape(N,-1)
            T_ND_n2 = [split(T,(0,4*C,8*C),-1) for T in split(T_ND, type_idx, K=K)]
            T_3NC_n2 = [split(T,(0,C,2*C),-1) for T in split(T_N4B[:,1:].transpose(1,0,2), type_idx, 1, K=K)]
            if nbrs_lists is not None:
                nbrs_mn = [split(nbrs%type_idx[-1] + (nbrs//type_idx[-1])*(type_idx[i+1]-type_idx[i]) - type_idx[i],
                                 type_idx, K=K) for i, nbrs in enumerate(nbrs_lists)]
                sharding = jax.sharding.PositionalSharding(jax.devices()).replicate()
                T_ND_n2, T_3NC_n2 = lax.with_sharding_constraint([T_ND_n2, T_3NC_n2], sharding)
            F_nmE = [[linear_norm(E)(T_ND_n2[nsel[i]][0])[:,None]
                    + (linear_norm(E)(T_ND_n2[j][1])[nbrs_mn[j][nsel[i]]] if nbrs_lists is not None else
                        jnp.repeat(linear_norm(E)(T_ND_n2[j][1]),M//N,axis=0))
                    + (R_3nm[nsel[i]][j][...,None] * (linear_norm(E)(T_3NC_n2[nsel[i]][0])[:,:,None,]
                          + (linear_norm(E)(T_3NC_n2[j][1])[:,nbrs_mn[j][nsel[i]]] if nbrs_lists is not None else
                            jnp.repeat(linear_norm(E)(T_3NC_n2[j][1]),M//N,axis=1)[:,None]))).sum(0)
                    + embedR_nmE[i][j]   for j in range(Y)] for i in range(len(nsel))]
            embed_NMW = concat([concat([embedding_net(self.params['embedMP_widths'], in_bias_only=True,
                                                        dt_layers=range(2,len(self.params['embedMP_widths'])))(f)
                                                        for f in F], axis=1) for F in F_nmE], axis=0, K=K)
        # compute fitting net with input G = T @ T_sub; energy is sum of output; A for any axis dimension
        T_NXW = (R_XNM.transpose(1,0,2) @ embed_NMW) / self.params['Nnbrs']
        T_NW, T_N3W, T_N6W = T_NXW[:,0]+self.param('Tbias',zeros_init,T_NXW.shape[-1:]), T_NXW[:,1:4], T_NXW[:,4:] 
        G_NAW = T_NW[:,None]*T_NW[:,:A,None] + (T_N3W[:,:,None]*T_N3W[:,:,:A,None]).sum(1)
        if self.params['use_2nd']:
            G2_axis_N6A = tensor_3to6(T_N3W[:,:,A:2*A], axis=1) + T_N6W[:,:,A:2*A]
            G_NAW += (G2_axis_N6A[...,None] * T_N6W[:,:,None]).sum(1)
        if not self.params['atomic']: # Energy prediction
            fit_n1 = [fitting_net(self.params['fit_widths'])(G)
                      for G in split(G_NAW.reshape(N,-1),type_idx,0,K=K)]
            pred = (mask * concat([f[:,0]+Eb for f,Eb in zip(fit_n1,self.params['Ebias'])], K=K)).sum()
        else: # Atomic tensor prediction
            sel_idx = [0]
            [sel_idx.append(sel_idx[-1]+type_idx[i+1]-type_idx[i]) for i in nsel]
            fit_nW = [fitting_net(self.params['fit_widths'], use_final=False)(G)
                       for G in split(G_NAW.reshape(sel_idx[-1],-1),sel_idx,0,K=K)]
            T_n3W = split(T_N3W, sel_idx, 0, K=K)
            pred = concat([(f[:,None]*T).sum(-1)[:static_args['type_idx'][nsel[i]+1]-static_args['type_idx'][nsel[i]]]
                           for i,(f,T) in enumerate(zip(fit_nW,T_n3W))])
        debug = (r_NM, T_NXW)
        return pred / self.params['out_norm'], debug

    def energy_and_force(self, variables, coord_3N, box_33, static_args):
        (pred, _), g = value_and_grad(self.apply, argnums=1, has_aux=True)(variables, coord_3N, box_33, static_args)
        return pred, -g
    
    def get_loss_fn(self):
        if self.params['atomic'] is False:
            vmap_energy_and_force = vmap(self.energy_and_force, (None, 0, 0, None))
            def loss_ef(variables, batch_data, pref, static_args):
                e, f = vmap_energy_and_force(variables, batch_data['coord'], batch_data['box'], static_args)
                le = ((batch_data['energy'] - e)**2).mean() / (f.shape[2])**2
                lf = ((batch_data['force'] - f)**2).mean()
                return pref['e']*le + pref['f']*lf, (le, lf)
            loss_and_grad = value_and_grad(loss_ef, has_aux=True)
            return loss_ef, loss_and_grad
        else:
            vmap_apply = vmap(self.apply, (None, 0, 0, None))
            def loss_atomic(variables, batch_data, static_args):
                pred, _ = vmap_apply(variables, batch_data['coord'], batch_data['box'], static_args)
                return ((batch_data['atomic'] - pred.transpose(0,2,1))**2).mean()
            loss_and_grad = value_and_grad(loss_atomic)
            return loss_atomic, loss_and_grad