import jax.numpy as jnp
import jax
from jax import vmap, value_and_grad, lax
import flax.linen as nn
from .utils import *

class DPModel(nn.Module):
    params: dict
    def get_input(self, coord, static_args, nbrs_nm):
        type_count = np.pad(np.array(static_args['type_count']), (0,self.params['ntypes']-len(static_args['type_count'])))[self.params['valid_types']]
        compress = self.params.get('is_compressed', False)
        if self.params['atomic']:
            nsel = [list(self.params['valid_types']).index(i) for i in self.params['nsel']]
        else:
            nsel = list(range(len(type_count)))
        if nbrs_nm is not None:
            nbrs_nm = [[nbrs_nm[i][j] for j in self.params['valid_types']] for i in self.params['valid_types']]
            K = jax.device_count()
            type_count_new = [-(-type_count[i]//K) for i in range(len(type_count))]
            mask = get_mask_by_device(type_count)
            coord = reorder_by_device(coord, type_count)
            return coord, type_count_new, mask, compress, K, nsel, nbrs_nm
        else:
            return coord, type_count, jnp.ones_like(coord[:,0]), compress, 1, nsel, None
            
    @nn.compact
    def __call__(self, coord_N3, box_33, static_args, nbrs_nm=None):
        # prepare input parameters
        coord_N3, type_count, mask, compress, K, nsel, nbrs_nm = self.get_input(coord_N3, static_args, nbrs_nm)
        A, L = self.params['axis'], static_args['lattice']['lattice_max'] if nbrs_nm is None else None
        # compute relative coordinates x_3NM, distance r_NM, s(r) and normalized s(r)
        x_n3m, r_nm = get_relative_coord(coord_N3, box_33, type_count, static_args.get('lattice',None), nbrs_nm)
        sr_nm = [[sr(r, self.params['rcut']) for r in R] for R in r_nm]
        sr_norm_nm = [[r/std for r in R] for R,std in zip(sr_nm,self.params['sr_std'])]
        sr_centernorm_nm = [[(r-mean)/std for r in R] for R,mean,std in zip(sr_nm,self.params['sr_mean'],self.params['sr_std'])]
        # environment matrix: sr_norm_nm (0th-order), R_n3m (1st-order), R2_n6m (2nd-order)
        x_norm_n3m = [[x/(r+1e-16)[:,None] for x,r in zip(X,R)] for X,R in zip(x_n3m,r_nm)]
        R_n3m = [[3**0.5 * sr[:,None] * x for sr,x in zip(SR,X)] for SR,X in zip(sr_norm_nm,x_norm_n3m)]
        R_n4m = [[concat([sr[:,None],r], axis=1) for sr,r in zip(SR,R)] for SR,R in zip(sr_norm_nm,R_n3m)]
        R_nsel6m = [[3*sr[:,None]*tensor_3to6(x,axis=1,bias=1/3) for sr,x
                     in zip(sr_norm_nm[nsel[i]],x_norm_n3m[nsel[i]])] for i in range(len(nsel))]
        R_nselXm = [[concat([sr[:,None],r3] + ([r6] if self.params['use_2nd'] else []), axis=1)
                    for sr,r3,r6 in zip(sr_norm_nm[nsel[i]],R_n3m[nsel[i]],R_nsel6m[i])] for i in range(len(nsel))]
        # compute embedding net and atomic features T
        if not self.params.get('use_mp', False): # original DP without message passing
            T_NselXW = concat([sum([embedding_net(self.params['embed_widths'])(sr[:,:,None],compress,rx) for sr,rx in
                        zip(sr_centernorm_nm[nsel[i]], R_nselXm[i])]) for i in range(len(nsel))], K=K) / self.params['Nnbrs']
        else: # Message Passing: Compute atomic features T; linear transform, add into F; Y=#types; B=2C, D=4C
            C, E = self.params['embed_widths'][-1], self.params['embedMP_widths'][0]
            embed_nselmE = [[embedding_net(self.params['embed_widths']+(E,), out_linear_only=True)(sr[:,:,None],
                            compress) for sr in sr_centernorm_nm[nsel[i]]] for i in range(len(nsel))]
            T_2_n4C = [[sum([embedding_net(self.params['embed_widths'])(sr[:,:,None],compress,r4) for sr,r4 in zip(SR,R4)])
                        / self.params['Nnbrs'] for SR,R4 in zip(sr_centernorm_nm,R_n4m)] for _ in range(2)]
            T_2_nD = [[(t[:,:,None]*t[:,:,:4,None]).sum(1).reshape(-1,4*C) for t in T] for T in T_2_n4C]
            T_2_n3C = [[t[:,1:] for t in T] for T in T_2_n4C]
            if nbrs_nm is not None:
                sharding = jax.sharding.PositionalSharding(jax.devices()).replicate()
                T_2_nD, T_2_n3C = lax.with_sharding_constraint([T_2_nD, T_2_n3C], sharding)
            F_nselmE = [[(linear_norm(E)(T_2_nD[0][i])[:,None]
                      + (linear_norm(E)(T_2_nD[1][j])[nbrs_nm[i][j]] if nbrs_nm is not None else
                         jnp.repeat(linear_norm(E)(T_2_nD[1][j]),L,axis=0))
                      + (R_n3m[i][j][...,None] * (linear_norm(E)(T_2_n3C[0][i])[:,:,None]
                          + (linear_norm(E)(T_2_n3C[1][j])[nbrs_nm[i][j]].transpose(0,2,1,3) if nbrs_nm is not None else
                            jnp.repeat(linear_norm(E)(T_2_n3C[1][j]),L,axis=0).transpose(1,0,2)))).sum(1)
                      + emb) * (self.param('layer_norm_%d_%d'%(i,j), ones_init, (1,))**2 if self.params['atomic'] else 1)
                        for j,emb in enumerate(EMB)] for i,EMB in zip(nsel,embed_nselmE)]
            T_NselXW = concat([sum([embedding_net(self.params['embedMP_widths'], in_bias_only=True,
                        dt_layers=range(2,len(self.params['embedMP_widths'])))(f, reducer=rx)
                        for f,rx in zip(F,RX)]) for F,RX in zip(F_nselmE, R_nselXm)],K=K) / self.params['Nnbrs'] 
        # compute fitting net with input G = T @ T_sub; energy is sum of output; A for any axis dimension
        T_NselW, T_Nsel3W, T_Nsel6W = T_NselXW[:,0]+self.param('Tbias',zeros_init,T_NselXW.shape[-1:]), T_NselXW[:,1:4], T_NselXW[:,4:] 
        G_NselAW = T_NselW[:,None]*T_NselW[:,:A,None] + (T_Nsel3W[:,:,None]*T_Nsel3W[:,:,:A,None]).sum(1)
        if self.params['use_2nd']:
            G2_axis_Nsel6A = tensor_3to6(T_Nsel3W[:,:,A:2*A], axis=1) + T_Nsel6W[:,:,A:2*A]
            G_NselAW += (G2_axis_Nsel6A[...,None] * T_Nsel6W[:,:,None]).sum(1)
        if not self.params['atomic']: # Energy prediction
            fit_n1 = [fitting_net(self.params['fit_widths'])(G) for G in split(G_NselAW.reshape(G_NselAW.shape[0],-1),type_count,0,K=K)]
            pred = (mask * concat([f[:,0]+Eb for f,Eb in zip(fit_n1,self.params['Ebias'])], K=K)).sum()
        else: # Atomic tensor prediction
            sel_count = [type_count[i] for i in nsel]
            fit_nselW = [fitting_net(self.params['fit_widths'], use_final=False)(G) for G in split(G_NselAW.reshape(G_NselAW.shape[0],-1),sel_count,0,K=K)]
            T_nsel3W = split(T_Nsel3W, sel_count, 0, K=K)            
            pred = concat([lax.with_sharding_constraint((f[:,None]*T).sum(-1)[:static_args['type_count'][self.params['nsel'][i]]],
                jax.sharding.PositionalSharding(jax.devices()).replicate()) for i,(f,T) in enumerate(zip(fit_nselW,T_nsel3W))])
        debug = T_NselXW
        return pred * self.params['out_norm'], debug

    def energy_and_force(self, variables, coord_N3, box_33, static_args, nbrs_lists=None):
        (pred, _), g = value_and_grad(self.apply, argnums=1, has_aux=True)(variables, coord_N3, box_33, static_args, nbrs_lists)
        return pred, -g
    
    def wc_predict(self, variables, coord_N3, box_33, static_args, nbrs_nm=None):
        wc_relative = self.apply(variables, coord_N3, box_33, static_args, nbrs_nm)[0]
        coord_ref = [c for i,c in enumerate(split(coord_N3, static_args['type_count'])) if i in self.params['nsel']]
        return concat(coord_ref) + wc_relative
        # return lax.with_sharding_constraint(concat(coord_ref) + wc_relative, jax.sharding.PositionalSharding(jax.devices()[0])
    
    def get_loss_fn(self):
        if self.params['atomic'] is False:
            vmap_energy_and_force = vmap(self.energy_and_force, (None, 0, 0, None))
            def loss_ef(variables, batch_data, pref, static_args):
                e, f = vmap_energy_and_force(variables, batch_data['coord'], batch_data['box'], static_args)
                le = ((batch_data['energy'] - e)**2).mean() / (f.shape[1])**2
                lf = ((batch_data['force'] - f)**2).mean()
                return pref['e']*le + pref['f']*lf, (le, lf)
            loss_and_grad = value_and_grad(loss_ef, has_aux=True)
            return loss_ef, loss_and_grad
        else:
            vmap_apply = vmap(self.apply, (None, 0, 0, None))
            def loss_atomic(variables, batch_data, static_args):
                pred, _ = vmap_apply(variables, batch_data['coord'], batch_data['box'], static_args)
                return ((batch_data['atomic'] - pred)**2).mean()
            loss_and_grad = value_and_grad(loss_atomic)
            return loss_atomic, loss_and_grad