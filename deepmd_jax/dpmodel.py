import jax.numpy as jnp
import jax
from jax import vmap, value_and_grad
import flax.linen as nn
from .utils import sr, embedding_net, fitting_net, get_relative_coord, slice_t, concat_t, linear_init, ones_init, zeros_init, reorder_by_device
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

class DPModel(nn.Module):
    params: dict
    def get_stats(self, coord_B3N, box_B33, static_args):
        r_BNM = vmap(get_relative_coord,(0,0,None))(coord_B3N, box_B33, static_args['lattice'])[1]
        sr_BNM = sr(r_BNM, self.params['rcut'])
        sr_BnM = slice_type(sr_BNM, static_args['type_idx'], 1)
        self.params['sr_mean'] = [jnp.mean(i[i>1e-15]) for i in sr_BnM]
        self.params['sr_std'] = [jnp.std(i[i>1e-15]) for i in sr_BnM]
        self.params['norm'] = (sr_BNM > 0).sum(2).mean() + 1

    def get_idx(self, static_args, nbrs_lists, coord):
        if nbrs_lists is not None:
            K, type_idx, ntype_idx, type_idx_new = static_args['K'], static_args['type_idx'], [0], [0]
            coord = reorder_by_device(coord.T, tuple(type_idx), K).T
            [ntype_idx.append(ntype_idx[-1]+nbrs.idx.shape[1]) for nbrs in nbrs_lists]
            [type_idx_new.append(type_idx_new[-1]-(-type_idx[i+1]+type_idx[i])//K) for i in range(len(type_idx)-1)]
            nbrs_idx = jnp.concatenate([nbrs.idx for nbrs in nbrs_lists], axis=1)
            return type_idx_new, ntype_idx, nbrs_idx, K, coord
        else:
            return static_args['type_idx'], static_args['ntype_idx'], None, 1, coord
            
    @nn.compact
    def __call__(self, coord_3N, box_33, static_args, nbrs_lists=None):
        type_idx, ntype_idx, nbrs_idx, K, coord_3N = self.get_idx(static_args, nbrs_lists, coord_3N)
        # compute relative coordinates x_3NM, distance r_NM, s(r) and normalized s(r)
        x_3NM, r_NM = get_relative_coord(coord_3N, box_33, static_args['lattice'] if nbrs_idx is None else nbrs_idx)
        sr_nM = slice_t(sr(r_NM, self.params['rcut']), type_idx, 0, K=K)
        sr_norm_NM = concat_t([x/y for x,y in zip(sr_nM, self.params['sr_std'])], K=K)
        sr_centernorm_nm = [slice_t((x-y)/z,ntype_idx,1) for x,y,z in zip(sr_nM,self.params['sr_mean'],self.params['sr_std'])]
        # environment matrix: sr_norm_NM(0th-order), R_3NM(1st-order), R2_6NM(2nd-order))
        (N, M), C = r_NM.shape, self.params['embed_widths'][-1]
        x_norm_3NM = x_3NM / (r_NM+1e-16) 
        R_3NM = 3**0.5 * sr_norm_NM * x_norm_3NM
        R2diag_3NM = 9**0.5 * sr_norm_NM * ((x_norm_3NM)**2 - 1/3)
        R2offd_3NM = 18**0.5 * sr_norm_NM * (x_norm_3NM*jnp.stack([x_norm_3NM[1],x_norm_3NM[2],x_norm_3NM[0]]))
        R_XNM = jnp.concatenate([sr_norm_NM[None],R_3NM] + ([R2diag_3NM,R2offd_3NM] if self.params['use_2nd'] else []))
        # compute embedding feature and atomic features T for 0,1,2 order
        embed_NMC = concat_t([concat_t([embedding_net(self.params['embed_widths'])(
                        sr[:,:,None]) for sr in SR], axis=1) for SR in sr_centernorm_nm], K=K)
        T_NXC = (R_XNM.transpose(1,0,2) @ embed_NMC) / self.params['norm']
        # compute fitting net with input G = T @ T_sub; energy is sum of output; A for any axis dimension
        T_NC, T_N3C, T_N6C = T_NXC[:,0]+self.param('Tbias',zeros_init,(1,C)), T_NXC[:,1:4], T_NXC[:,4:]  
        G_NAC = T_NC[:,None]*T_NC[:,:self.params['axis'],None] + (T_N3C[:,:,None]*T_N3C[:,:,:self.params['axis'],None]).sum(1)
        if self.params['use_2nd']:
            G1_axis_N3A = T_N3C[:,:,self.params['axis']:2*self.params['axis']]
            G11_axis_N6A = jnp.concatenate([G1_axis_N3A**2, 2**0.5 * G1_axis_N3A*jnp.stack([
                        G1_axis_N3A[:,1],G1_axis_N3A[:,2],G1_axis_N3A[:,0]], axis=1)], axis=1)
            G2_axis_N6A = G11_axis_N6A + T_N6C[:,:,self.params['axis']:2*self.params['axis']]
            G_NAC += (G2_axis_N6A[...,None] * T_N6C[:,:,None]).sum(1)
        fit_n1 = [fitting_net(self.params['fit_widths'])(i) for i in slice_t(G_NAC.reshape(N,-1),type_idx,0,K=K)]
        energy = sum([(f+Eb).sum() for f,Eb in zip(fit_n1, self.params['Ebias'])])
        debug = (r_NM, T_NXC, fit_n1)
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
    
class DPMPModel(DPModel):
    @nn.compact
    def __call__(self, coord_3N, box_33, static_args):
        # compute relative coordinates x, relative distance r, and s(r)
        x_3NM, r_NM = get_relative_coord(coord_3N, box_33, static_args['lattice'])
        (N, M) = r_NM.shape
        E, B = self.params['embedMP_widths'][0], self.params['embedMP_widths'][-1]
        # prepare normalized s(r)
        sr_nM = slice_type(sr(r_NM, self.params['rcut']), static_args['type_idx'], 0)
        sr_norm_NM = jnp.concatenate(list(map(lambda x,y:x/y, sr_nM, self.params['sr_std'])))
        sr_centernorm_nm = [slice_type(s,static_args['ntype_idx'],1) for s in
                map(lambda x,y,z:(x-y)/z,sr_nM,self.params['sr_mean'],self.params['sr_std'])]
        x_norm_3NM = x_3NM / (r_NM+1e-16) * (r_NM > 2e-15)
        R2diag_3NM = 9**0.5 * sr_norm_NM * ((x_norm_3NM)**2 - (x_norm_3NM**2).mean(0))
        R2offd_3NM = 18**0.5 * sr_norm_NM * (x_norm_3NM*jnp.stack([x_norm_3NM[1],x_norm_3NM[2],x_norm_3NM[0]]))
        R_3NM = 3**0.5 * sr_norm_NM * x_norm_3NM
        R_4NM = jnp.concatenate([sr_norm_NM[None],R_3NM])
        R_3nm = [slice_type(r,static_args['ntype_idx'],2) for r in slice_type(R_3NM,static_args['type_idx'],1)]
        # compute embedding net features
        embedR_nmE = [[embedding_net(self.params['embedIJ_widths']+(E,), out_linear_only=True)(j[:,:,None]) for j in i] for i in sr_centernorm_nm]
        embed_NMB = jnp.concatenate([jnp.concatenate([jnp.concatenate([embedding_net(self.params['embedIJ_widths'])(
                        j[:,:,None]) for _ in range(2)], axis=-1) for j in i], axis=1) for i in sr_centernorm_nm])
        # Beginning MP, Compute atomic features T, linear transform into F, K = number of atom types, D = 4C
        B, C = embed_NMB.shape[-1], embed_NMB.shape[-1]//2
        T_N4B = (R_4NM.transpose(1,0,2) @ embed_NMB) / self.params['norm']
        T_NB, T_N3B = T_N4B[:,0], T_N4B[:,1:]
        T_23NC, T_2NC = T_N3B.reshape(N,3,2,-1).transpose(2,1,0,3), T_NB.reshape(N,2,-1).transpose(1,0,2)
        T_2ND = (T_2NC[:,:,None] * T_2NC[:,:,:4,None] + (T_23NC[:,:,:,None] * T_23NC[...,:4,None]).sum(1)).reshape(2,N,-1)
        F_2KnE = [T[:,None] @ (self.param('linear1_%d'%i,linear_init,(2,len(R_3nm),4*C,E)) * (self.param('norm_1_%d'%i,ones_init,(2,len(R_3nm),1,1)))**2) for i,T in enumerate(slice_type(T_2ND,static_args['type_idx'],1))]
        F_2K3nE = [T[:,None] @ (self.param('linear3_%d'%i,linear_init,(2,len(R_3nm),1,C,E)) * (self.param('norm_2_%d'%i,ones_init,(2,len(R_3nm),1,1,1)))**2) for i,T in enumerate(slice_type(T_23NC,static_args['type_idx'],2))]
        # add all features for each type pair (n,m)
        FI_n1E = [f[0,:,:,None] for f in F_2KnE]
        FJ_1mE = [[(f[1,i,:,None]*jnp.ones((M//N,1))).reshape(1,-1,E) for f in F_2KnE] for i in range(len(R_3nm))]
        FI3_nmE = [[R[j].transpose(1,2,0) @ F[0,j].transpose(1,0,2) for j in range(len(R_3nm))] for F,R in zip(F_2K3nE,R_3nm)]
        FJ3_nmE = [[(r.transpose(2,1,0) @ (F[1,i,:,:,None]*jnp.ones((M//N,1))).reshape(3,-1,E).transpose(1,0,2)).transpose(1,0,2)
                    for F,r in zip(F_2K3nE,R)] for i,R in enumerate(R_3nm)]
        F_nmE = [[sum(f) for f in zip(*F)] for F in zip(FI_n1E, FJ_1mE, FI3_nmE, FJ3_nmE, embedR_nmE)]
        embedMP_nmB = [[embedding_net(self.params['embedMP_widths'], in_bias_only=True, dt_layers=range(2,len(self.params['embedMP_widths'])))(f) for f in F] for F in F_nmE]
        embedMP_NMB = jnp.concatenate([jnp.concatenate(i, axis=1) for i in embedMP_nmB])
        # Begin fitting, Compute atomic features T, fitting net input G = T @ T_sub; A for any axis dimension
        R_XNM = jnp.concatenate([R_4NM,R2diag_3NM,R2offd_3NM]) if self.params['use_2nd'] else R_4NM
        T_NXC = (R_XNM.transpose(1,0,2) @ embedMP_NMB) / self.params['norm']
        T_NC, T_N3C, T_N6C = T_NXC[:,0]+self.param('Tbias',zeros_init,(1,B)), T_NXC[:,1:4], T_NXC[:,4:]  
        G00_NAC = (T_NC[:,None] * T_NC[:,:self.params['axis'][0],None])
        G11_NAC = (T_N3C[:,:,None] * T_N3C[:,:,self.params['axis'][0]:sum(self.params['axis'][:2]),None]).sum(1)
        if self.params['use_2nd']:
            G1_axis_N3A = T_N3C[:,:,sum(self.params['axis'][:2]):sum(self.params['axis'][:3])]
            G11_axis_N6A = jnp.concatenate([G1_axis_N3A**2, 2**0.5 * G1_axis_N3A*jnp.stack([
                        G1_axis_N3A[:,1],G1_axis_N3A[:,2],G1_axis_N3A[:,0]], axis=1)], axis=1)
            G2_axis_N6A = T_N6C[:,:,sum(self.params['axis'][:3]):sum(self.params['axis'])]
            G121_NAC = (G11_axis_N6A[...,None] * T_N6C[:,:,None]).sum(1)
            G22_NAC = (G2_axis_N6A[...,None] * T_N6C[:,:,None]).sum(1)
        G_ND = jnp.concatenate([G00_NAC,G11_NAC]+([G121_NAC,G22_NAC] if self.params['use_2nd'] else []), axis=1).reshape(N,-1)
        fit_n1 = [fitting_net(self.params['fit_widths'])(i) for i in slice_type(G_ND,static_args['type_idx'],0)]
        energy = sum(list(map(lambda x,y:(x+y).sum(), fit_n1, self.params['Ebias'])))
        debug = (r_NM, FI3_nmE, FJ3_nmE, FI_n1E, FJ_1mE, F_nmE, fit_n1)
        return energy, debug
