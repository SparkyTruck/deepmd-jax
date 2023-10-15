import jax.numpy as jnp
from jax import vmap, value_and_grad
import flax.linen as nn
from .utils import sr, embedding_net, fitting_net, get_relative_coord, slice_type

class DPModel(nn.Module):
    params: dict

    def get_stats(self, coord_B3N, box_B33, static_args):
        x_B3NM, r_BNM = vmap(get_relative_coord,(0,0,None))(coord_B3N, box_B33, static_args['lattice'])
        sr_BNM = sr(r_BNM, static_args['rcut'])
        sr_BnM = slice_type(sr_BNM, static_args['type_index'], 1)
        xrsr_B3nM = slice_type(x_B3NM*(sr_BNM/r_BNM)[:,None], static_args['type_index'], 2)
        self.params['srmean'] = [jnp.mean(i[i>1e-15]) for i in sr_BnM]
        self.params['srstd'] = [jnp.std(i[i>1e-15]) for i in sr_BnM]
        self.params['xrsrstd'] = [jnp.std(i[jnp.abs(i)>1e-15]) for i in xrsr_B3nM]
        self.params['avgneigh'] = (sr_BNM > 0).sum(2).mean()
        self.params['normalizer'] = self.params['avgneigh'] + 1
        self.params['e3norm'] = 1.

    @nn.compact
    def __call__(self, coord_3N, box_33, static_args):
        # compute relative coordinates and s(r)
        x_3NM, r_NM = get_relative_coord(coord_3N, box_33, static_args['lattice'])
        sr_NM = sr(r_NM, static_args['rcut'])
        (N, M), C = r_NM.shape, self.params['embed_widths'][-1]
        # prepare normalized s(r)
        sr_nM = slice_type(sr_NM, static_args['type_index'], 0)
        srbiasnorm_NM = jnp.concatenate(list(map(lambda x,y:x/y, sr_nM, self.params['srstd'])))
        srnorm_nM = list(map(lambda x,y,z:(x-y)/z, sr_nM, self.params['srmean'], self.params['srstd']))
        srnorm_nm = [slice_type(s,static_args['ntype_index'],1) for s in srnorm_nM]
        # compute distance matrix R = (inv:srbiasnorm_NM, equiv:R_3NM)
        rsr_nM = list(map(lambda x,y:x/y, slice_type(sr_NM/r_NM,static_args['type_index'],0), self.params['xrsrstd']))
        R_3NM = self.params['e3norm']*(jnp.concatenate(rsr_nM)[None]+1e-15)*x_3NM
        # alternative implementation
        # R_4NM = jnp.concatenate([srbiasnorm_NM[None], R_3NM])
        # R_4nm = [slice_type(r,static_args['ntype_index'],2) for r in slice_type(R_4NM,static_args['type_index'],1)]
        # R_4nM = slice_type(R_4NM,static_args['type_index'],1)
        # G_bias = self.param('Gbias',nn.initializers.normal(stddev=0.01),(C,))
        # G_4nC = [(jnp.concatenate([embedding_net(self.params['embed_widths'])(i[:,:,None])
        #                           for i in j], axis=1) * r[...,None]).sum(2) + G_bias for j, r in zip(srnorm_nm, R_4nM)]
        # G_4nC = [sum([(embedding_net(self.params['embed_widths'])(i[:,:,None])*j[...,None]).sum(2)
        #                for i,j in J]) for J in map(zip, srnorm_nm, R_4nm)]
        # G_4NC = jnp.concatenate(G_4nC, axis=1) + self.param('Gbias',nn.initializers.normal(stddev=0.01),(C,))
        # Feat_nX = [(g[:,:,None,:] * g[:,:,:self.params['axis_neuron'],None]).sum(0).reshape(g.shape[1],-1) for g in G_4nC]
        # fit_n1 = [fitting_net(self.params['fit_widths'])(i) for i in Feat_nX]
        # compute feature matrix G = E @ R and Feat = GG^T
        embed_nmC = [[embedding_net(self.params['embed_widths'])(j[:,:,None]) for j in i] for i in srnorm_nm]
        embed_NMC = jnp.concatenate([jnp.concatenate(i, axis=1) for i in embed_nmC], axis=0)
        Ginv_NC = (srbiasnorm_NM[:,:,None] * embed_NMC).sum(1) / self.params['normalizer']
        Geqv_3NC = (R_3NM[...,None] * embed_NMC).sum(2) / self.params['normalizer']
        Gbias = self.param('Gbias', nn.initializers.zeros_init(), (C,))
        G_4NC = jnp.concatenate([(Ginv_NC + Gbias)[None], Geqv_3NC])
        # G_4NC = jnp.concatenate([Ginv_NC[None], Geqv_3NC])
        Feat_NX = (G_4NC[:,:,None,:] * G_4NC[:,:,:self.params['axis_neuron'],None]).sum(0).reshape(N,-1)
        # compute fitting net output and energy 
        fit_n1 = [fitting_net(self.params['fit_widths'])(i) for i in slice_type(Feat_NX,static_args['type_index'],0)]
        energy = sum(list(map(lambda x,y:(x+y).sum(), fit_n1, self.params['Ebias'])))
        # debug info
        debug = (r_NM, R_3NM, G_4NC, Feat_NX, fit_n1)
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
