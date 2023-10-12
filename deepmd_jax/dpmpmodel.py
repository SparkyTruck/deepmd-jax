import jax.numpy as jnp
from jax import vmap, value_and_grad
import flax.linen as nn
from .utils import sr, embedding_net, fitting_net, get_relative_coord, slice_type
    
class DPMPModel(nn.Module):
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
        (N, M) = r_NM.shape
        C, B = self.params['embedIJ_widths'][-1], self.params['embedMP_widths'][-1]
        # prepare normalized s(r)
        sr_nM = slice_type(sr_NM, static_args['type_index'], 0)
        srbiasnorm_NM = jnp.concatenate(list(map(lambda x,y:x/y, sr_nM, self.params['srstd'])))
        srnorm_nM = list(map(lambda x,y,z:(x-y)/z, sr_nM, self.params['srmean'], self.params['srstd']))
        srnorm_nm = [slice_type(s,static_args['ntype_index'],1) for s in srnorm_nM]
        # compute embedding net features
        embedI_nmC = [[embedding_net(self.params['embedIJ_widths'])(j[:,:,None]) for j in i] for i in srnorm_nm]
        embedI_NMC = jnp.concatenate([jnp.concatenate(i, axis=1) for i in embedI_nmC], axis=0)
        embedJ_nmC = [[embedding_net(self.params['embedIJ_widths'])(j[:,:,None]) for j in i] for i in srnorm_nm]
        embedJ_NMC = jnp.concatenate([jnp.concatenate(i, axis=1) for i in embedJ_nmC], axis=0)
        embed_nmC = [[embedding_net(self.params['embedIJ_widths'])(j[:,:,None]) for j in i] for i in srnorm_nm]
        embed_NMC = jnp.concatenate([jnp.concatenate(i, axis=1) for i in embed_nmC], axis=0)
        embed2_nmD = [[embedding_net(self.params['embed2_widths'])(j[:,:,None]) for j in i] for i in srnorm_nm]
        embed2_NMD = jnp.concatenate([jnp.concatenate(i, axis=1) for i in embed2_nmD], axis=0)
        # compute distance matrix R = (inv:srbiasnorm_NM, equiv:R_3NM)
        rsr_nM = list(map(lambda x,y:x/y, slice_type(sr_NM/r_NM,static_args['type_index'],0), self.params['xrsrstd']))
        R_3NM = self.params['e3norm']*(jnp.concatenate(rsr_nM)[None]+1e-15)*x_3NM
        # compute feature matrix G = embed @ R
        GIinv_NC = ((srbiasnorm_NM[:,:,None] * embedI_NMC).sum(1) / self.params['normalizer'])
                    # + self.param('GIbias',nn.initializers.normal(stddev=0.01),(C,)))
        GJinv_NC = ((srbiasnorm_NM[:,:,None] * embedJ_NMC).sum(1) / self.params['normalizer'])
                    # + self.param('GJbias',nn.initializers.normal(stddev=0.01),(C,)))
        GJinv_MC = (GJinv_NC[:,None] * jnp.ones((M//N,1))).reshape(M,C)
        Ginv_NC = (((srbiasnorm_NM[:,:,None] * embed_NMC).sum(1) / self.params['normalizer'])
                    + self.param('GGbias',nn.initializers.normal(stddev=0.01),(C,)))
        GIeqv_3NC = (R_3NM[...,None] * embedI_NMC).sum(2) / self.params['normalizer']
        Geqv_3NC = (R_3NM[...,None] * embed_NMC).sum(2) / self.params['normalizer']
        GJeqv_3NC = (R_3NM[...,None] * embedJ_NMC).sum(2) / self.params['normalizer']
        GJeqv_3MC = (GJeqv_3NC[:,:,None]*jnp.ones((M//N,1))).reshape(3,M,C)
        # Features: (embed2, FI=GIeqv@R3, FJ=GJeqv@R3, FII=GIeqv@GIeqv) #GIinv, GJinv, 
        FI_NMC = (GIeqv_3NC[:,:,None] * R_3NM[...,None]).sum(0)
        FJ_NMC = (GJeqv_3MC[:,None] * R_3NM[...,None]).sum(0)
        # FII_NME = ((GIeqv_3NC[:,:,None] * GIeqv_3NC[:,:,:2,None]).sum(0)).reshape(N,1,-1) * srbiasnorm_NM[:,:,None] #* jnp.ones((M,1))
        Geqv_4NC = jnp.concatenate([Ginv_NC[None], Geqv_3NC])
        Geqv_4MC = (Geqv_4NC[:,:,None] * jnp.ones((M//N,1))).reshape(4,M,-1)
        FII_NME = ((Geqv_4NC[:,:,None] * Geqv_4NC[:,:,:2,None]).sum(0)).reshape(N,1,-1) * jnp.ones((M,1))
        GJeqv_4MC = jnp.concatenate([GJinv_MC[None], GJeqv_3MC])
        # FJJ_NME = ((GJeqv_4MC[:,:,None] * GJeqv_4MC[:,:,:2,None]).sum(0)).reshape(M,-1) * jnp.ones((N,1,1))
        FJJ_NME = ((Geqv_4MC[:,:,None] * Geqv_4MC[:,:,:2,None]).sum(0)).reshape(1,M,-1) * jnp.ones((N,1,1))
        Rnorm_NM = jnp.linalg.norm(R_3NM + 1e-15, axis=0)
        R0_3NM = x_3NM / r_NM
        FIT_NMC = jnp.linalg.norm(GIeqv_3NC, axis=0)[:,None] * Rnorm_NM[:,:,None]
        FJT_NMC = jnp.linalg.norm(GJeqv_3MC, axis=0) * Rnorm_NM[:,:,None]
        # Fi_NMC = GIinv_NC[:,None] * jnp.ones((M,C))
        Fi_NMC = GIinv_NC[:,None] * sr_NM[:,:,None]
        # Fj_NMC = GJinv_MC * jnp.ones((N,1,C))
        Fj_NMC = GJinv_MC * sr_NM[:,:,None]
        FIJ_NMC = (GIeqv_3NC[:,:,None] * GJeqv_3MC[:,None]).sum(0) * sr_NM[:,:,None]
        Fij_NMC = (jnp.cross(GIeqv_3NC[:,:,None], R_3NM[...,None], axisa=0, axisb=0, axisc=0) * GJeqv_3MC[:,None]).sum(0)
        F_NMX = jnp.concatenate([embed2_NMD, FI_NMC, FJ_NMC, FII_NME, FJJ_NME], axis=-1)
        # F_NMX += self.param('Fbias',nn.initializers.normal(stddev=0.01),(F_NMX.shape[-1],))
        F_nmX = [slice_type(f,static_args['ntype_index'],1) for f in slice_type(F_NMX,static_args['type_index'],0)]
        embedMP_nmB = [[embedding_net_mp(self.params['embedMP_widths'])(j) for j in i] for i in F_nmX]
        embedMP_NMB = jnp.concatenate([jnp.concatenate(i, axis=1) for i in embedMP_nmB])
        # compute feature matrix G = E @ R and Feat = GG^T
        Ginv_NB = (srbiasnorm_NM[:,:,None] * embedMP_NMB).sum(1) / self.params['normalizer']
        Geqv_3NB = (R_3NM[...,None] * embedMP_NMB).sum(2) / self.params['normalizer']
        Gbias = self.param('Gbias',nn.initializers.normal(stddev=0.01),(B,))
        G_4NB = jnp.concatenate([(Ginv_NB + Gbias)[None], Geqv_3NB])
        Feat_NX = (G_4NB[:,:,None,:] * G_4NB[:,:,:self.params['axis_neuron'],None]).sum(0).reshape(N,-1)
        # compute fitting net output and energy 
        fit_n1 = [fitting_net(self.params['fit_widths'])(i) for i in slice_type(Feat_NX,static_args['type_index'],0)]
        energy = sum(list(map(lambda x,y:(x+y).sum(), fit_n1, self.params['Ebias'])))
        # debug info
        debug = (r_NM, embedI_NMC, embed2_NMD, Fi_NMC, Fj_NMC, FI_NMC, FJ_NMC, FII_NME, FIJ_NMC, Fij_NMC, embedMP_NMB, R_3NM, G_4NB, Feat_NX, fit_n1)
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


he_init = nn.initializers.he_normal()
vs_init = nn.initializers.variance_scaling(0.5, "fan_avg", "normal")
std_init = nn.initializers.normal(1)
dt_init = lambda k, s: 0.5 + nn.initializers.normal(0.01)(k, s)
zero_init = nn.initializers.zeros_init()
class embedding_net_mp(nn.Module): # embedding net for message passing
    widths: list
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.widths)):
            if i == 0:
                x = nn.tanh(nn.Dense(self.widths[i], kernel_init=he_init, bias_init=zero_init)(x))
            else:
                K = self.widths[i] / self.widths[i-1]
                x_prev = (x[...,None] * jnp.ones((int(K),))).reshape((x.shape[:-1]) + (-1,))
                x = nn.tanh(nn.Dense(self.widths[i], kernel_init=he_init, bias_init=std_init)(x))
                if K.is_integer():
                    if i > 1:
                        dt = self.param('dt'+str(i), dt_init, (self.widths[i],))
                        x = x * dt + x_prev 
                    else:
                        x = x + x_prev
        return x