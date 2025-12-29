'''
   "Vacuum" region:
      epsilon_r, mu_r, sigma are sclar but user defined.

'''
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_const import mu0, epsilon0
from petram.phys.coefficient import SCoeff
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.phys_model import PhysCoefficient, PhysConstant
from petram.phys.em3duw.em3duw_base import EM3DUW_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3DUW_Vac')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('epsilonr', VtableElement('epsilonr', type='complex',
                                   guilabel='epsilonr',
                                   default=1.0,
                                   tip="relative permittivity")),
        ('mur', VtableElement('mur', type='complex',
                              guilabel='mur',
                              default=1.0,
                              tip="relative permeability")),
        ('sigma', VtableElement('sigma', type='complex',
                                guilabel='sigma',
                                default=0.0,
                                tip="contuctivity")),)

def EpsSigmaCoeff(exprs1, exprs2, ind_vars, l, g, omega, e_norm):
    #
    # (e_r e0 + i sgima/w)/e_norm
    #   cnorm is typically e0
    #
    coeff1 = SCoeff(exprs1, ind_vars, l, g, return_complex=True, scale=epsilon0/e_norm)
    coeff2 = SCoeff(exprs2, ind_vars, l, g, return_complex=True, scale=1j/omega/e_norm)

    return coeff1 + coeff2

def MuCoeff(exprs, ind_vars, l, g, omega, mu_norm):
    # mu_r *mu_0/mu_norm
    fac = mu0/mu_norm
    coeff = SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff



def domain_constraints():
    return [EM3DUW_Vac]


class EM3DUW_Vac(EM3DUW_Domain):
    vt = Vtable(data)
    # nlterms = ['epsilonr']

    def has_bf_contribution(self, kfes):
        if kfes == 0:
            return True
        else:
            return False

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        enorm, munorm = self.get_root_phys().get_coeff_norm()

        dprint1("coefficient normalization, e_n, mu_n, c_n",  enorm, munorm,
                np.sqrt(1/enorm/munorm))

        e, m, s = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        eps_cf = EpsSigmaCoeff([e], [s], ind_vars, l, g, omega, enorm)
        mu_cf = MuCoeff([m], ind_vars, l, g, omega, munorm)

        c = np.sqrt(1/enorm/munorm)

        cf1 = eps_cf*(1j*omega/c)
        cf2 = mu_cf*(-1j*omega/c)

        cf3 = eps_cf.conj()*(-1j*omega/c)
        cf4 = mu_cf.conj()*(1j*omega/c)

        cf5 = eps_cf*eps_cf.conj()*(omega**2/c**2)
        cf6 = mu_cf*mu_cf.conj()*(omega**2/c**2)

        return cf1, cf2, cf3, cf4, cf5, cf6

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if kfes != 0:
            return
        if real:
            dprint1("Add BF contribution(complex)" + str(self._sel_index))
        else:
            return

        cf1, cf2, cf3, cf4, cf5, cf6 = self.get_coeffs()
        TrialSpace, TestSpace = self.space_idx()
        #a.StoreMatrices()  # needed for AMR

        # xxxxxx -i ω ϵ (E , G) = i (- ω ϵ E, G)
        # i ω ϵ (E , G)
        self.add_dpg_integrator(engine, cf1, a.AddTrialIntegrator,
                                mfem.VectorFEMassIntegrator,
                                TrialSpace["E_space"],
                                TestSpace["G_space"],
                                transpose=True,)
        #a.AddTrialIntegrator(None,
        #                     mfem.TransposeIntegrator(
        #                         mfem.VectorFEMassIntegrator(cf1)),
        #                     TrialSpace["E_space"],
        #                     TestSpace["G_space"])

        # xxxxx  i ω μ (H, F)
        # -i ω μ (H, F)
        self.add_dpg_integrator(engine, cf2, a.AddTrialIntegrator,
                                mfem.VectorFEMassIntegrator,
                                TrialSpace["H_space"],
                                TestSpace["F_space"],
                                transpose=True,)
        #a.AddTrialIntegrator(None, mfem.TransposeIntegrator(
        #    mfem.VectorFEMassIntegrator(cf2)),
        #    TrialSpace["H_space"],
        #    TestSpace["F_space"])

        # (1,1)
        # |μ|^2 ω^2 (F,δF)
        self.add_dpg_integrator(engine, cf6, a.AddTestIntegrator,
                                mfem.VectorFEMassIntegrator,
                                TestSpace["F_space"],
                                TestSpace["F_space"],)
        #a.AddTestIntegrator(mfem.VectorFEMassIntegrator(mu2omeg2_cf), None,
        #                    TestSpace["F_space"],
        #                    TestSpace["F_space"])

        # (2, 1)
        # -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
        # (i ω μ* F,∇ × δG)
        self.add_dpg_integrator(engine, cf4, a.AddTestIntegrator,
                                mfem.MixedVectorWeakCurlIntegrator,
                                TestSpace["F_space"],
                                TestSpace["G_space"],)
        #a.AddTestIntegrator(None, mfem.MixedVectorWeakCurlIntegrator(cf3),
        #                    TestSpace["F_space"],
        #                    TestSpace["G_space"])

        # xxxx -i ω ϵ (∇ × F, δG)
        # (i ω ϵ ∇ × F, δG)
        self.add_dpg_integrator(engine, cf1, a.AddTestIntegrator,
                                mfem.MixedVectorCurlIntegrator,
                                TestSpace["F_space"],
                                TestSpace["G_space"],)
        #a.AddTestIntegrator(None, mfem.MixedVectorCurlIntegrator(cf1),
        #                    TestSpace["F_space"],
        #                    TestSpace["G_space"])
        # (2, 1)
        # *****i ω μ (∇ × G,δF)
        # (-i ω μ ∇ × G,δF)
        self.add_dpg_integrator(engine, cf2, a.AddTestIntegrator,
                                mfem.MixedVectorCurlIntegrator,
                                TestSpace["G_space"],
                                TestSpace["F_space"],)
        #a.AddTestIntegrator(None, mfem.MixedVectorCurlIntegrator(cf2),
        #                    TestSpace["G_space"],
        #                    TestSpace["F_space"])

        # xxxxx i ω ϵ (G, ∇ × δF )
        # (-i ω ϵ* (G, ∇ × δF )
        self.add_dpg_integrator(engine, cf3, a.AddTestIntegrator,
                                mfem.MixedVectorWeakCurlIntegrator,
                                TestSpace["G_space"],
                                TestSpace["F_space"],)
        #a.AddTestIntegrator(None, mfem.MixedVectorWeakCurlIntegrator(cf4),
        #                    TestSpace["G_space"],
        #                    TestSpace["F_space"])
        # (2, 2)
        # |ϵ|^2 ω^2 (G,δG)
        self.add_dpg_integrator(engine, cf5, a.AddTestIntegrator,
                                mfem.VectorFEMassIntegrator,
                                TestSpace["G_space"],
                                TestSpace["G_space"],)
        #a.AddTestIntegrator(mfem.VectorFEMassIntegrator(cf5), None,
        #                    TestSpace["G_space"],
        #                    TestSpace["G_space"])


    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_constant

        if len(self._sel_index) == 0:
            return

        e, m, s = self.vt.make_value_or_expression(self)

        self.do_add_scalar_expr(v, suffix, ind_vars,
                                'sepsilonr', e, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'smur', m, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'ssigma', s, add_diag=3)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')
