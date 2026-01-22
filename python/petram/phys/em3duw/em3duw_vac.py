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
    coeff1 = SCoeff(exprs1, ind_vars, l, g,
                    return_complex=True, scale=epsilon0/e_norm)
    coeff2 = SCoeff(exprs2, ind_vars, l, g,
                    return_complex=True, scale=1j/omega/e_norm)

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

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        self._jited_coeff = self.get_dpg_coeffs()
        
    def get_dpg_coeffs(self):
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

        cf1, cf2, cf3, cf4, cf5, cf6 = self.jited_coeff
        self.add_bf_epsmu_contribution(engine, a, cf1, cf2, cf3, cf4, cf5, cf6)


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
