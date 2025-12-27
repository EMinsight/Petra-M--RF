'''
   external current source

    sqrt(mu_n/eps_n) J
'''
from petram.phys.coefficient import VCoeff
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.em3duw.em3duw_base import EM3DUW_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3DUW_extJ')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('jext', VtableElement('jext', type='complex',
                               guilabel='External J',
                               suffix=('x', 'y', 'z'),
                               default=[0, 0, 0],
                               tip="volumetric external current")),)


def JextCoeff(exprs, ind_vars, l, g, fac):
    coeff = VCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff


def domain_constraints():
    return [EM3DUW_ExtJ]


class EM3DUW_ExtJ(EM3DUW_Domain):
    is_secondary_condition = True
    has_3rd_panel = False
    vt = Vtable(data)

    def has_bf_contribution(self, kfes=0):
        if kfes != 1:
            return False
        return True

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if kfes != 1:
            return
        if real:
            dprint1("Add LF contribution(real)" + str(self._sel_index))
        else:
            return

        freq, omega = self.get_root_phys().get_freq_omega()
        enorm, munorm = self.get_root_phys().get_coeff_norm()
        
        f_name = self.vt.make_value_or_expression(self)
        coeff1 = JextCoeff(f_name[0],  self.get_root_phys().ind_vars,
                           self._local_ns, self._global_ns, np.sqrt(munorm/enorm))
        
        TrialSpace, TestSpace = self.space_idx()
        self.add_dpg_integrator(engine, coeff1, a.AddDomainLFIntegrator,
                                mfem.VectorFEDomainLFIntegrator,
                                TestSpace["G_space"],)



