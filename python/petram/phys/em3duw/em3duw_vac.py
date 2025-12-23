'''
   Vacuum region:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma


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

def eps_cf(exprs1, exprs2, ind_vars, l, g, omega, e_norm):
    #
    # (e_r e0 - i sgima/w)/e_norm
    #   cnorm is typically e0
    #
    coeff1 = SCoeff(exprs1, ind_vars, l, g, return_complex=True, scale=epsilon0/e_norm)
    coeff2 = SCoeff(exprs2, ind_vars, l, g, return_complex=True, scale=-1j/omega/e_norm)

    ans = coeff1 + coeff2

def mu_cf(exprs, ind_vars, l, g, omega, mu_norm):
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
        cnorm = self.get_root_phys().get_coeff_norm()

        e, m, s = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = eps_cf([e], [s], ind_vars, l, g, omega, epsilon0)
        coeff2 = mu_cf([m], ind_vars, l, g, omega, mu0)

        return coeff1, coeff2

    def add_dpg_integrator(self, engine, coeff, adder, integrator, sp1, sp2,
                           idx=None,  transpose=False, real=True):
        if coeff is None:
            return

        if real:
            coeff = coeff.get_real_coefficient()
        else:
            ceoff = coeff.get_imag_coefficient()

        
        if coeff[0] is None:
            return
        elif isinstance(coeff[0], mfem.Coefficient):
            coeff = self.restrict_coeff(coeff, engine, idx=idx)
        elif isinstance(coeff[0], mfem.VectorCoefficient):
            coeff = self.restrict_coeff(coeff, engine, vec=True, idx=idx)
        elif isinstance(coeff[0], mfem.MatrixCoefficient):
            coeff = self.restrict_coeff(coeff, engine, matrix=True, idx=idx)
        elif issubclass(integrator, mfem.PyBilinearFormIntegrator):
            pass
        else:
            assert False, "Unknown coefficient type: " + str(type(coeff[0]))

        args = list(coeff)
        args.extend(itg_params)

        kwargs = {}

        itg = integrator(coeff)
        itg._linked_coeff = coeff
        
        if transpose:
            itg2 = mfem.TransposeIntegrator(itg)
            itg2._link = itg
        else:
            if ir is not None:
                itg.SetIntRule(ir)
            itg2 = itg

        if real:
            adder(itg2, None, sp1, sp2)
        else:
            adder(None, itg2, sp1, sp2)
        return itg
    

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if not real:
            return 
        # e, m, s
        coeff1, coeff2 = self.get_coeffs()
        cf1 = eps_cf*(-1j*omega)
        cf2 = mu_cf*(1j*omega)
        cf3 = mu_cf*(-1j*omega)                       
        cf4 = eps_cf*(1j*omega)
        cf5 = eps_cf**2(omega**2)
        
        self.set_integrator_realimag_mode(real)

        if kfes != 0:
            return
        if real:
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))

        print(a, kfes)
        one = mfem.ConstantCoefficient(1.0)
    
        TrialSpace = {"E_space": 0
                      "H_space": 1
                      "hatE_space": 2
                      "hatE_space": 3}
        TextSpace = {"F_space": 0
                     "G_space": 1}
        
        a.StoreMatrices()  # needed for AMR

        # (E,∇ × F)
        self.add_integrator(engine, 'one', one,
                            a, AddTrialIntegrator
                            mfem.CurlCurlIntegrator,
                            transpose=True)

        
        a.AddTrialIntegrator(mfem.TransposeIntegrator(mfem.MixedCurlIntegrator(one)),
                             None,
                             TrialSpace["E_space"],
                             TestSpace["F_space"])
        # -i ω ϵ (E , G) = i (- ω ϵ E, G)
        a.AddTrialIntegrator(None,
                             mfem.TransposeIntegrator(
                                 mfem.VectorFEMassIntegrator(cf1)),
                             TrialSpace["E_space"],
                             TestSpace["G_space"])
        #  (H,∇ × G)
        a.AddTrialIntegrator(mfem.TransposeIntegrator(mfem.MixedCurlIntegrator(one)),
                             None,
                             TrialSpace["H_space"],
                             TestSpace["G_space"])
        # < n×Ĥ ,G>
        a.AddTrialIntegrator(mfem.TangentTraceIntegrator(), None,
                             TrialSpace["hatH_space"],
                             TestSpace["G_space"])
        # test integrators
        # (∇×G ,∇× δG)
        a.AddTestIntegrator(mfem.CurlCurlIntegrator(one), None,
                            TestSpace["G_space"],
                            TestSpace["G_space"])
        # (G,δG)
        a.AddTestIntegrator(mfem.VectorFEMassIntegrator(one), None,
                            TestSpace["G_space"],
                            TestSpace["G_space"])

        # i ω μ (H, F)
        a.AddTrialIntegrator(None, mfem.TransposeIntegrator(
            mfem.VectorFEMassIntegrator(cf2)),
            TrialSpace["H_space"],
            TestSpace["F_space"])
                              
        # < n×Ê,F>
        a.AddTrialIntegrator(mfem.TangentTraceIntegrator(), None,
                             TrialSpace["hatE_space"],
                             TestSpace["F_space"])

        # test integrators
        # (∇×F,∇×δF)
        a.AddTestIntegrator(mfem.CurlCurlIntegrator(one), None,
                            TestSpace["F_space"],
                            TestSpace["F_space"])
        # (F,δF)
        a.AddTestIntegrator(mfem.VectorFEMassIntegrator(one), None,
                            TestSpace["F_space"],
                            TestSpace["F_space"])
        # μ^2 ω^2 (F,δF)
        a.AddTestIntegrator(mfem.VectorFEMassIntegrator(mu2omeg2_cf), None,
                            TestSpace["F_space"],
                            TestSpace["F_space"])
        # -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)

        a.AddTestIntegrator(None, mfem.MixedVectorWeakCurlIntegrator(cf3),
                            TestSpace["F_space"],
                            TestSpace["G_space"])
        # -i ω ϵ (∇ × F, δG)
        a.AddTestIntegrator(None, mfem.MixedVectorCurlIntegrator(cf1),
                            TestSpace["F_space"],
                            TestSpace["G_space"])
        # i ω μ (∇ × G,δF)
        a.AddTestIntegrator(None, mfem.MixedVectorCurlIntegrator(cf2),
                            TestSpace["G_space"],
                            TestSpace["F_space"])
        
        # i ω ϵ (G, ∇ × δF )
        a.AddTestIntegrator(None, mfem.MixedVectorWeakCurlIntegrator(cf4),
                            TestSpace["G_space"],
                            TestSpace["F_space"])
        # ϵ^2 ω^2 (G,δG)
        a.AddTestIntegrator(mfem.VectorFEMassIntegrator(cf5.get_real_coefficient()),
                            mfem.VectorFEMassIntegrator(cf5.get_imag_coefficient()),
                            TestSpace["G_space"],
                            TestSpace["G_space"])
        

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
