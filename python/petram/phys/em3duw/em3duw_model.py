#
# UltraWeak-DPG version of EM3D.
#
# This formulations solves the first order system
#
#  -i ω μ H + ∇ × E = 0,   in Ω
#   i ω ϵ E + ∇ × H = J,   in Ω
#
#  as usual exp(-iωt) is used.

from petram.phys.phys_model import Phys, PhysModule, VectorPhysCoefficient
from petram.phys.vtable import VtableElement, Vtable
import numpy as np
import traceback

from petram.mfem_config import use_parallel
from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model import Phys
from petram.phys.common.emuw_base import EMUWPhysModule
from petram.phys.em3duw.em3duw_base import EM3DUW_Bdry, EM3DUW_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3DUWModel')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


model_basename = 'EM3DUW'

# define variable for this BC.
data = (('Einit', VtableElement('Einit', type='float',
                                guilabel='E(init)',
                                suffix=('x', 'y', 'z'),
                                default=np.array([0, 0, 0]),
                                tip="initial_E",
                                chkbox=True)),)


class Einit(VectorPhysCoefficient):
    def EvalValue(self, x):
        v = super(Einit, self).EvalValue(x)
        if self.real:
            val = v.real
        else:
            val = v.imag
        return val


class EM3DUW_DefDomain(EM3DUW_Domain):
    can_delete = False
    nlterms = []

    def __init__(self, **kwargs):
        super(EM3DUW_DefDomain, self).__init__(**kwargs)

    def attribute_set(self, v):
        super(EM3DUW_DefDomain, self).attribute_set(v)
        v['sel_readonly'] = True
        v['sel_index'] = ['all']
        v['sel_index_txt'] = 'all'
        return v

    def panel1_param(self):
        return [['Common contribs.',   "",  2, {}], ]

    def get_panel1_value(self):
        return ["(E,∇ × F), (H,∇ × G),  \n(∇×G ,∇× δG), (G,δG), (∇×F,∇×δF), (F,δF) \n<n×Ĥ ,G>, < n×Ê,F>"]

    def import_panel1_value(self, v):
        pass

    def panel1_tip(self):
        return None

    def get_possible_domain(self):
        return []

    def get_possible_child(self):
        return self.parent.get_possible_domain()

    def has_bf_contribution(self, kfes):
        if kfes == 0:
            return True
        else:
            return False

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if not real:
            return
        if real:
            dprint1("Add BF contribution(complex)" + str(self._sel_index))
        else:
            return

        TrialSpace, TestSpace = self.space_idx()
        one = mfem.ConstantCoefficient(1.0)

        freq, omega = self.get_root_phys().get_freq_omega()
        enorm, munorm = self.get_root_phys().get_coeff_norm()
        c = np.sqrt(1/enorm/munorm)
        one_scaled =   omega**2/c**2
        fac = 5
        dprint1("adjoint graph norm L2 scale factor: ", fac, "x", one_scaled)
        one_scaled = mfem.ConstantCoefficient(fac*one_scaled)

        # < n×Ĥ ,G>
        # < n×Ê,F>
        a.AddTrialIntegrator(mfem.TangentTraceIntegrator(), None,
                             TrialSpace["hatH_space"],
                             TestSpace["G_space"])
        a.AddTrialIntegrator(mfem.TangentTraceIntegrator(), None,
                             TrialSpace["hatE_space"],
                             TestSpace["F_space"])
        # (E,∇ × F)
        # (H,∇ × G)
        '''
        self.add_dpg_integrator(engine,  one, a.AddTrialIntegrator,
                                mfem.MixedCurlIntegrator,
                                TrialSpace["E_space"],
                                TestSpace["F_space"],
                                transpose=True)
        self.add_dpg_integrator(engine, one, a.AddTrialIntegrator,
                                mfem.MixedCurlIntegrator,
                                TrialSpace["H_space"],
                                TestSpace["G_space"],
                                transpose=True)
        '''
        a.AddTrialIntegrator(mfem.TransposeIntegrator(mfem.MixedCurlIntegrator(one)),
                             None,
                             TrialSpace["E_space"],
                             TestSpace["F_space"])
        a.AddTrialIntegrator(mfem.TransposeIntegrator(mfem.MixedCurlIntegrator(one)),
                             None,
                             TrialSpace["H_space"],
                             TestSpace["G_space"])

        # test integrators
        # (∇×G ,∇× δG)
        # (G,δG)
        # (∇×F,∇×δF)
        # (F,δF)
        '''
        self.add_dpg_integrator(engine, one, a.AddTestIntegrator,
                                mfem.CurlCurlIntegrator,
                                TestSpace["G_space"],
                                TestSpace["G_space"],)
        self.add_dpg_integrator(engine, one, a.AddTestIntegrator,
                                mfem.VectorFEMassIntegrator,
                                TestSpace["G_space"],
                                TestSpace["G_space"],)
        self.add_dpg_integrator(engine, one, a.AddTestIntegrator,
                                mfem.CurlCurlIntegrator,
                                TestSpace["F_space"],
                                TestSpace["F_space"],)
        self.add_dpg_integrator(engine, one, a.AddTestIntegrator,
                                mfem.VectorFEMassIntegrator,
                                TestSpace["F_space"],
                                TestSpace["F_space"],)
        '''
        a.AddTestIntegrator(mfem.CurlCurlIntegrator(one), None,
                            TestSpace["G_space"],
                            TestSpace["G_space"])
        a.AddTestIntegrator(mfem.VectorFEMassIntegrator(one_scaled), None,
                            TestSpace["G_space"],
                            TestSpace["G_space"])
        a.AddTestIntegrator(mfem.CurlCurlIntegrator(one), None,
                            TestSpace["F_space"],
                            TestSpace["F_space"])
        a.AddTestIntegrator(mfem.VectorFEMassIntegrator(one_scaled), None,
                            TestSpace["F_space"],
                            TestSpace["F_space"])


data2 = (('label1', VtableElement(None,
                                  guilabel='Default Bdry (PMC)',
                                  default="Ht = 0",
                                  tip="this is a natural BC")),)


class EM3DUW_DefBdry(EM3DUW_Bdry):
    can_delete = False
    is_essential = False
    nlterms = []
    vt = Vtable(data2)

    def __init__(self, **kwargs):
        super(EM3DUW_DefBdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM3DUW_DefBdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v

    def get_possible_bdry(self):
        return []


class EM3DUW_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(EM3DUW_DefPair, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM3DUW_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_possible_pair(self):
        return []


class EM3DUW(EMUWPhysModule):
    geom_dim = 3

    def __init__(self, **kwargs):
        super(EM3DUW, self).__init__(**kwargs)
        self['Domain'] = EM3DUW_DefDomain()
        self['Boundary'] = EM3DUW_DefBdry()
        self['Pair'] = EM3DUW_DefPair()

        self._use_amr = False

    @property
    def dep_vars(self):
        ret = ['E', 'B', 'Et', 'Bt']
        return [x + self.dep_vars_suffix for x in ret]

    @property
    def dep_vars0(self):
        ret = ['E', 'B', 'Et', 'Bt']
        return [x + self.dep_vars_suffix for x in ret]

    @property
    def dep_vars_base(self):
        ret = ['E', 'B', 'Et', 'Bt']
        return ret

    @property
    def vdim(self):
        return [3, 3, 1, 1]

    @vdim.setter
    def vdim(self, val):
        pass

    def fec_order(self, idx):
        self.vt_order.preprocess_params(self)
        if idx == 0 or idx == 1:
            return self.order-1
        return self.order

    def get_fec_type(self, idx):
        values = ['L2v', 'L2v', 'ND_Trace', 'ND_Trace']
        return values[idx]

    def get_fec(self):
        v = self.dep_vars
        return ((v[0], 'L2_FECollection'),
                (v[1], 'L2_FECollection'),
                (v[2], 'ND_Trace_FECollection'),
                (v[3], 'ND_Trace_FECollection'),)

    def attribute_set(self, v):
        v = super(EM3DUW, self).attribute_set(v)
        v["element"] = ', '.join(
            ('L2_FECollection', 'L2_FECollection', 'ND_Trace_FECollection', 'ND_Trace_FECollection'))
        v["freq_txt"] = "1.0e9"
        v["ndim"] = 3
        v["ind_vars"] = 'x, y, z'
        v["dep_vars_suffix"] = ''
        v["static_cond"] = False

        self._use_amr = False        
        return v

    def panel1_param(self):
        panels = super(EM3DUW, self).panel1_param()
        a, b = self.get_var_suffix_var_name_panel()
        panels.extend([  # self.make_param_panel('freq',  self.freq_txt),
            ["independent vars.", self.ind_vars, 0, {}],
            a,
            ["dep. vars.", ','.join(self.dep_vars), 2, {}],
            ["ns vars.", '', 2, {}],
            ["static cond.", True, 3, {"text": ""}],])

        return panels

    def get_panel1_value(self):
        names = ', '.join(self.dep_vars)
        names2 = ', '.join(list(self.get_default_ns()))
        val = super(EM3DUW, self).get_panel1_value()
        val.extend([  # self.freq_txt,
            self.ind_vars, self.dep_vars_suffix,
            names, names2, self.static_cond])
        return val

    def import_panel1_value(self, v):
        v = super(EM3DUW, self).import_panel1_value(v)
        # self.freq_txt = str(v[0])
        self.ind_vars = str(v[0])
        self.dep_vars_suffix = str(v[1])
        self.static_cond = v[-1]

    def get_possible_bdry(self):
        if EM3DUW._possible_constraints is None:
            self._set_possible_constraints('em3duw')
        bdrs = super(EM3DUW, self).get_possible_bdry()
        return EM3DUW._possible_constraints['bdry'] + bdrs

    def get_possible_domain(self):
        if EM3DUW._possible_constraints is None:
            self._set_possible_constraints('em3duw')

        doms = super(EM3DUW, self).get_possible_domain()
        return EM3DUW._possible_constraints['domain'] + doms

    def get_possible_edge(self):
        return []

    def get_possible_pair(self):
        if EM3DUW._possible_constraints is None:
            self._set_possible_constraints()
        return EM3DUW._possible_constraints['pair']

    def add_variables(self, v, name, solr, soli=None):
        from petram.helper.variables import add_component_expression as addc_expression
        from petram.helper.variables import (add_coordinates,
                                             add_components,
                                             add_expression,
                                             add_surf_normals,
                                             GFScalarVariable,
                                             GFVectorVariable,
                                             PlaceholderVariable)

        from petram.helper.eval_deriv import eval_curl

        v = super(EM3DUW, self).add_variables(v, name, solr, soli)

        freq, omega = self.get_freq_omega()

        ind_vars = [x.strip()
                    for x in self.ind_vars.split(',') if x.strip() != '']

        suffix = self.dep_vars_suffix

        from petram.helper.variables import TestVariable
        # v['debug_test'] =  TestVariable()

        add_coordinates(v, ind_vars)
        add_surf_normals(v, ind_vars)

        from petram.phys.phys_const import mu0, epsilon0, q0

        def add_E_B(name):
            for k, nn in enumerate(('x', 'y', 'z')):
                name1 = name + nn
                if solr is not None:
                    v[name1] = GFScalarVariable(solr, soli, comp=k+1)
                else:
                    v[name1] = PlaceholderVariable(name1)
            if solr is not None:
                v[name] = GFVectorVariable(solr, soli)
            else:
                v[name] = PlaceholderVariable(name)

        if name.startswith('Bt'):
            add_E_B(name)
        elif name.startswith('Et'):
            add_E_B(name)
        elif name.startswith('B'):
            add_E_B(name)
        elif name.startswith('E'):
            add_E_B(name)
            add_expression(v, 'normE', suffix, ind_vars,
                           '(conj(Ex)*Ex + conj(Ey)*Ey +conj(Ez)*Ez)**(0.5)',
                           ['E'])

            add_expression(v, 'normB', suffix, ind_vars,
                           '(conj(Bx)*Bx + conj(By)*By + conj(Bz)*Bz)**(0.5)',
                           ['B'])

            # Poynting Flux
            addc_expression(v, 'Poy', suffix, ind_vars,
                            '(conj(Ey)*Bz - conj(Ez)*By)/mu0',
                            ['B', 'E'], 0)
            addc_expression(v, 'Poy', suffix, ind_vars,
                            '(conj(Ez)*Bx - conj(Ex)*Bz)/mu0',
                            ['B', 'E'], 1)
            addc_expression(v, 'Poy', suffix, ind_vars,
                            '(conj(Ex)*By - conj(Ey)*Bx)/mu0',
                            ['B', 'E'], 2)

            # e = - epsion * w^2 - i * sigma * w
            # Jd : displacement current  = -i omega* e0 er E
            addc_expression(v, 'Jd', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E))*freq*2*pi*e0)[0]',
                            ['epsilonr', 'E', 'freq'],  0)
            addc_expression(v, 'Jd', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E))*freq*2*pi*e0)[1]',
                            ['epsilonr', 'E', 'freq'], 1)
            addc_expression(v, 'Jd', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E))*freq*2*pi*e0)[2]',
                            ['epsilonr', 'E', 'freq'], 2)
            # Ji : induced current = sigma *E
            addc_expression(v, 'Ji', suffix, ind_vars,
                            '(dot(sigma, E))[0]',
                            ['sigma', 'E'], 0)
            addc_expression(v, 'Ji', suffix, ind_vars,
                            '(dot(sigma, E))[1]',
                            ['sigma', 'E'], 1)
            addc_expression(v, 'Ji', suffix, ind_vars,
                            '(dot(sigma, E))[2]',
                            ['sigma', 'E'], 2)
            # Jp : polarization current (Jp = -i omega* e0 (er - 1) E
            addc_expression(v, 'Jp', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E) - E)*freq*2*pi*e0)[0]',
                            ['epsilonr', 'E', 'freq'], 0)
            addc_expression(v, 'Jp', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E) - E)*freq*2*pi*e0)[1]',
                            ['epsilonr', 'E', 'freq'], 1)
            addc_expression(v, 'Jp', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E) - E)*freq*2*pi*e0)[2]',
                            ['epsilonr', 'E', 'freq'], 2)

            # Js : surface current (n x B / mu)
            addc_expression(v, 'Js', suffix, ind_vars,
                            '(cross([nx, ny, nz], inv(mur).dot(B))/mu0)[0]',
                            ['nx', 'ny', 'nz', 'B', 'mur', 'mu0'], 0)
            addc_expression(v, 'Js', suffix, ind_vars,
                            '(cross([nx, ny, nz], inv(mur).dot(B))/mu0)[1]',
                            ['nx', 'ny', 'nz', 'B', 'mur', 'mu0'], 1)
            addc_expression(v, 'Js', suffix, ind_vars,
                            '(cross([nx, ny, nz], inv(mur).dot(B))/mu0)[2]',
                            ['nx', 'ny', 'nz', 'B', 'mur', 'mu0'], 2)

        return v

    def has_diagform(self, kfes1):
        if kfes1 < 4:
            return True
        return False

    @property
    def test_order(self):
        order = self.order
        delta_order = 1
        torder = order + delta_order
        return torder

    def set_dpg_amr(self):
        self._use_amr = True

    def get_diagform_callable(self, fes_arr):
        def callable(fes_arr=fes_arr):
            if use_parallel:
                dpg_form = mfem.dpg.ParComplexDPGWeakForm

            else:
                dpg_form = mfem.dpg.ComplexDPGWeakForm

            order = self.order
            test_order = self.test_order

            dim = 3

            F_fec = mfem.ND_FECollection(test_order, dim)
            G_fec = mfem.ND_FECollection(test_order, dim)

            test_fec = (F_fec, G_fec)
            trial_fes = fes_arr

            form = dpg_form(trial_fes, test_fec)
            form._test_fec = test_fec
            form._trial_fes = trial_fes

            if self.static_cond:
                form.EnableStaticCondensation()
            if self._use_amr:
                form.StoreMatrices()  # needed for AMR

            return form

        def callable_none(fes_arr=fes_arr):
            # we use ComplexDPGWeakform.
            # imaginary part is not filled separately
            return None

        return callable, callable_none

    def diag_formlinearsystem(self, ess_tdof_list, a,  x):
        from petram.debug import debug_dpg_essential

        if debug_dpg_essential:
            ess_tdof_list = mfem.intArray()
        return super(EM3DUW, self).diag_formlinearsystem(ess_tdof_list, a,  x)

    def split_AhXB_complex(self, Ah, X, B):
        from petram.phys.phys_diagform_utils import split_AhXB_complex_mode1

        mblk, xblk, bblk = split_AhXB_complex_mode1(Ah, X, B)

        if self.static_cond:
            # put None to form 4x4 return value
            mblk = ([None, None, None, None,
                     None, None, None, None,
                     None, None, mblk[0][0], mblk[0][1],
                     None, None, mblk[0][2], mblk[0][3],],
                    [None, None, None, None,
                     None, None, None, None,
                     None, None, mblk[1][0], mblk[1][1],
                     None, None, mblk[1][2], mblk[1][3],],)
            xblk = ([None, None, xblk[0][0], xblk[0][1]],
                    [None, None, xblk[1][0], xblk[1][1]])
            bblk = ([None, None, bblk[0][0], bblk[0][1]],
                    [None, None, bblk[1][0], bblk[1][1]])
        return mblk, xblk, bblk
