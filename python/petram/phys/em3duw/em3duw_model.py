
'''
UltraWeak-DPG version of EM3D
'''
from petram.phys.phys_model import Phys, PhysModule, VectorPhysCoefficient
from petram.phys.vtable import VtableElement, Vtable
import numpy as np
import traceback

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model import Phys
from petram.phys.common.em_base import EMPhysModule
from petram.phys.em3duw.em3duw_base import EM3DUW_Bdry
from petram.phys.em3duw.em3duw_vac import EM3DUW_Vac

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3DUWModel')

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


class EM3DUW_DefDomain(EM3DUW_Vac):
    can_delete = False
    nlterms = []

    def __init__(self, **kwargs):
        super(EM3DUW_DefDomain, self).__init__(**kwargs)

    def panel1_param(self):
        return [['Default Domain (Vac)',   "eps_r=1, mu_r=1, sigma=0",  2, {}], ]

    def get_panel1_value(self):
        return ["eps_r=1, mu_r=1, sigma=0", ]

    def import_panel1_value(self, v):
        pass

    def panel1_tip(self):
        return None

    def get_possible_domain(self):
        return []

    def get_possible_child(self):
        return self.parent.get_possible_domain()


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


class EM3DUW(EMPhysModule):
    der_var_base = ['Bx', 'By', 'Bz']
    der_var_vec = ['B']
    geom_dim = 3

    def __init__(self, **kwargs):
        super(EM3DUW, self).__init__(**kwargs)
        self['Domain'] = EM3DUW_DefDomain()
        self['Boundary'] = EM3DUW_DefBdry()
        self['Pair'] = EM3DUW_DefPair()

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
        return v

    def panel1_param(self):
        panels = super(EM3DUW, self).panel1_param()
        a, b = self.get_var_suffix_var_name_panel()
        panels.extend([  # self.make_param_panel('freq',  self.freq_txt),
            ["independent vars.", self.ind_vars, 0, {}],
            a,
            ["dep. vars.", ','.join(self.dep_vars), 2, {}],
            ["ns vars.", ','.join(EM3DUW.der_var_base), 2, {}], ])

        return panels

    def get_panel1_value(self):
        names = ', '.join(self.dep_vars)
        names2 = ', '.join(list(self.get_default_ns()))
        val = super(EM3DUW, self).get_panel1_value()
        val.extend([  # self.freq_txt,
            self.ind_vars, self.dep_vars_suffix,
            names, names2, ])
        return val

    def import_panel1_value(self, v):
        v = super(EM3DUW, self).import_panel1_value(v)
        # self.freq_txt = str(v[0])
        self.ind_vars = str(v[0])
        self.dep_vars_suffix = str(v[1])

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
        from petram.helper.variables import add_coordinates
        from petram.helper.variables import add_scalar
        from petram.helper.variables import add_components
        from petram.helper.variables import add_component_expression as addc_expression
        from petram.helper.variables import add_expression
        from petram.helper.variables import add_surf_normals
        from petram.helper.variables import add_constant

        from petram.helper.eval_deriv import eval_curl

        v = super(EM3DUW, self).add_variables(v, name, solr, soli)

        freq, omega = self.get_freq_omega()

        def evalB(gfr, gfi=None):
            gfr, gfi, extra = eval_curl(gfr, gfi)
            gfi /= omega   # real B
            gfr /= -omega  # imag B
            # flipping gfi and gfr so that it returns
            # -i * (-gfr + i gfi) = gfi + i gfr
            return gfi, gfr, extra

        ind_vars = [x.strip()
                    for x in self.ind_vars.split(',') if x.strip() != '']

        suffix = self.dep_vars_suffix

        from petram.helper.variables import TestVariable
        # v['debug_test'] =  TestVariable()

        add_coordinates(v, ind_vars)
        add_surf_normals(v, ind_vars)

        from petram.phys.phys_const import mu0, epsilon0, q0

        if name.startswith('E'):

            add_components(v, 'E', suffix, ind_vars, solr, soli)
            add_components(v, 'B', suffix, ind_vars, solr, soli,
                           deriv=evalB)
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

        elif name.startswith('psi'):
            add_scalar(v, 'psi', suffix, ind_vars, solr, soli)


        return v

    def has_diag_form(self, kfes1):
        if kfes1 < 4:
            return True
        return False

    def get_diagform_callable(self, fes_arr):
        def callable(fes_arr=fes_arr):
            
            from petram.mfem_config import use_parallel
            if use_parallel:
                import mfem.par as mfem
                dpg_form = mfem.dpg.ParComplexDPGWeakForm                
                
            else:
                import mfem.ser as mfem                
                dpg_form = mfem.dpg.ComplexDPGWeakForm
                
            order=self.order
            
            delta_order = 1
            test_order = order + delta_order
            dim = 3
       
            F_fec = mfem.ND_FECollection(test_order, dim)
            G_fec = mfem.ND_FECollection(test_order, dim)

            test_fec = (F_fec, G_fec)
            trial_fes = fes_arr

            form = dpg_form(trial_fes,test_fec)
            form._test_fec = test_fec
            form._trial_fes = trial_fes

            return form
        
        return callable
