'''

   3D port boundary condition


'''
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.common.rf_port_geometry import (analyze_geom_norm_ref,
                                                 analyze_rect_geom,
                                                 analyze_coax_geom,
                                                 analyze_circular_geom)
from petram.phys.em3duw.em3duw_base import EM3DUW_Bdry, EM3DUW_Domain
from petram.model import Bdry
from petram.mfem_config import use_parallel
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3DUW_Port')


if use_parallel:
    import mfem.par as mfem
    from mfem.common.mpi_debug import nicePrint
    from petram.helper.mpi_recipes import *
else:
    import mfem.ser as mfem
    nicePrint = dprint1


data = (('inc_amp', VtableElement('inc_amp', type='complex',
                                  guilabel='incoming amp',
                                  default=1.0,
                                  tip="amplitude of incoming wave")),
        ('inc_phase', VtableElement('inc_phase', type='float',
                                    guilabel='incoming phase (deg)',
                                    default=0.0,
                                    tip="phase of incoming wave")),
        ('epsilonr', VtableElement('epsilonr', type='complex',
                                   guilabel='epsilonr',
                                   default=1.0,
                                   tip="relative permittivity")),
        ('mur', VtableElement('mur', type='complex',
                              guilabel='mur',
                              default=1.0,
                              tip="relative permeability")),)


def bdry_constraints():
    return [EM3DUW_Port]


class EM3DUW_Port(EM3DUW_Bdry):
    extra_diagnostic_print = True
    vt = Vtable(data)

    def __init__(self, mode='TE', mn='0,1', inc_amp='1',
                 inc_phase='0', port_idx=1):
        super(EM3DUW_Port, self).__init__(mode=mode,
                                          mn=mn,
                                          inc_amp=inc_amp,
                                          inc_phase=inc_phase,
                                          port_idx=port_idx)

    def extra_DoF_name(self):
        return self.get_root_phys().dep_vars[0] + "_port_" + str(self.port_idx)

    def get_probes(self):
        return [self.get_root_phys().dep_vars[0] + "_port_" + str(self.port_idx)]

    def attribute_set(self, v):
        super(EM3DUW_Port, self).attribute_set(v)
        v['port_idx'] = 1
        v['mode'] = 'TE'
        v['mn'] = [1, 0]
        v['inc_amp_txt'] = '1.0'
        v['inc_phase_txt'] = '0.0'
        v['inc_amp'] = 1.0
        v['inc_phase'] = 0.0
        v['epsilonr'] = 1.0
        v['mur'] = 1.0
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['isTimeDependent_RHS'] = True
        v['ref_pt'] = '1'
        return v

    def panel1_param(self):
        return ([["port id", str(self.port_idx), 0, {}],
                 ["mode", self.mode, 4, {"readonly": True,
                                         "choices": ["TE", "TEM", "Coax(TEM)", "Circular(TE)"]}],
                 ["m/n", ','.join(str(x) for x in self.mn), 0, {}],
                 ["ref. pt.", "", 0, {}],] +
                self.vt.panel_param(self))

    def get_panel1_value(self):
        return ([str(self.port_idx),
                 self.mode, ','.join(str(x) for x in self.mn),
                 self.ref_pt] +
                self.vt.get_panel_value(self))

    def import_panel1_value(self, v):
        self.port_idx = v[0]
        self.mode = v[1]
        self.mn = [int(x) for x in v[2].split(',')]
        self.ref_pt = v[3]
        self.vt.import_panel_value(self, v[4:])


    def panel4_param(self):
        ll = super(EM3DUW_Port, self).panel4_param()
        ll.append(['Varying (in time/for loop) RHS', False, 3, {"text": ""}])
        return ll

    def panel4_tip(self):
        return None

    def import_panel4_value(self, value):
        super(EM3DUW_Port, self).import_panel4_value(value[:-1])
        self.isTimeDependent_RHS = value[-1]

    def get_panel4_value(self):
        value = super(EM3DUW_Port, self).get_panel4_value()
        value.append(self.isTimeDependent_RHS)
        return value

    def verify_setting(self):
        if self.isTimeDependent_RHS:
            flag = True
        else:
            flag = False
        return flag, 'Varying RHS is not set', 'This potntially causes an error with PortScan. Set it Time/NL Dep. panel '

    def update_param(self):
        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)


    def preprocess_params(self, engine):
        # find normal (outward) vector...
        mesh = engine.get_emesh(mm=self)

        #fespace = engine.fespaces[self.get_root_phys().dep_vars[0]]

        nbe = mesh.GetNBE()
        ibe = np.array([i for i in range(nbe)
                        if mesh.GetBdrElement(i).GetAttribute() ==
                        self._sel_index[0]])
        
        norm, rptx = analyze_geom_norm_ref(mesh, ibe, self.ref_pt)
        self.norm = norm
        
        dprint1("Normal Vector " + str(self.norm))
        dprint1("Ref. Point" + str(rptx))

        if str(self.mode).upper().strip() in ['TE', 'TM', 'TEM']:
            geom_data, vv = analyze_rect_geom(self, mesh, ibe, norm)
            self.a = geom_data["a"]
            self.a_vec = geom_data["a_vec"]
            self.b = geom_data["b"]
            self.b_vec = geom_data["b_vec"]            
            self.c = geom_data["c"]            
            dprint1("Long Edge  " + self.a.__repr__())
            dprint1("Long Edge Vec." + list(self.a_vec).__repr__())
            dprint1("Short Edge  " + self.b.__repr__())
            dprint1("Short Edge Vec." + list(self.b_vec).__repr__())
            
        elif self.mode == 'Coax(TEM)':
            geom_data, vv = analyze_coax_geom(portmodel, mesh, ibe, norm, ref_ptx)
            self.a = geom_data["a"]
            self.b = geom_data["b"]
            self.ctr = geom_data["ctr"]            
            
        elif self.mode == 'Circular(TE)':
            geom_data, vv = analyze_circular_geom(portmodel, mesh, ibe, norm, ref_ptx)
            self.a = geom_data["a"]
            self.ctr = geom_data["ctr"]
        else:
            assert False, "unknown mode"
            
        C_Et, C_jwHt = self.get_coeff_cls()

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)
        dprint1("E field pattern", eps, mur)
        Et = C_Et(3, self, real=True, eps=eps, mur=mur)
        for p in vv:
            dprint1(p.__repr__() + ' : ' + Et.EvalValue(p).__repr__())
            
        dprint1("H field pattern")
        cnorm = self.get_cnorm()
        Ht = C_jwHt(3, self, real=False, eps=eps, mur=mur, cnorm=cnorm)
        for p in vv:
            dprint1(p.__repr__() + ' : ' + Ht.EvalValue(p).__repr__())

    def get_coeff_cls(self):
        from petram.phys.common.rf_portmode import get_portmode_coeff_cls
        
        return get_portmode_coeff_cls(self.mode)

    def get_cnorm(self):
        #  1/j omega * c /mu0 (jwH -> cB)
        from petram.phys.phys_const import mu0
        
        root_phys = self.get_root_phys()        
        freq, omega = root_phys.get_freq_omega()        
        enorm, munorm = root_phys.get_coeff_norm()
        cnorm = 1/np.sqrt(enorm*munorm)/(1j*omega)*mu0
        return cnorm

    def has_extra_DoF(self, kfes):
        if kfes in (2, 3):
            return True
        return False

    def get_extra_NDoF(self):
        return 1

    def is_extra_RHSonly(self):
        return True

    def postprocess_extra(self, sol, flag, sol_extra):
        name = self.name() + '_' + str(self.port_idx)
        sol_extra[name] = sol.toarray()

    def check_extra_update(self, mode):
        '''
        mode = 'B' or 'M'
        'M' return True, if M needs to be updated
        'B' return True, if B needs to be updated
        '''
        if self._update_flag:
            if mode == 'B':
                return self.isTimeDependent_RHS
            if mode == 'M':
                return self.isTimeDependent
        return False

    def add_extra_contribution(self, engine, kfes=-1, **kwargs):
        if kfes not in (3,):
            return

        dprint1("Add Extra contribution" + str(self._sel_index))
        from mfem.common.chypre import LF2PyVec, PyVec2PyMat, Array2PyVec, IdentityPyMat

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)

        C_Et, C_jwHt = self.get_coeff_cls()

        if kfes == 2:
            fes_Et = engine.get_fes(self.get_root_phys(), 2)
            lfEr = engine.new_lf(fes_Et)
            Etr = C_Et(3, self, real=True, eps=eps, mur=mur)
            Etr = self.restrict_coeff(Etr, engine, vec=True)
            intg = mfem.VectorFEDomainLFIntegrator(Etr)
            lfEr.AddBoundaryIntegrator(intg)
            lfEr.Assemble()

            lfEi = engine.new_lf(fes_Et)
            Eti = C_Et(3, self, real=False, eps=eps, mur=mur)
            Eti = self.restrict_coeff(Eti, engine, vec=True)
            intg = mfem.VectorFEDomainLFIntegrator(Eti)
            lfEi.AddBoundaryIntegrator(intg)
            lfEi.Assemble()

            xr = engine.new_gf(fes_Et)
            xr.Assign(0.0)
            arr = self.get_restriction_array(engine)
            xr.ProjectBdrCoefficientTangent(Etr, arr)
            xi = engine.new_gf(fes_Et)
            xi.Assign(0.0)
            arr = self.get_restriction_array(engine)
            xi.ProjectBdrCoefficientTangent(Eti, arr)

            et = engine.x2X(xr).GetDataArray() + 1j * \
                engine.x2X(xi).GetDataArray()
            vec = engine.b2B(lfEr).GetDataArray() + 1j * \
                engine.b2B(lfEi).GetDataArray()

            weight_E = np.sum(et.dot(vec))
            print("weight", weight_E)
            v2 = LF2PyVec(lfEr, lfEi, horizontal=True)
            v2 *= 1. / weight_E
        else:

            fes_Ht = engine.get_fes(self.get_root_phys(), 3)
            cnorm = self.get_cnorm()
                 
            lfHr = engine.new_lf(fes_Ht)
            Htr = C_jwHt(3, self, real=True, eps=eps, mur=mur, cnorm=cnorm)
            Htr = self.restrict_coeff(Htr, engine, vec=True)
            intg = mfem.VectorFEDomainLFIntegrator(Htr)
            lfHr.AddBoundaryIntegrator(intg)
            lfHr.Assemble()
            #lfHrB = engine.b2B(lfHr)

            lfHi = engine.new_lf(fes_Ht)
            Hti = C_jwHt(3, self, real=False, eps=eps, mur=mur, cnorm=cnorm)
            Hti = self.restrict_coeff(Hti, engine, vec=True)
            intg = mfem.VectorFEDomainLFIntegrator(Hti)
            lfHi.AddBoundaryIntegrator(intg)
            lfHi.Assemble()
            #lfHiB = engine.b2B(lfHi)
            
            xr = engine.new_gf(fes_Ht)
            xr.Assign(0.0)
            arr = self.get_restriction_array(engine)
            xr.ProjectBdrCoefficientTangent(Htr, arr)
            #Xr = engine.x2X(xr)
            
            xi = engine.new_gf(fes_Ht)
            xi.Assign(0.0)
            arr = self.get_restriction_array(engine)
            xi.ProjectBdrCoefficientTangent(Hti, arr)
            #Xi = engine.x2X(xi)            

            #et = Xr.GetDataArray() + 1j * Xi.GetDataArray()
            et = engine.x2X(xr).GetDataArray() + \
                1j * engine.x2X(xi).GetDataArray()
            
            #vec =  lfHrB.GetDataArray() + 1j* lfHiB.GetDataArray()
            vec =  engine.b2B(lfHr).GetDataArray() + \
                1j* engine.b2B(lfHi).GetDataArray()
            
            weight_E = np.sum(et.dot(vec))
            if use_parallel:
                weight_E = np.sum(allgather(weight_E))
                
            v2 = LF2PyVec(lfHr, lfHi, horizontal=True)
            v2 *= 1. / weight_E

            # t4
            inc_wave = inc_amp * np.exp(1j * inc_phase / 180. * np.pi)
            phase = np.angle(inc_wave) * 180 / np.pi
            amp = np.sqrt(np.abs(inc_wave))

            t4 = np.array(
                [[amp * np.exp(1j * phase / 180. * np.pi)]])

            # weight = mfem.InnerProduct(engine.x2X(x), engine.b2B(lf2))

        # v1 = PyVec2PyMat(v1)
        v2 = PyVec2PyMat(v2.transpose())
        v2 = v2.transpose()
        
        #
        t4 = Array2PyVec(np.array([0.0]))
        t3 = IdentityPyMat(1, diag=1)

        '''
        Format of extar   (t2 is returnd as vertical(transposed) matrix)
        [M,  t1]   [  ]
        [      ] = [  ]
        [t2, t3]   [t4]

        and it returns if Lagurangian will be saved.
        '''
        return (None, v2, t3, t4, True)
        #return (None, None, t3, t4, True)
