from __future__ import print_function
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.em3duw.em3duw_portmode import (C_Et_TE,
                                                C_jwHt_TE,
                                                C_Et_TEM,
                                                C_jwHt_TEM,
                                                C_Et_CoaxTEM,
                                                C_jwHt_CoaxTEM,)
from petram.helper.geom import find_circle_center_radius
from petram.helper.geom import connect_pairs
from petram.phys.em3duw.em3duw_base import EM3DUW_Bdry, EM3DUW_Domain
from petram.model import Bdry
from petram.mfem_config import use_parallel
'''

   3D port boundary condition


'''

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
        return v

    def panel1_param(self):
        return ([["port id", str(self.port_idx), 0, {}],
                 ["mode", self.mode, 4, {"readonly": True,
                                         "choices": ["TE", "TEM", "Coax(TEM)"]}],
                 ["m/n", ','.join(str(x) for x in self.mn), 0, {}], ] +
                self.vt.panel_param(self))

    def get_panel1_value(self):
        return ([str(self.port_idx),
                 self.mode, ','.join(str(x) for x in self.mn)] +
                self.vt.get_panel_value(self))

    def import_panel1_value(self, v):
        self.port_idx = v[0]
        self.mode = v[1]
        self.mn = [int(x) for x in v[2].split(',')]
        self.vt.import_panel_value(self, v[3:])

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

        '''
        self.update_inc_amp_phase()

    def update_inc_amp_phase(self):
        try:
            self.inc_amp = self.eval_phys_expr(str(self.inc_amp_txt),  'inc_amp')[0]
            self.inc_phase = self.eval_phys_expr(str(self.inc_phase_txt), 'inc_phase', chk_float = True)[0]
        except:
            raise ValueError("Cannot evaluate amplitude/phase to float number")
        '''

    def preprocess_params(self, engine):
        # find normal (outward) vector...
        mesh = engine.get_emesh(mm=self)

        fespace = engine.fespaces[self.get_root_phys().dep_vars[0]]

        nbe = mesh.GetNBE()
        ibe = np.array([i for i in range(nbe)
                        if mesh.GetBdrElement(i).GetAttribute() ==
                        self._sel_index[0]])

        el = mesh.GetBdrElement(ibe[0])
        Tr = fespace.GetBdrElementTransformation(ibe[0])
        rules = mfem.IntegrationRules()
        ir = rules.Get(el.GetGeometryType(), 1)
        Tr.SetIntPoint(ir.IntPoint(0))
        nor = mfem.Vector(3)
        mfem.CalcOrtho(Tr.Jacobian(), nor)

        self.norm = nor.GetDataArray().copy()
        self.norm = self.norm / np.sqrt(np.sum(self.norm**2))

        # freq = self._global_ns["freq"]
        # self.omega = freq * 2 * np.pi
        # dprint1("Frequency " + (freq).__repr__())
        dprint1("Normal Vector " + list(self.norm).__repr__())

        # find rectangular shape
        if str(self.mode).upper().strip() in ['TE', 'TM', 'TEM']:
            edges = np.array([mesh.GetBdrElementEdges(i)[0]
                              for i in ibe]).flatten()
            d = {}
            for x in edges:
                d[x] = x in d
            edges = [x for x in d.keys() if not d[x]]
            ivert = [mesh.GetEdgeVertices(x) for x in edges]
            ivert = connect_pairs(ivert)
            vv = np.vstack([mesh.GetVertexArray(i) for i in ivert])

            self.ctr = (np.max(vv, 0) + np.min(vv, 0)) / 2.0
            dprint1("Center " + list(self.ctr).__repr__())

            # rectangular port
            # idx = np.argsort(np.sqrt(np.sum((vv - self.ctr)**2,1)))
            # corners = vv[idx[-4:],:]
            # since vv is cyclic I need to omit last one element here..
            idx = np.argsort(np.sqrt(np.sum((vv[:-1] - self.ctr)**2, 1)))
            corners = vv[:-1][idx[-4:], :]
            for i in range(4):
                dprint1("Corner " + list(corners[i]).__repr__())
            tmp = np.sort(np.sqrt(np.sum((corners - corners[0, :])**2, 1)))
            self.b = tmp[1]
            self.a = tmp[2]
            tmp = np.argsort(np.sqrt(np.sum((corners - corners[0, :])**2, 1)))
            self.c = corners[0]  # corner
            self.b_vec = corners[tmp[1]] - corners[0]
            self.a_vec = np.cross(self.b_vec, self.norm)
#            self.a_vec = corners[tmp[2]]-corners[0]
            self.b_vec = self.b_vec / np.sqrt(np.sum(self.b_vec**2))
            self.a_vec = self.a_vec / np.sqrt(np.sum(self.a_vec**2))
            if np.sum(np.cross(self.a_vec, self.b_vec) * self.norm) > 0:
                self.a_vec = -self.a_vec

            if self.mode == 'TEM':
                '''
                special handling
                set a vector along PEC-like edge, regardless the actual
                length of edges
                '''
                for i in range(nbe):
                    if (edges[0] in mesh.GetBdrElementEdges(i)[0] and
                            self._sel_index[0] != mesh.GetBdrAttribute(i)):
                        dprint1("Checking surface :", mesh.GetBdrAttribute(i))
                        attr = mesh.GetBdrAttribute(i)
                        break
                for node in self.get_root_phys().walk():
                    if not isinstance(node, Bdry):
                        continue
                    if not node.enabled:
                        continue
                    if attr in node._sel_index:
                        break
                from petram.model import Pair
                ivert = mesh.GetEdgeVertices(edges[0])
                vect = mesh.GetVertexArray(
                    ivert[0]) - mesh.GetVertexArray(ivert[1])
                vect = vect / np.sqrt(np.sum(vect**2))
                do_swap = False
                if (isinstance(node, Pair) and
                        np.abs(np.sum(self.a_vec * vect)) > 0.9):
                    do_swap = True
                if (not isinstance(node, Pair) and
                        np.abs(np.sum(self.a_vec * vect)) < 0.001):
                    do_swap = True
                if do_swap:
                    dprint1("swapping port edges")
                    tmp = self.a_vec
                    self.a_vec = -self.b_vec
                    self.b_vec = tmp
                    # - sign is to keep a \times b direction.
                    tmp = self.a
                    self.a = self.b
                    self.b = tmp
            if self.a_vec[np.argmax(np.abs(self.a_vec))] < 0:
                self.a_vec = -self.a_vec
                self.b_vec = -self.b_vec
            dprint1("Long Edge  " + self.a.__repr__())
            dprint1("Long Edge Vec." + list(self.a_vec).__repr__())
            dprint1("Short Edge  " + self.b.__repr__())
            dprint1("Short Edge Vec." + list(self.b_vec).__repr__())
        elif self.mode == 'Coax(TEM)':
            edges = np.array([mesh.GetBdrElementEdges(i)[0]
                              for i in ibe]).flatten()
            d = {}
            for x in edges:
                d[x] = x in d
            edges = [x for x in d.keys() if not d[x]]
            ivert = [mesh.GetEdgeVertices(x) for x in edges]
            iv1, iv2 = connect_pairs(ivert)  # index of outer/inner circles
            vv1 = np.vstack([mesh.GetVertexArray(i) for i in iv1])
            vv2 = np.vstack([mesh.GetVertexArray(i) for i in iv2])
            ctr1, a1 = find_circle_center_radius(vv1, self.norm)
            ctr2, b1 = find_circle_center_radius(vv2, self.norm)
            self.ctr = np.mean((ctr1, ctr2), 0)
            self.a = a1 if a1 < b1 else b1
            self.b = a1 if a1 > b1 else b1
            dprint1("Big R:  " + self.b.__repr__())
            dprint1("Small R: " + self.a.__repr__())
            dprint1("Center:  " + self.ctr.__repr__())
            vv = vv1
        C_Et, C_jwHt = self.get_coeff_cls()

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)
        dprint1("E field pattern", eps, mur)
        Et = C_Et(3, self, real=True, eps=eps, mur=mur)
        for p in vv:
            dprint1(p.__repr__() + ' : ' + Et.EvalValue(p).__repr__())
        dprint1("H field pattern")
        Ht = C_jwHt(3, 0.0, self, real=False, eps=eps, mur=mur)
        for p in vv:
            dprint1(p.__repr__() + ' : ' + Ht.EvalValue(p).__repr__())

    def get_coeff_cls(self):
        if self.mode == 'TEM':
            return C_Et_TEM, C_jwHt_TEM
        elif self.mode == 'TE':
            return C_Et_TE, C_jwHt_TE
        elif self.mode == 'Coax(TEM)':
            return C_Et_CoaxTEM, C_jwHt_CoaxTEM
        else:
            raise NotImplementedError(
                "you must implement this mode")

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

            lfHr = engine.new_lf(fes_Ht)
            Htr = C_jwHt(3, 0, self, real=True, eps=eps, mur=mur)
            Htr = self.restrict_coeff(Htr, engine, vec=True)
            intg = mfem.VectorFEDomainLFIntegrator(Htr)
            lfHr.AddBoundaryIntegrator(intg)
            lfHr.Assemble()

            lfHi = engine.new_lf(fes_Ht)
            Hti = C_jwHt(3, 0, self, real=False, eps=eps, mur=mur)
            Hti = self.restrict_coeff(Hti, engine, vec=True)
            intg = mfem.VectorFEDomainLFIntegrator(Hti)
            lfHi.AddBoundaryIntegrator(intg)
            lfHi.Assemble()

            xr = engine.new_gf(fes_Ht)
            xr.Assign(0.0)
            arr = self.get_restriction_array(engine)
            xr.ProjectBdrCoefficientTangent(Htr, arr)
            xi = engine.new_gf(fes_Ht)
            xi.Assign(0.0)
            arr = self.get_restriction_array(engine)
            xi.ProjectBdrCoefficientTangent(Hti, arr)

            et = engine.x2X(xr).GetDataArray() + 1j * \
                engine.x2X(xi).GetDataArray()
            vec = engine.b2B(lfHr).GetDataArray() + 1j * \
                engine.b2B(lfHi).GetDataArray()

            weight_E = np.sum(et.dot(vec))
            print("weight", weight_E)
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
        print(v2)
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
