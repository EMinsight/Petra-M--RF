import numpy as np

from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_model import Phys, PhysModule, VectorPhysCoefficient
from petram.model import Domain, Bdry, Pair

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3DUWbase')


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


class EM3DUW_Domain(Domain, Phys):
    has_3rd_panel = True
    vt3 = Vtable(data)

    def __init__(self, **kwargs):
        super(EM3DUW_Domain, self).__init__(**kwargs)
        Phys.__init__(self)

    @staticmethod
    def space_idx():
        TrialSpace = {"E_space": 0,
                      "H_space": 1,
                      "hatE_space": 2,
                      "hatH_space": 3}
        TestSpace = {"F_space": 0,
                     "G_space": 1}

        return TrialSpace, TestSpace

    def attribute_set(self, v):
        super(EM3DUW_Domain, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_init_coeff(self, engine, real=True, kfes=0):
        if kfes != 0:
            return
        if not self.use_Einit:
            return

        f_name = self.vt3.make_value_or_expression(self)
        coeff = Einit(3, f_name[0],
                      self.get_root_phys().ind_vars,
                      self._local_ns, self._global_ns,
                      real=real)
        return self.restrict_coeff(coeff, engine, vec=True)

    def add_dpg_integrator(self, engine, coeff, adder, integrator, sp1, sp2=None,
                           transpose=False, ir=None):

        if hasattr(coeff, "get_real_coefficient"):
            coeffr = coeff.get_real_coefficient()
            coeffi = coeff.get_imag_coefficient()
        else:
            coeffr = coeff
            coeffi = None

        if isinstance(coeffr, mfem.Coefficient):
            coeffr = self.restrict_coeff(coeffr, engine)
            if coeffi is not None:
                coeffi = self.restrict_coeff(coeffi, engine)
        elif isinstance(coeffr, mfem.VectorCoefficient):
            coeffr = self.restrict_coeff(coeffr, engine, vec=True)
            if coeffi is not None:
                coeffi = self.restrict_coeff(coeffi, engine, vec=True)
        elif isinstance(coeff, mfem.MatrixCoefficient):
            coeffr = self.restrict_coeff(coeffr, engine, matrix=True)
            if coeffi is not None:
                coeffi = self.restrict_coeff(coeffi, engine, matrix=True)
        else:
            assert False, "Unknown coefficient type: " + str(type(coeff[0]))

        itgr = integrator(coeffr)
        itgr._linked_coeff = coeffr
        if coeffi is not None:
            itgi = integrator(coeffi)
            itgi._linked_coeff = coeffi
        else:
            itgi = None

        if ir is not None:
            itgr.SetIntRule(ir)
            if itgi is not None:
                itgi.SetIntRule(ir)

        if transpose:
            itg2r = mfem.TransposeIntegrator(itgr)
            itg2r._link = itgr
            if itgi is not None:
                itg2i = mfem.TransposeIntegrator(itgi)
                itg2i._link = itgi
            else:
                itg2i = None
        else:
            itg2r = itgr
            itg2i = itgi

        # print(adder,itg2r, itg2i, sp1, sp2)
        if sp2 is None:   # a.AddDomainLFIntegrator
            adder(itg2r, itg2i, sp1)
        else:             # a.AddTestIntegrator, a.AddTrialIntegrator
            adder(itg2r, itg2i, sp1, sp2)


    def add_bf_epsmu_contribution(self, engine, a, cf1, cf2, cf3, cf4, cf5, cf6):
        TrialSpace, TestSpace = self.space_idx()

        # a.StoreMatrices()  # needed for AMR

        #  attempt to increase the integration order to avoid non SPD G-matrix
        #phys = self.get_root_phys()
        #mesh = engine.emeshes[phys.emesh_idx]
        #geom = mesh.GetElementGeometry(0)
        #ir = mfem.IntRules.Get(geom, 2*phys.test_order+2)
        ir = None
        # xxxxxx -i ω ϵ (E , G) = i (- ω ϵ E, G)
        # i ω ϵ (E , G)

        self.add_dpg_integrator(engine, cf1, a.AddTrialIntegrator,
                                mfem.VectorFEMassIntegrator,
                                TrialSpace["E_space"],
                                TestSpace["G_space"],
                                transpose=True, ir=ir)
        # a.AddTrialIntegrator(None,
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
                                transpose=True, ir=ir)
        # a.AddTrialIntegrator(None, mfem.TransposeIntegrator(
        #    mfem.VectorFEMassIntegrator(cf2)),
        #    TrialSpace["H_space"],
        #    TestSpace["F_space"])

        # (1,1)
        # |μ|^2 ω^2 (F,δF)
        self.add_dpg_integrator(engine, cf6, a.AddTestIntegrator,
                                mfem.VectorFEMassIntegrator,
                                TestSpace["F_space"],
                                TestSpace["F_space"], ir=ir)
        # a.AddTestIntegrator(mfem.VectorFEMassIntegrator(mu2omeg2_cf), None,
        #                    TestSpace["F_space"],
        #                    TestSpace["F_space"])

        # (2, 1)
        # -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
        # (i ω μ* F,∇ × δG)
        self.add_dpg_integrator(engine, cf4, a.AddTestIntegrator,
                                mfem.MixedVectorWeakCurlIntegrator,
                                TestSpace["F_space"],
                                TestSpace["G_space"], ir=ir)
        # a.AddTestIntegrator(None, mfem.MixedVectorWeakCurlIntegrator(cf3),
        #                    TestSpace["F_space"],
        #                    TestSpace["G_space"])

        # xxxx -i ω ϵ (∇ × F, δG)
        # (i ω ϵ ∇ × F, δG)
        self.add_dpg_integrator(engine, cf1, a.AddTestIntegrator,
                                mfem.MixedVectorCurlIntegrator,
                                TestSpace["F_space"],
                                TestSpace["G_space"], ir=ir)
        # a.AddTestIntegrator(None, mfem.MixedVectorCurlIntegrator(cf1),
        #                    TestSpace["F_space"],
        #                    TestSpace["G_space"])
        # (1, 2)
        # *****i ω μ (∇ × G,δF)
        # (-i ω μ ∇ × G,δF)
        self.add_dpg_integrator(engine, cf2, a.AddTestIntegrator,
                                mfem.MixedVectorCurlIntegrator,
                                TestSpace["G_space"],
                                TestSpace["F_space"],)
        # a.AddTestIntegrator(None, mfem.MixedVectorCurlIntegrator(cf2),
        #                    TestSpace["G_space"],
        #                    TestSpace["F_space"])

        # xxxxx i ω ϵ (G, ∇ × δF )
        # (-i ω ϵ* (G, ∇ × δF )
        self.add_dpg_integrator(engine, cf3, a.AddTestIntegrator,
                                mfem.MixedVectorWeakCurlIntegrator,
                                TestSpace["G_space"],
                                TestSpace["F_space"],)
        # a.AddTestIntegrator(None, mfem.MixedVectorWeakCurlIntegrator(cf4),
        #                    TestSpace["G_space"],
        #                    TestSpace["F_space"])
        # (2, 2)
        # |ϵ|^2 ω^2 (G,δG)
        self.add_dpg_integrator(engine, cf5, a.AddTestIntegrator,
                                mfem.VectorFEMassIntegrator,
                                TestSpace["G_space"],
                                TestSpace["G_space"],)
        # a.AddTestIntegrator(mfem.VectorFEMassIntegrator(cf5), None,
        #                    TestSpace["G_space"],
        #                    TestSpace["G_space"])





class EM3DUW_Bdry(Bdry, Phys):
    has_3rd_panel = True
    vt3 = Vtable(data)

    def __init__(self, **kwargs):
        super(EM3DUW_Bdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM3DUW_Bdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_init_coeff(self, engine, real=True, kfes=0):
        if kfes != 0:
            return
        if not self.use_Einit:
            return

        f_name = self.vt3.make_value_or_expression(self)
        coeff = Einit(3, f_name[0],
                      self.get_root_phys().ind_vars,
                      self._local_ns, self._global_ns,
                      real=real)
        return self.restrict_coeff(coeff, engine, vec=True)
