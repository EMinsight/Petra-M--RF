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

    def add_dpg_integrator(self, engine, coeff, adder, integrator, sp1, sp2,
                           idx=None,  transpose=False):

        if hasattr(coeff, "get_real_coefficient"):
            coeffr = coeff.get_real_coefficient()
            coeffi = coeff.get_imag_coefficient()
        else:
            coeffr = coeff
            coeffi = None

        if isinstance(coeffr, mfem.Coefficient):
            coeffr = self.restrict_coeff(coeffr, engine, idx=idx)
            if coeffi is not None:
                coeffi = self.restrict_coeff(coeffi, engine, idx=idx)
        elif isinstance(coeffr, mfem.VectorCoefficient):
            coeffr = self.restrict_coeff(coeffr, engine, vec=True, idx=idx)
            if coeffi is not None:
                coeffi = self.restrict_coeff(coeffi, engine, vec=True, idx=idx)
        elif isinstance(coeff, mfem.MatrixCoefficient):
            coeffr = self.restrict_coeff(coeffr, engine, matrix=True, idx=idx)
            if coeffi is not None:
                coeffi = self.restrict_coeff(
                    coeffi, engine, matrix=True, idx=idx)
        else:
            assert False, "Unknown coefficient type: " + str(type(coeff[0]))

        itgr = integrator(coeffr)
        itgr._linked_coeff = coeffr
        if coeffi is not None:
            itgi = integrator(coeffi)
            itgi._linked_coeff = coeffi
        else:
            itgi = None

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

        #print(adder,itg2r, itg2i, sp1, sp2)
        adder(itg2r, itg2i, sp1, sp2)


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
