from petram.solver.std_solver_model import StdSolver, StandardSolver

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('DpgAmrSolver')
rprint = debug.regular_print('DpgAmrSolver')


class DpgAmrSolver(StdSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'AMR Stationary (DPG)'

    @classmethod
    def fancy_tree_name(self):
        return 'AMR Stationary (DPG)'

    def attribute_set(self, v):
        v = super(DpgAmrSolver, self).attribute_set(v)
        v['amr_max_num'] = 5
        v['amr_th'] = 1.0
        return v

    def panel1_param(self):
        ret = super(DpgAmrSolver, self).panel1_param()

        ret.extend([("#meth adapt",   self.amr_max_num,  0, {}, ),
                    ("thres.",   self.amr_th,  0, {}),])

        return ret

    def get_panel1_value(self):
        value = super(DpgAmrSolver, self). get_panel1_value()
        value.extend([self.amr_max_num, self.amr_th])

        return value

    def import_panel1_value(self, v):
        super(DpgAmrSolver, self).import_panel1_value(v[:-2])
        self.amr_max_num = v[-2]
        self.amr_th = v[-1]

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering run (is_first= ", is_first, ") ", self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = DpgAmrSolverInstance(
            self, engine) if self.instance is None else self.instance

        instance.set_blk_mask()
        if return_instance:
            return instance

        instance.configure_probes(self.probe)

        for i in range(self.amr_max_num):
            if self.init_only:
                engine.run_fill_X_block()
                engine.sol = engine.assembled_blocks[1][0]
                instance.sol = engine.sol

            elif self.load_sol:
                if is_first:
                    instance.assemble()
                    is_first = False
                instance.load_sol(self.sol_file)
            else:
                instance.assemble()

                update_operator = engine.check_block_matrix_changed(
                    instance.blk_mask)
                instance.solve(update_operator=update_operator)

            instance.save_solution(ksol=0,
                                   skip_mesh=False,
                                   mesh_only=False,
                                   save_parmesh=self.save_parmesh,
                                   save_sersol=self.save_sersol)
            engine.sol = instance.sol

            instance.save_probe()

            instance.run_amr(self.amr_th)

        self.instance = instance

        dprint1(debug.format_memory_usage())
        return is_first


class DpgAmrSolverInstance(StandardSolver):
    def __init__(self, *args, **kwargs):
        suepr(DpgAmrSolverInstance, self).__init__(*args, **kwargs)

        self.elements_to_refine = mfem.intArray()
        self.dpg_phys = []

    def set_amr(self):
        phys_target = self.get_phys()
        for phys in phys_target:
            if has_attr(phys, 'set_dpg_amr'):
                phys.set_dpg_amr()
                if phys not in self.dpg_phys:
                    self.dpg_phys.append(phys)

    def assemble(self, inplace=True):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        # use get_phys to apply essential to all phys in solvestep
        dprint1("Asembling system matrix",
                [x.name() for x in phys_target],
                [x.name() for x in phys_range])

        self.set_amr()

        engine.run_verify_setting(phys_target, self.gui)

        M_updated = engine.run_assemble_mat(
            phys_target, phys_range)
        B_updated = engine.run_assemble_b(phys_target)

        engine.run_apply_essential(phys_target, phys_range)

        _blocks, M_changed = self.engine.run_assemble_MXB_blocks(self.compute_A,
                                                                 self.compute_rhs,
                                                                 inplace=inplace)
        # A, X, RHS, Ae, B, M, names = blocks
        self.assembled = True
        return M_changed

    def run_amr(self, amr_th)
    from petram.mfem_config import use_parallel

    engine = self.engine
    dpg_phys = self.dpg_phys[0]

    # compute_residul
    for physname in engine.form_info:
        phys, ifess, rifess, depvars = engine.form_info[physname]

        if phys == dpg_phys:
            self.do_amr(engine, phys, ifess, rifess, depvars, amr_th)

    def do_amr(self, engine, phys, ifess, rifess, depvars, amr_th):

        a = engine.r_a[ifess[0], rifess[0]]
        x = engine.gf_alloc[(engine.access_idx, physname)].blockvector

        residuals = a.ComputeResidual(x)

        residual = residuals.Norml2()
        maxresidual = residuals.Max()
        globalresidual = residual * residual

        if use_parallel:
            maxresidual = MPI.COMM_WORLD.allreduce(maxresidual, op=MPI.MAX)
            globalresidual = MPI.COMM_WORLD.allreduce(
                globalresidual, op=MPI.SUM)
            globalresidual = np.sqrt(globalresidual)

        dprint1("Maxl Residual: ", maxresidual)
        dprint1("Global Residual: ", globalresidual)

        self.residual = residual

        emesh_idx = phys.emesh_idx
        mesh = engine.emeshes[emesh_idx]

        elements_to_refine = self.elements_to_refine
        if amr_th > 0.0:
            elements_to_refine.SetSize(0)
            for iel in range(pmesh.GetNE()):
                if residuals[iel] > theta * maxresidual:
                    elements_to_refine.Append(iel)
            mesh.GeneralRefinement(elements_to_refine, 1, 1)
        else:
            mesh.UniformRefinement()

        trial_fes = engine.fepaces[name] for name in depvars]
            for i in range(len(trial_fes)):
        trial_fes[i].Update(False)
            a.Update()
