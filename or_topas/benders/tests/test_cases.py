import pyomo.environ as pyo

from or_topas.benders import (
    BendersGenerator_Serial,
)
from or_topas.util.mymunch import MyMunch
import itertools
from math import pi as pi_value
import time


class modified_absolute_value:
    @staticmethod
    def create_root(
        eta_count=None,
        eta_lb=-10,
        eta_ub=None,
        x_lb=None,
        x_ub=None,
        obj_offset=0,
    ):
        m = pyo.ConcreteModel()
        # x is benders variable, saved for later
        m.x = pyo.Var(bounds=(x_lb, x_ub), initialize=0)
        if eta_count is None:
            m.eta = pyo.Var(bounds=(eta_lb, eta_ub))
            m.obj = pyo.Objective(expr=obj_offset + m.eta, sense=pyo.minimize)
        else:
            m.scenarios = pyo.Set(initialize=range(eta_count), ordered=True)
            m.eta = pyo.Var(m.scenarios, bounds=(eta_lb, eta_ub))
            m.obj = pyo.Objective(
                expr=obj_offset + sum(m.eta[s] for s in m.scenarios), sense=pyo.minimize
            )
        return m

    @staticmethod
    def create_subproblem(root_x, data):
        """
        This subproblem implements the following function

        Q(x) =  max{R(x-a), -L(x-a)} if x \in [LB, UB]
                +\infty              if x \not \in [LB,UB]

        The subproblem that implements this is:
        Q(x) = min_{y >= 0} R*y[R] + L*y[L]
                s.t.    -y[R] + y[L] == a - x
                        y[UB_Slack] == UB - x
                        y[LB_Slack] == -LB + x

        Note that this is only LP representable when -L <= R
        This subproblem should result in the following cuts:
        Opt Cuts:
        \theta >= R(x-a)
        \theta >= -L(x-a)
        Corresponding to dual vertices [R, 0, 0]' and [-L, 0, 0]'

        Feas Cuts:
        0 >= x - UB
        0 >= -x + LB
        Corresponding to extreme rays [0, -1, 0]' and [0, 0, -1]
        """

        if data == None:
            data = MyMunch(a=0, L=1, R=1, LB=-5, UB=5)
        else:
            assert isinstance(data, MyMunch), "Need data argument to be a MyMunch"
            # these variables need default values
            data.a = 0 if ("a" not in data or data.a is None) else data.a
            data.L = 1 if ("L" not in data or data.L is None) else data.L
            data.R = 1 if ("R" not in data or data.R is None) else data.R

            # optional arguments
            data.LB = -5 if "LB" not in data else data.LB
            data.UB = 5 if "UB" not in data else data.UB

        m = pyo.ConcreteModel()
        m.x = pyo.Var()

        y_indices = ["Right", "Left"]
        if data.LB is not None:
            y_indices.append("LB_Slack")
        if data.UB is not None:
            y_indices.append("UB_Slack")
        m.y_indices = pyo.Set(initialize=y_indices)
        m.y = pyo.Var(m.y_indices, bounds=(0, None))

        # objective
        m.obj = pyo.Objective(expr=data.R * m.y["Right"] + data.L * m.y["Left"])

        # vertex constriant
        m.vertex_cons = pyo.Constraint(expr=-m.y["Right"] + m.y["Left"] == data.a - m.x)

        # optional lower bound constraint
        if data.LB is not None:
            m.lb_cons = pyo.Constraint(expr=m.y["LB_Slack"] - m.x == -data.LB)

        if data.UB is not None:
            m.ub_cons = pyo.Constraint(expr=m.y["UB_Slack"] + m.x == data.UB)

        complicating_vars_map = pyo.ComponentMap()
        complicating_vars_map[root_x] = m.x

        return m, complicating_vars_map

    @staticmethod
    def setup_modified_absolute_value(
        solver_name,
        CutGenerator=BendersGenerator_Serial,
        **kwargs,
    ):
        raise NotImplementedError(
            "The modified absolute value problem requires a persistent at present to enable feasibility cuts"
        )

    @staticmethod
    def setup_modified_absolute_value_persistent(
        solver_name,
        CutGenerator=BendersGenerator_Serial,
        **kwargs,
    ):
        transform = kwargs.get("transform", None)
        eta_count = kwargs.get("eta_count", None)
        data = kwargs.get("data", None)
        obj_offset = kwargs.get("obj_offset", 0)
        m = modified_absolute_value.create_root(
            eta_count=eta_count, obj_offset=obj_offset
        )
        # TODO/N.B. using list(m.x) here breaks
        root_vars = [m.x]
        m.benders = CutGenerator()
        m.benders.set_input(
            root_vars=root_vars,
            tol=1e-8,
            transform=transform,
            allow_infeasible=True,
        )
        if eta_count is None:
            subproblem_fn_kwargs = dict()
            subproblem_fn_kwargs["root_x"] = m.x
            subproblem_fn_kwargs["data"] = data
            m.benders.add_subproblem(
                subproblem_fn=modified_absolute_value.create_subproblem,
                subproblem_fn_kwargs=subproblem_fn_kwargs,
                root_eta=m.eta,
                subproblem_solver=solver_name,
            )
        else:
            for s in m.scenarios:
                subproblem_fn_kwargs = dict()
                subproblem_fn_kwargs["root_x"] = m.x
                subproblem_fn_kwargs["data"] = data
                m.benders.add_subproblem(
                    subproblem_fn=modified_absolute_value.create_subproblem,
                    subproblem_fn_kwargs=subproblem_fn_kwargs,
                    root_eta=m.eta[s],
                    subproblem_solver=solver_name,
                )
        opt = pyo.SolverFactory(solver_name)
        opt.set_instance(m)
        return opt, m

    @staticmethod
    def run_modified_absolute_value(
        mip_solver,
        mode="d",
        transform="standard_lp",
        add_upper_bounds=False,
        include_print=False,
        is_persistent=False,
        data=None,
        obj_offset=0,
    ):
        t0 = time.time()
        if mode == "d":
            eta_count = None
        elif mode == "s":
            eta_count = 2
        else:
            eta_count = mode

        setup_handle = modified_absolute_value.setup_modified_absolute_value
        if is_persistent:
            setup_handle = (
                modified_absolute_value.setup_modified_absolute_value_persistent
            )

        opt, m = setup_handle(
            solver_name=mip_solver,
            transform=transform,
            eta_count=eta_count,
            data=data,
            obj_offset=obj_offset,
        )

        if add_upper_bounds:
            if eta_count is None:
                m.eta.setub(100)
            else:
                for s in m.scenarios:
                    m.eta[s].setub(100)
        if include_print:
            print("{0:<15}{1:<15}{2:<15}".format("# Cuts", "X", "Total_Time"))
        for i in range(30):
            if is_persistent:
                res = opt.solve(tee=False, save_results=False)
                cuts_added = m.benders.generate_cut()
                for c in cuts_added:
                    opt.add_constraint(c)
            else:
                res = opt.solve(m, tee=False)
                cuts_added = m.benders.generate_cut()
            if include_print:
                print(
                    "{0:<15}{1:<15.2f}{2:<15.2f}".format(
                        len(cuts_added),
                        m.x.value,
                        time.time() - t0,
                    )
                )
            if len(cuts_added) == 0:
                break

        return opt, m


class absolute_value:
    @staticmethod
    def create_root(
        eta_count=None, eta_lb=-10, eta_ub=None, x_lb=None, x_ub=None, obj_offset=0
    ):
        m = pyo.ConcreteModel()
        # x is benders variable, saved for later
        m.x = pyo.Var(bounds=(x_lb, x_ub), initialize=0)
        if eta_count is None:
            m.eta = pyo.Var(bounds=(eta_lb, eta_ub))
            m.obj = pyo.Objective(expr=obj_offset + m.eta, sense=pyo.minimize)
        else:
            m.scenarios = pyo.Set(initialize=range(eta_count), ordered=True)
            m.eta = pyo.Var(m.scenarios, bounds=(eta_lb, eta_ub))
            m.obj = pyo.Objective(
                expr=obj_offset + sum(m.eta[s] for s in m.scenarios), sense=pyo.minimize
            )
        return m

    @staticmethod
    def create_subproblem(root):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y1 = pyo.Var(bounds=(0, None))
        m.y2 = pyo.Var(bounds=(0, None))
        m.obj = pyo.Objective(expr=m.y1 + m.y2)
        m.c1 = pyo.Constraint(expr=m.y1 - m.y2 == m.x)

        complicating_vars_map = pyo.ComponentMap()
        complicating_vars_map[root.x] = m.x

        return m, complicating_vars_map

    @staticmethod
    def setup_absolute_value(
        solver_name,
        CutGenerator=BendersGenerator_Serial,
        **kwargs,
    ):
        transform = kwargs.get("transform", None)
        eta_count = kwargs.get("eta_count", None)
        obj_offset = kwargs.get("obj_offset", 0)
        m = absolute_value.create_root(eta_count=eta_count, obj_offset=obj_offset)
        # TODO/N.B. using list(m.x) here breaks
        root_vars = [m.x]
        m.benders = CutGenerator()
        m.benders.set_input(root_vars=root_vars, tol=1e-8, transform=transform)
        if eta_count is None:
            subproblem_fn_kwargs = dict()
            subproblem_fn_kwargs["root"] = m
            m.benders.add_subproblem(
                subproblem_fn=absolute_value.create_subproblem,
                subproblem_fn_kwargs=subproblem_fn_kwargs,
                root_eta=m.eta,
                subproblem_solver=solver_name,
            )
        else:
            for s in m.scenarios:
                subproblem_fn_kwargs = dict()
                subproblem_fn_kwargs["root"] = m
                m.benders.add_subproblem(
                    subproblem_fn=absolute_value.create_subproblem,
                    subproblem_fn_kwargs=subproblem_fn_kwargs,
                    root_eta=m.eta[s],
                    subproblem_solver=solver_name,
                )
        opt = pyo.SolverFactory(solver_name)
        return opt, m

    @staticmethod
    def setup_absolute_value_persistent(
        solver_name,
        CutGenerator=BendersGenerator_Serial,
        **kwargs,
    ):
        transform = kwargs.get("transform", None)
        eta_count = kwargs.get("eta_count", None)
        obj_offset = kwargs.get("obj_offset", 0)
        m = absolute_value.create_root(eta_count=eta_count, obj_offset=obj_offset)
        root_vars = [m.x]
        m.benders = CutGenerator()
        m.benders.set_input(root_vars=root_vars, tol=1e-8, transform=transform)
        if eta_count is None:
            subproblem_fn_kwargs = dict()
            subproblem_fn_kwargs["root"] = m
            m.benders.add_subproblem(
                subproblem_fn=absolute_value.create_subproblem,
                subproblem_fn_kwargs=subproblem_fn_kwargs,
                root_eta=m.eta,
                subproblem_solver=solver_name,
            )
        else:
            for s in m.scenarios:
                subproblem_fn_kwargs = dict()
                subproblem_fn_kwargs["root"] = m
                m.benders.add_subproblem(
                    subproblem_fn=absolute_value.create_subproblem,
                    subproblem_fn_kwargs=subproblem_fn_kwargs,
                    root_eta=m.eta[s],
                    subproblem_solver=solver_name,
                )
        opt = pyo.SolverFactory(solver_name)
        opt.set_instance(m)
        return opt, m

    @staticmethod
    def run_absolute_value(
        mip_solver,
        mode="d",
        transform="standard_lp",
        add_upper_bounds=False,
        include_print=False,
        is_persistent=False,
        obj_offset=0,
    ):
        t0 = time.time()
        if mode == "d":
            eta_count = None
        else:
            eta_count = 2

        setup_handle = absolute_value.setup_absolute_value
        if is_persistent:
            setup_handle = absolute_value.setup_absolute_value_persistent

        opt, m = setup_handle(
            solver_name=mip_solver,
            transform=transform,
            eta_count=eta_count,
            obj_offset=obj_offset,
        )

        if add_upper_bounds:
            if eta_count is None:
                m.eta.setub(100)
            else:
                for s in m.scenarios:
                    m.eta[s].setub(100)
        if include_print:
            print("{0:<15}{1:<15}{2:<15}".format("# Cuts", "X", "Total_Time"))
        for i in range(30):
            if is_persistent:
                res = opt.solve(tee=False, save_results=False)
                cuts_added = m.benders.generate_cut()
                for c in cuts_added:
                    opt.add_constraint(c)
            else:
                res = opt.solve(m, tee=False)
                cuts_added = m.benders.generate_cut()
            if include_print:
                print(
                    "{0:<15}{1:<15.2f}{2:<15.2f}".format(
                        len(cuts_added),
                        m.x.value,
                        time.time() - t0,
                    )
                )
            if len(cuts_added) == 0:
                break

        return opt, m


class Farmer:
    def __init__(self):
        self.crops = ["WHEAT", "CORN", "SUGAR_BEETS"]
        self.total_acreage = 500
        self.PriceQuota = {
            "WHEAT": 100000.0,
            "CORN": 100000.0,
            "SUGAR_BEETS": 6000.0,
        }
        self.SubQuotaSellingPrice = {
            "WHEAT": 170.0,
            "CORN": 150.0,
            "SUGAR_BEETS": 36.0,
        }
        self.SuperQuotaSellingPrice = {
            "WHEAT": 0.0,
            "CORN": 0.0,
            "SUGAR_BEETS": 10.0,
        }
        self.CattleFeedRequirement = {
            "WHEAT": 200.0,
            "CORN": 240.0,
            "SUGAR_BEETS": 0.0,
        }
        self.PurchasePrice = {
            "WHEAT": 238.0,
            "CORN": 210.0,
            "SUGAR_BEETS": 100000.0,
        }
        self.PlantingCostPerAcre = {
            "WHEAT": 150.0,
            "CORN": 230.0,
            "SUGAR_BEETS": 260.0,
        }
        self.scenarios = [
            "BelowAverageScenario",
            "AverageScenario",
            "AboveAverageScenario",
        ]
        self.crop_yield = dict()
        self.crop_yield["BelowAverageScenario"] = {
            "WHEAT": 2.0,
            "CORN": 2.4,
            "SUGAR_BEETS": 16.0,
        }
        self.crop_yield["AverageScenario"] = {
            "WHEAT": 2.5,
            "CORN": 3.0,
            "SUGAR_BEETS": 20.0,
        }
        self.crop_yield["AboveAverageScenario"] = {
            "WHEAT": 3.0,
            "CORN": 3.6,
            "SUGAR_BEETS": 24.0,
        }
        self.scenario_probabilities = dict()
        self.scenario_probabilities["BelowAverageScenario"] = 0.3333
        self.scenario_probabilities["AverageScenario"] = 0.3334
        self.scenario_probabilities["AboveAverageScenario"] = 0.3333

    @staticmethod
    def create_root(farmer):
        m = pyo.ConcreteModel()

        m.crops = pyo.Set(initialize=farmer.crops, ordered=True)
        m.scenarios = pyo.Set(initialize=farmer.scenarios, ordered=True)

        m.devoted_acreage = pyo.Var(m.crops, bounds=(0, farmer.total_acreage))
        m.eta = pyo.Var(m.scenarios)
        for s in m.scenarios:
            m.eta[s].setlb(-432000 * farmer.scenario_probabilities[s])

        m.total_acreage_con = pyo.Constraint(
            expr=sum(m.devoted_acreage.values()) <= farmer.total_acreage
        )

        m.obj = pyo.Objective(
            expr=sum(
                farmer.PlantingCostPerAcre[crop] * m.devoted_acreage[crop]
                for crop in m.crops
            )
            + sum(m.eta.values())
        )
        return m

    @staticmethod
    def create_subproblem(root, farmer, scenario):
        m = pyo.ConcreteModel()

        m.crops = pyo.Set(initialize=farmer.crops, ordered=True)

        m.devoted_acreage = pyo.Var(m.crops)
        m.QuantitySubQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
        m.QuantitySuperQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
        m.QuantityPurchased = pyo.Var(m.crops, bounds=(0.0, None))

        def EnforceCattleFeedRequirement_rule(m, i):
            return (
                farmer.CattleFeedRequirement[i]
                <= (farmer.crop_yield[scenario][i] * m.devoted_acreage[i])
                + m.QuantityPurchased[i]
                - m.QuantitySubQuotaSold[i]
                - m.QuantitySuperQuotaSold[i]
            )

        m.EnforceCattleFeedRequirement = pyo.Constraint(
            m.crops, rule=EnforceCattleFeedRequirement_rule
        )

        def LimitAmountSold_rule(m, i):
            return (
                m.QuantitySubQuotaSold[i]
                + m.QuantitySuperQuotaSold[i]
                - (farmer.crop_yield[scenario][i] * m.devoted_acreage[i])
                <= 0.0
            )

        m.LimitAmountSold = pyo.Constraint(m.crops, rule=LimitAmountSold_rule)

        def EnforceQuotas_rule(m, i):
            return (0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[i])

        m.EnforceQuotas = pyo.Constraint(m.crops, rule=EnforceQuotas_rule)

        obj_expr = sum(
            farmer.PurchasePrice[crop] * m.QuantityPurchased[crop] for crop in m.crops
        )
        obj_expr -= sum(
            farmer.SubQuotaSellingPrice[crop] * m.QuantitySubQuotaSold[crop]
            for crop in m.crops
        )
        obj_expr -= sum(
            farmer.SuperQuotaSellingPrice[crop] * m.QuantitySuperQuotaSold[crop]
            for crop in m.crops
        )
        m.obj = pyo.Objective(expr=farmer.scenario_probabilities[scenario] * obj_expr)

        complicating_vars_map = pyo.ComponentMap()
        for crop in m.crops:
            complicating_vars_map[root.devoted_acreage[crop]] = m.devoted_acreage[crop]

        return m, complicating_vars_map

    @staticmethod
    def setup_farmer_persistent(
        Farmer_Data,
        CutGenerator=BendersGenerator_Serial,
        **kwargs,
    ):
        # designed for gurobi_persistent
        solver_name = kwargs.get("solver_name", "gurobi_persistent")
        farmer = Farmer_Data
        m = Farmer.create_root(farmer=farmer)
        root_vars = list(m.devoted_acreage.values())
        m.benders = CutGenerator()
        transform = kwargs.get("transform", None)
        m.benders.set_input(root_vars=root_vars, tol=1e-8, transform=transform)
        for s in farmer.scenarios:
            subproblem_fn_kwargs = dict()
            subproblem_fn_kwargs["root"] = m
            subproblem_fn_kwargs["farmer"] = farmer
            subproblem_fn_kwargs["scenario"] = s
            m.benders.add_subproblem(
                subproblem_fn=Farmer.create_subproblem,
                subproblem_fn_kwargs=subproblem_fn_kwargs,
                root_eta=m.eta[s],
                subproblem_solver=solver_name,
            )
        opt = pyo.SolverFactory(solver_name)
        opt.set_instance(m)
        return opt, m

    @staticmethod
    def setup_farmer(
        Farmer_Data,
        solver_name,
        CutGenerator=BendersGenerator_Serial,
        **kwargs,
    ):
        farmer = Farmer_Data
        m = Farmer.create_root(farmer=farmer)
        root_vars = list(m.devoted_acreage.values())
        m.benders = CutGenerator()
        transform = kwargs.get("transform", None)
        m.benders.set_input(root_vars=root_vars, tol=1e-8, transform=transform)
        for s in farmer.scenarios:
            subproblem_fn_kwargs = dict()
            subproblem_fn_kwargs["root"] = m
            subproblem_fn_kwargs["farmer"] = farmer
            subproblem_fn_kwargs["scenario"] = s
            m.benders.add_subproblem(
                subproblem_fn=Farmer.create_subproblem,
                subproblem_fn_kwargs=subproblem_fn_kwargs,
                root_eta=m.eta[s],
                subproblem_solver=solver_name,
            )
        opt = pyo.SolverFactory(solver_name)
        return opt, m

    @staticmethod
    def run_farmer(
        mip_solver,
        mode="s",
        transform="standard_lp",
        add_upper_bounds=False,
        include_print=False,
        is_persistent=False,
        include_assert_checks=False,
    ):
        t0 = time.time()
        local_farmer = Farmer()
        if mode == "d":
            # local_farmer = Farmer()
            local_farmer.scenario_probabilities = {"AverageScenario": 1.0}
            local_farmer.scenarios = ["AverageScenario"]

        farmer_setup_handle = Farmer.setup_farmer
        if is_persistent:
            farmer_setup_handle = Farmer.setup_farmer_persistent

        opt, m = farmer_setup_handle(
            local_farmer,
            solver_name=mip_solver,
            transform=transform,
        )

        if add_upper_bounds:
            for s in m.scenarios:
                m.eta[s].setub(1_000_000)
        if include_print:
            print(
                "{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}".format(
                    "# Cuts", "Corn", "Sugar Beets", "Wheat", "Total_Time"
                )
            )
        for i in range(30):
            if is_persistent:
                res = opt.solve(tee=False, save_results=False)
                cuts_added = m.benders.generate_cut()
                for c in cuts_added:
                    opt.add_constraint(c)
            else:
                res = opt.solve(m, tee=False)
                cuts_added = m.benders.generate_cut()
            if include_print:
                print(
                    "{0:<15}{1:<15.2f}{2:<15.2f}{3:<15.2f}{4:<15.2f}".format(
                        len(cuts_added),
                        m.devoted_acreage["CORN"].value,
                        m.devoted_acreage["SUGAR_BEETS"].value,
                        m.devoted_acreage["WHEAT"].value,
                        time.time() - t0,
                    )
                )
            if len(cuts_added) == 0:
                break

        if include_assert_checks:
            if mode == "s":
                tol = 1e-7
                assert abs(m.devoted_acreage["CORN"].value - 80) < tol
                assert abs(m.devoted_acreage["SUGAR_BEETS"].value - 250) < tol
                assert abs(m.devoted_acreage["WHEAT"].value - 170) < tol
        return opt, m


class Grothey:

    @staticmethod
    def create_root():
        m = pyo.ConcreteModel()
        m.y = pyo.Var(bounds=(1, None))
        m.eta = pyo.Var(bounds=(-10, None))
        m.obj = pyo.Objective(expr=m.y**2 + m.eta)
        return m

    @staticmethod
    def create_subproblem(root):
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var()
        m.x2 = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=-m.x2)
        m.c1 = pyo.Constraint(expr=(m.x1 - 1) ** 2 + m.x2**2 <= pyo.log(m.y))
        m.c2 = pyo.Constraint(expr=(m.x1 + 1) ** 2 + m.x2**2 <= pyo.log(m.y))

        complicating_vars_map = pyo.ComponentMap()
        complicating_vars_map[root.y] = m.y

        return m, complicating_vars_map


class EnergyGrid:
    def __init__(self):
        # model data

        # we can change these three parameters to get a unique value
        # originally not designed to be unique

        # original values
        self.load_dict = {"bus1": 0, "bus2": 0, "bus3": 100}
        self.gen_max_dict = {"bus1": 100, "bus2": 100, "bus3": 0}
        self.cost_dict = {"bus1": 50, "bus2": 50, "bus3": 0}
        self.flow_bounds_dict = {
            ("bus1", "bus2"): 100,
            ("bus1", "bus3"): 100,
            ("bus2", "bus3"): 100,
        }

        self.susceptance_dict = {
            ("bus1", "bus2"): 100,
            ("bus1", "bus3"): 100,
            ("bus2", "bus3"): 100,
        }

        self.flow_bounds_dict = {
            ("bus1", "bus2"): 100,
            ("bus1", "bus3"): 100,
            ("bus2", "bus3"): 100,
        }
        self.theta_bounds_dict = {"bus1": pi_value, "bus2": pi_value, "bus3": pi_value}
        self.buses = ["bus1", "bus2", "bus3"]
        self.lines = itertools.combinations(self.buses, 2)
        # {('bus1', 'bus2'), ('bus1', 'bus3'), ('bus2', 'bus3')}

    # DC-Optimal Power Flow Example
    # Adapted from TPEC paper example model
    # mode 0 is Copper Plate
    # mode 1 is Network Flow
    # mode 2 is DC-OPF with flow variables
    @staticmethod
    def create_tiny_opf(grid, mode=2):
        assert mode in [0, 1, 2], "Needs to be mode 0, 1, or 2"
        if mode == 0:
            print("Model in Copper Plate Mode")
        elif mode == 1:
            print("Model in Network Flow Mode")
        elif mode == 2:
            print("Model in DC-OPF")
        # model creation
        model = pyo.ConcreteModel()
        model.buses = pyo.Set(initialize=grid.buses)

        # always on components
        model.loads = pyo.Param(model.buses, initialize=grid.load_dict)
        model.capacity = pyo.Param(model.buses, initialize=grid.gen_max_dict)
        model.costs = pyo.Param(model.buses, initialize=grid.cost_dict)

        def gen_max_rule(m, bus):
            return (-1, m.capacity[bus] + 1)

        model.generation = pyo.Var(
            model.buses, domain=pyo.NonNegativeReals, bounds=gen_max_rule
        )
        # model.generation = pyo.Var(
        #     model.buses, domain=pyo.PositiveReals, bounds=gen_max_rule
        # )

        # Explicit lower bound constraint so dual accessed in with dual not reduced cost attribute
        def gen_lower_rule(m, bus):
            return m.generation[bus] >= 0

        model.gen_lower = pyo.Constraint(model.buses, rule=gen_lower_rule)

        # Explicit upper bound constraint so dual accessed in with dual not reduced cost attribute
        def gen_upper_rule(m, bus):
            return m.generation[bus] <= m.capacity[bus]

        model.gen_upper = pyo.Constraint(model.buses, rule=gen_upper_rule)

        def obj_rule(m):
            return pyo.summation(m.costs, m.generation)

        model.obj = pyo.Objective(rule=obj_rule)

        # copper plate mode
        if mode == 0:
            model.copper_plate_balance = pyo.Constraint(
                expr=sum(model.generation[i] for i in model.buses)
                - sum(model.loads[i] for i in model.buses)
                == 0
            )
        else:
            # Network flow or DC OPF Mode
            model.lines = pyo.Set(initialize=grid.lines)
            model.flow_max = pyo.Param(model.lines, initialize=grid.flow_bounds_dict)

            def NodesOut_init(m, node):
                for i, j in m.lines:
                    if i == node:
                        yield j

            model.NodesOut = pyo.Set(model.buses, initialize=NodesOut_init)

            def NodesIn_init(m, node):
                for i, j in m.lines:
                    if j == node:
                        yield i

            model.NodesIn = pyo.Set(model.buses, initialize=NodesIn_init)

            def flow_max_rule(m, i, j):
                return (-m.flow_max[i, j] - 1, m.flow_max[i, j] + 1)

            model.flows = pyo.Var(model.lines, bounds=flow_max_rule)

            # Explicit lower bound constraint so dual accessed in with dual not reduced cost attribute
            def flow_lower_rule(m, i, j):
                return m.flows[i, j] >= -m.flow_max[i, j]

            model.flow_lower = pyo.Constraint(model.lines, rule=flow_lower_rule)

            # Explicit upper bound constraint so dual accessed in with dual not reduced cost attribute
            def flow_upper_rule(m, i, j):
                return m.flows[i, j] <= m.flow_max[i, j]

            model.flow_upper = pyo.Constraint(model.lines, rule=flow_upper_rule)

            def FlowBalance_rule(m, node):
                return (
                    m.generation[node]
                    + sum(m.flows[i, node] for i in m.NodesIn[node])
                    - m.loads[node]
                    - sum(m.flows[node, j] for j in m.NodesOut[node])
                    == 0
                )

            model.FlowBalance = pyo.Constraint(model.buses, rule=FlowBalance_rule)

            if mode == 2:
                # DC-OPF Mode
                model.theta_max = pyo.Param(
                    model.buses, initialize=grid.theta_bounds_dict
                )

                def theta_max_rule(m, bus):
                    return (-m.theta_max[bus] - 1, m.theta_max[bus] + 1)

                model.angles = pyo.Var(model.buses, bounds=theta_max_rule)

                # Explicit lower bound constraint so dual accessed in with dual not reduced cost attribute
                def angles_lower_rule(m, bus):
                    return m.angles[bus] >= -m.theta_max[bus]

                model.angles_lower = pyo.Constraint(model.buses, rule=angles_lower_rule)

                # Explicit upper bound constraint so dual accessed in with dual not reduced cost attribute
                def angles_upper_rule(m, bus):
                    return m.angles[bus] <= m.theta_max[bus]

                model.angles_upper = pyo.Constraint(model.buses, rule=angles_upper_rule)
                model.susceptance = pyo.Param(
                    model.lines, initialize=grid.susceptance_dict
                )

                # what happens when flow on line is zero, this enforces a_i = a_j when f_{i,j} = 0
                def power_rule(m, *line):
                    return m.flows[line] == m.susceptance[line] * (
                        m.angles[line[1]] - m.angles[line[0]]
                    )

                model.PowerBalance = pyo.Constraint(model.lines, rule=power_rule)
        return model

    @staticmethod
    def create_root(grid):
        model = pyo.ConcreteModel()
        model.buses = pyo.Set(initialize=grid.buses)
        model.costs = pyo.Param(model.buses, initialize=grid.cost_dict)
        model.capacity = pyo.Param(model.buses, initialize=grid.gen_max_dict)

        def gen_max_rule(m, bus):
            return (-1, model.capacity[bus] + 1)

        model.generation = pyo.Var(
            model.buses, domain=pyo.NonNegativeReals, bounds=gen_max_rule
        )

        # model.generation = pyo.Var(
        #     model.buses, domain=pyo.PositiveReals, bounds=gen_max_rule
        # )

        # Explicit lower bound constraint so dual accessed in with dual not reduced cost attribute
        def gen_lower_rule(m, bus):
            return m.generation[bus] >= 0

        model.gen_lower = pyo.Constraint(model.buses, rule=gen_lower_rule)

        # Explicit upper bound constraint so dual accessed in with dual not reduced cost attribute
        def gen_upper_rule(m, bus):
            return m.generation[bus] <= m.capacity[bus]

        model.gen_upper = pyo.Constraint(model.buses, rule=gen_upper_rule)

        def obj_rule(m):
            return pyo.summation(m.costs, m.generation)

        model.obj = pyo.Objective(rule=obj_rule)

        # add unbounded var as eta placeholder since this is feasibility only
        # need to initialize to trivial value to avoid None issues
        model.eta = pyo.Var(initialize=0.0)
        return model

    @staticmethod
    def create_subproblem(root, grid, mode=2, feasibility_only=False):
        m = EnergyGrid.create_tiny_opf(grid=grid, mode=mode)
        # can either zero the subproblem objective and treat as combo-optimality/feasibility
        # or keep original objective and treat as feasibility only
        if not feasibility_only:
            m.obj = pyo.Objective(expr=0)
        complicating_vars_map = pyo.ComponentMap()
        for b in grid.buses:
            complicating_vars_map[root.generation[b]] = m.generation[b]
        # print('Done creating subproblem')
        return m, complicating_vars_map

    @staticmethod
    def setup_energy_grid(
        solver_name,
        CutGenerator=BendersGenerator_Serial,
        **kwargs,
    ):
        raise NotImplementedError(
            "The Energy Grid problem requires a persistent solver at present to enable feasibility cuts"
        )

    @staticmethod
    def setup_energy_grid_persistent(
        solver_name,
        CutGenerator=BendersGenerator_Serial,
        **kwargs,
    ):
        grid = kwargs.get("grid", EnergyGrid())
        m = EnergyGrid.create_root(grid=grid)
        transform = kwargs.get("transform", "standard_lp")
        root_vars = list(m.generation.values())
        m.benders = CutGenerator()
        m.benders.set_input(
            root_vars=root_vars,
            tol=1e-8,
            transform=transform,
            allow_infeasible=True,
        )
        m.benders.add_subproblem(
            subproblem_fn=EnergyGrid.create_subproblem,
            subproblem_fn_kwargs={
                "root": m,
                "grid": grid,
            },
            root_eta=m.eta,
            subproblem_solver=solver_name,
        )
        opt = pyo.SolverFactory(solver_name)
        opt.set_instance(m)
        return opt, m

    @staticmethod
    def run_energy_grid(
        mip_solver,
        transform="standard_lp",
        add_upper_bounds=False,
        include_print=False,
        is_persistent=False,
        grid=None,
    ):
        t0 = time.time()

        if grid is None:
            grid = EnergyGrid()

        setup_handle = EnergyGrid.setup_energy_grid
        if is_persistent:
            setup_handle = EnergyGrid.setup_energy_grid_persistent

        opt, m = setup_handle(
            solver_name=mip_solver,
            transform=transform,
            grid=grid,
        )

        if add_upper_bounds:
            m.eta.setub(100)
            m.eta.setlb(-100)

        if include_print:
            # print("{0:<15}{1:<15}{2:<15}{3:<15}".format("# Cuts","Gen 1", "Gen 2" "Total_Time"))
            print(
                "{0:<15}{1:<15}{2:<15}{3:<15}".format(
                    "# Cuts", "gen_bus1", "gen_bus2", "Total_Time"
                )
            )
        for i in range(30):
            if is_persistent:
                res = opt.solve(tee=False, save_results=False)
                cuts_added = m.benders.generate_cut()
                for c in cuts_added:
                    print(f"{c.name}, {c.expr}")
                    opt.add_constraint(c)
            else:
                res = opt.solve(m, tee=False)
                cuts_added = m.benders.generate_cut()
            if include_print:
                print(
                    "{0:<15}{1:<15.2f}{2:<15}{3:<15}".format(
                        len(cuts_added),
                        pyo.value(m.generation["bus1"]),
                        pyo.value(m.generation["bus2"]),
                        time.time() - t0,
                    )
                )
            if len(cuts_added) == 0:
                break

        return opt, m
