import pyomo.environ as pyo

from or_topas.benders import (
    BendersGenerator_Serial,
)
from or_topas.util.mymunch import MyMunch


class modified_absolute_value:
    @staticmethod
    def create_root():
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(None, None), initialize=0)
        m.eta = pyo.Var(bounds=(-10, None))
        m.obj = pyo.Objective(expr=m.eta)
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


class absolute_value:
    @staticmethod
    def create_root():
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(None, None), initialize=0)
        m.eta = pyo.Var(bounds=(-10, None))
        m.obj = pyo.Objective(expr=m.eta)
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
