#has to be invariant to dimension, set the axis for which to solve, different types of cost functions

import numpy as np
import models as model
from scipy.interpolate import RegularGridInterpolator


model
class DPSolver:
       def __init__(
        self,
        *,
        model: model.BaseModel,
        state_grids: list[np.ndarray],
        control_grid: np.ndarray,
        step_size: float,
        horizon: float,
        cost_func,
        terminal_cost,
        mode: str = "space",
    ):
        self.model = model
        self.state_grids = state_grids
        self.control_grid = control_grid
        self.step_size = step_size
        self.horizon = horizon
        self.cost_func = cost_func
        self.terminal_cost = terminal_cost
        self.mode = mode

        self.state_shape = tuple(len(g) for g in state_grids)
        self.J = np.full(self.state_shape, np.inf)
        self.U = np.zeros(self.state_shape)

        def _interpolator(self):
                return RegularGridInterpolator(self.state_grids, self.J, bounds_error=False, fill_value=np.inf)

        def _step(self, state, control):
            if self.mode == "time":
                return self.model.step_dt(state=state, control=control, dt=self.step_size)
            elif self.mode == "space":
                return self.model.step_ds(state=state, control=control, ds=self.step_size)
            else:
                raise ValueError("mode must be 'time' or 'space'")
            
        def initialize_terminal_cost(self):
            """Set terminal cost field J = terminal_cost(x)."""
            
            mesh = np.meshgrid(*self.state_grids, indexing="ij")
            state_points = np.stack([m.flatten() for m in mesh], axis=-1)
            terminal_values = np.apply_along_axis(self.terminal_cost, 1, state_points)
            self.J = terminal_values.reshape(self.state_shape)
    