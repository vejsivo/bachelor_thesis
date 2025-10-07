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
        optim_axis: int,
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
        self.optim_axis = optim_axis

        self.state_shape = tuple(len(g) for g in state_grids)
        self.J = np.full(self.state_shape, np.inf)
        self.U = np.zeros(self.state_shape)

    def _interpolator(self):
            return RegularGridInterpolator(self.state_grids, self.J, bounds_error=False, fill_value=np.inf)

    def _step(self, state, control):
        return self.model.step_ds(state=state, control=control, ds=self.step_size)
    
        """if self.mode == "time":
            return self.model.step_dt(state=state, control=control, dt=self.step_size)
        elif self.mode == "space":
            return self.model.step_ds(state=state, control=control, ds=self.step_size)
        else:
            raise ValueError("mode must be 'time' or 'space'")
        """
            
    def initialize_terminal_cost(self):
        mesh = np.meshgrid(*self.state_grids, indexing="ij")
        state_points = np.stack([m.flatten() for m in mesh], axis=-1)
        terminal_values = np.apply_along_axis(self.terminal_cost, 1, state_points)
        self.J = terminal_values.reshape(self.state_shape)

    def generate_state_list(self, state_grids: list[np.ndarray], axis: int):
        """returns list containing all states sorted by the given axis as per the state grid ordering"""

        mesh = np.meshgrid(*state_grids, indexing="ij")
        all_states = np.stack([m.flatten() for m in mesh], axis=-1)

        # Normalize axis index (convert -1 to the actual position)
        axis = axis % len(state_grids)

        # Sort descending along that coordinate
        sort_idx = np.argsort(all_states[:, axis])[::-1]
        all_states = all_states[sort_idx]
        return all_states

        

    def backward_sweep(self):
        """Perform dynamic programming backward recursion."""
        self.initialize_terminal_cost()
        all_states = self.generate_state_list(state_grids = self.state_grids, axis = self.optim_axis)
        n_steps = int(self.horizon / self.step_size + 1) #how many steps back in the optimizing dimension we go (generally max_axis_value + 1)
        
        for k in range(n_steps, -1, -1):
            J_interp = self._interpolator()
            new_J = np.full(self.state_shape, np.inf)
            new_U = np.zeros(self.state_shape)

            # Iterate through all grid states
            for idx, x in enumerate(all_states):
                best_cost = np.inf
                best_u = 0.0

                for u in self.control_grid:
                    x_next = self._step(x, u)
                    Jn = J_interp(x_next)
                    stage_cost = self.cost_func(x, u, self.step_size)
                    total_cost = stage_cost + Jn

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_u = u

                new_J.flat[idx] = best_cost
                new_U.flat[idx] = best_u

            self.J = new_J
            self.U = new_U
        return self.J, self.U