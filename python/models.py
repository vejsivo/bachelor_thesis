import numpy as np

class BaseModel:
    def __init__(self, *, params: dict):
        self.params = params

    def f_dt(self, *, state: np.ndarray, control: float) -> np.ndarray:
        """implemented in subclasses"""
        pass
    
    def f_ds(self, *, state: np.ndarray, control: float) -> np.ndarray:
        """implemented in subclasses"""
        pass

    def step_dt(self, *, state: np.ndarray, control: float, dt: float) -> np.ndarray:
        """state transition function in time"""
        return state + dt * self.f_dt(state=state, control=control)
    
    def step_ds(self, *, state: np.ndarray, control: float, ds: float) -> np.ndarray:
        """state transition function in distance"""
        return state + ds * self.f_ds(state=state, control=control)


class DragPointMass(BaseModel):
    def f_dt(self, *, state: np.ndarray, control: float) -> np.ndarray:
        
        s, v = state
        m = self.params["mass"]
        CdA = self.params["drag_coef"]
        u = control

        dsdt = v
        dvdt = (u - CdA * v**2) / m
        return np.array([dsdt, dvdt])

    def f_ds(self, *, state: np.ndarray, control: float) -> np.ndarray:
        s, v = state
        m = self.params["mass"]
        CdA = self.params["drag_coef"]
        u = control

        if abs(v) < 1e-9:
            v = 1e-9

        dsds = 1.0
        dvds = (u - CdA * v**2) / (m * v)
        return np.array([dsds, dvds])

