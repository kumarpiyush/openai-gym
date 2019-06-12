import pandas as pd

class Observation :
    def __init__(self, obs) :
        self.cart_pos = obs[0]
        self.cart_vel = obs[1]
        self.pole_ang = obs[2]
        self.pole_vel = obs[3]

class Action :
    def __init__(self, action) :
        self.action = action

def convert_event_sequence_to_training_set(event_log) :
    cart_pos = []
    cart_vel = []
    pole_ang = []
    pole_vel = []
    label = []

    for i in range(1, len(event_log)-1) :
        if isinstance(event_log[i], Action) and isinstance(event_log[i-1], Observation) and isinstance(event_log[i+1], Observation) :
            st0 = event_log[i-1]
            st1 = event_log[i+1]
            ac = event_log[i]

            # RL doesn't take this (weak) supervision
            if abs(st1.pole_ang) <= abs(st0.pole_ang) or abs(st1.pole_vel) <= abs(st0.pole_vel) or st1.pole_ang * st0.pole_ang < 0 :
                cart_pos.append(st0.cart_pos)
                cart_vel.append(st0.cart_vel)
                pole_ang.append(st0.pole_ang)
                pole_vel.append(st0.pole_vel)
                label.append(ac.action)

    return pd.DataFrame({"cart_pos": cart_pos, "cart_vel": cart_vel, "pole_ang": pole_ang, "pole_vel": pole_vel}), label
