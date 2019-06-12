from sklearn.tree import DecisionTreeClassifier

class Player :
    def play(self, env, state) :
        raise NotImplementedError

class RandomPlayer(Player) :
    def play(self, env, state) :
        return env.action_space.sample()

class DecisionTreePlayer(Player) :
    def __init__(self) :
        self.model = DecisionTreeClassifier(max_depth=2)

    def fit(self, features, labels) :
        return self.model.fit(features, labels)

    def predict_proba(self, features) :
        return self.model.predict_proba(features)

    def play(self, env, state) :
        ac = self.predict_proba([[state.cart_pos, state.cart_vel, state.pole_ang, state.pole_vel]])[0]

        return 0 if ac[0] > ac[1] else 1
