from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import Screen
from kivy.properties import ListProperty, BooleanProperty, ObjectProperty, NumericProperty, StringProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.metrics import dp
import numpy as np
from enum import Enum
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec
from tf_agents.specs import array_spec, TensorSpec
import tensorflow as tf
from tf_agents.agents import SacAgent
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.policies.policy_saver import PolicySaver
from typing import Tuple, List, Callable, Optional
from math import atan2, degrees
import os


class GameS(Screen):
    def __init__(self, **kwargs):
        super(GameS, self).__init__(**kwargs)
        App.get_running_app().gs = self


class GameScreen(BoxLayout):

    def __init__(self, **kwargs):
        super(GameScreen, self).__init__(**kwargs)
        App.get_running_app().gm = self


class Pitch(FloatLayout):

    n_score = NumericProperty(0)
    c_score = NumericProperty(0)
    down = NumericProperty(1)
    start_point = ListProperty()

    dict = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}
    op_coach = StringProperty()
    op_rating = NumericProperty()

    def __init__(self, **kwargs):
        super(Pitch, self).__init__(**kwargs)
        App.get_running_app().pitch = self
        self.size_hint = None, None
        self.players = []
        self.gamers = []
        self.in_play = False
        self.noughts_attack = True
        self.run_play = False
        self.start_point = [self.width / 2, self.width * 0.0625]
        self.nought = True
        self.fumble = False
        self.run_dist = self.height * 0.25
        self.throw_dist = self.height * 0.5
        self.OoB = self.width * 0.0375
        self.TD = self.width * 0.0625
        self.trick_play = False
        self.game_state = []
        self.moves = []
        self.move_clock = 0

    def get_state(self):
        game_state = [[x.center_x/self.width, x.center_y/self.height, 0 if (not x.possession if isinstance(x, Player) else isinstance(x, Ball)) else 1, 0] for x in self.gamers]
        down = [0, 0, 0, 0]
        down[self.down - 1] = 1
        game_state.append(down)
        run_play = [0, 0, 0, 0]
        run_play[0] = int(self.run_play)
        game_state.append(run_play)
        game_state = np.array(game_state)
        return game_state

    def reset(self):
        Clock.unschedule(self.update)
        for x in self.gamers:
            Clock.unschedule(x.update)
        App.get_running_app().gs.remove_widget(App.get_running_app().gm)
        pitch = GameScreen()
        App.get_running_app().gs.add_widget(pitch)

    def on_parent(self, *largs):
        if self.parent:
            self.parent.bind(size=self._parent_resize)
            self._parent_resize()
        # todo: handle removal from parent if needed

    def _parent_resize(self, *largs):
        max_width, max_height = self.parent.size
        height = max_width * (16 / 9)
        # Best case scenario, free verical space for 16:9 aspect
        if height <= max_height:
            self.width = max_width
            self.height = height

        else:
            self.height = max_height
            self.width = max_height * (9 / 16)

        self.run_dist = self.height * 0.25
        self.throw_dist = self.height * 0.5
        self.OoB = self.width * 0.0375
        self.TD = self.width * 0.0625
        self.start_point = [self.width / 2, self.width * 0.0625]

    def on_children(self, instance, value):
        self.players = [x for x in self.children if isinstance(x, Player)]
        self.gamers = self.players + [x for x in self.children if isinstance(x, Ball)]

    def set_cross_moves(self, moves):

        parent = App.get_running_app().gm
        ball = App.get_running_app().ball
        moves = moves.tolist()

        cross1 = parent.ids.cross1
        cross2 = parent.ids.cross2
        cross3 = parent.ids.cross3
        cross4 = parent.ids.cross4
        cross5 = parent.ids.cross5

        nought1 = parent.ids.nought1
        nought2 = parent.ids.nought2
        nought3 = parent.ids.nought3
        nought4 = parent.ids.nought4
        nought5 = parent.ids.nought5
        playlist = [nought1, nought2, nought3, nought4, nought5, ball, cross1, cross2, cross3, cross4, cross5]
        n = 0
        for x in playlist:
            if moves[0][n] < 0 and moves[0][n+1] < 0:
                x.destination = []
                x.moving = False
            else:
                if moves[0][n] >= 0:
                    x.destination = [moves[0][n] * self.width, 0]
                else:
                    x.destination = [x.center_x, 0]
                if moves[0][n+1] >= 0:
                    x.destination[1] = moves[0][n+1] * self.height
                else:
                    x.destination[1] = x.center_y
            n += 2

        if not any(x.possession for x in self.players):
            ball.destination = []
            ball.moving = False

        if ball.destination:
            if self.run_play:
                if Vector(ball.center).distance(ball.destination) < self.run_dist:
                    if 0 < degrees(atan2(ball.destination[1] - ball.center_y, ball.destination[0] - ball.center_y)) >= 0:
                        ball.destination = [ball.destination[0], ball.center_y]

                else:
                    if degrees(atan2(ball.destination[1] - ball.center_y, ball.destination[0] - ball.center_y)) >= 0:
                        direction = (Vector(ball.destination[0], ball.center_y) - Vector(ball.center)).normalize()
                        ball.destination = Vector(ball.center) + direction * self.run_dist

            else:
                if Vector(ball.center).distance(ball.destination) > self.throw_dist:

                    direction = (Vector(ball.destination) - Vector(ball.center)).normalize()
                    ball.destination = Vector(ball.center) + direction * self.throw_dist

        for x in self.players:

            if x.destination:

                if Vector(x.center).distance(x.destination) > self.run_dist:
                    direction = (Vector(x.destination) - Vector(x.center)).normalize()
                    x.destination = Vector(x.center) + direction * self.run_dist

                if ball.destination and x.possession:
                    x.destination = []
                    x.moving = False
        self.play()

    def tick_move_clock(self, dt):

        self.move_clock += 1
        if self.move_clock >= 20:
            for x in self.gamers:
                x.moving = False

    def play(self):
        self.in_play = True

        ball = App.get_running_app().ball

        for x in self.children:
            if x.destination:
                x.calculate_direction()
                x.moving = True
            if isinstance(x, Player):
                if x.possession:
                    if ball.destination:
                        x.possession = False
                        x.thrower = True
                else:
                    pass

        if ball.destination and not self.run_play:
            if self.noughts_attack:
                if (ball.direction[1] < 0) and ball.center_y < (self.start_point[1] + self.TD * 3):
                    self.trick_play = True
            else:
                if (ball.direction[1] > 0) and (ball.center_y > self.start_point[1] - self.TD * 3):
                    self.trick_play = True

        Clock.schedule_interval(self.update, 0.003)
        self.move_clock = 0
        Clock.schedule_interval(self.tick_move_clock, 1)

    def update(self, dt):
        env = App.get_running_app().env
        start_dist =  self.width * 0.25
        # start of play-loop
        if self.in_play:

            # incomplete passes
            ball = App.get_running_app().ball

            if not ball.destination and not any(x.possession for x in self.players):
                if not self.run_play and not self.trick_play and not self.fumble:

                    for x in self.gamers:
                        x.destination = []
                        x.moving = False

                    Clock.unschedule(self.update)
                    if self.down < 4:

                        self.down = self.down + 1
                        Clock.schedule_once(self.set_up, 0)
                        App.get_running_app().set_result_and_step(ActionState.DOWN_INCREASE,
                                                                  self.get_state(),
                                                                  self.moves)
                        return

                    else:
                        # App.get_running_app().response = ActionState.CHANGE_POSSESSION
                        # env._state = self.get_state()
                        # env.step(self.moves)
                        App.get_running_app().set_result_and_step(ActionState.CHANGE_POSSESSION,
                                                                  self.get_state(),
                                                                  self.moves)
                        self.reset()
                        return

                # fumble
                else:
                    if ball.pos[0] <= self.OoB or ball.right > self.width - self.OoB:
                        self.trick_play = False
                        for x in self.gamers:
                            x.destination = []
                            x.moving = False

                        Clock.unschedule(self.update)
                        if self.down < 4:

                            self.down = self.down + 1
                            # App.get_running_app().response = ActionState.DOWN_INCREASE
                            # env._state = self.get_state()
                            # env.step(self.moves)
                            if self.start_point[1] > start_dist:
                                reward = ball.center_y - self.start_point[1]
                            else:
                                reward = ball.center_y - start_dist
                            App.get_running_app().set_result_and_step(ActionState.DOWN_INCREASE,
                                                                      self.get_state(),
                                                                      self.moves, reward=reward)
                            self.get_start_point(True)
                            return

                        else:

                            # App.get_running_app().response = ActionState.CHANGE_POSSESSION
                            # env._state = self.get_state()
                            # env.step(self.moves)
                            App.get_running_app().set_result_and_step(ActionState.CHANGE_POSSESSION,
                                                                      self.get_state(),
                                                                      self.moves)
                            self.reset()
                            return

                    else:
                        if not self.fumble:
                            self.fumble = True
                            self.trick_play = False

            # Out of bounds throw - (to stop being able to catch out of bounds and gain territory)
            if ball.destination and ball.pos[0] < self.OoB or ball.right > self.width - self.OoB:
                self.trick_play = False
                for x in self.gamers:
                    x.destination = []
                    x.moving = False

                Clock.unschedule(self.update)
                if self.down < 4:

                    self.down = self.down + 1
                    # App.get_running_app().response = ActionState.DOWN_INCREASE
                    # env._state = self.get_state()
                    # env.step(self.moves)
                    App.get_running_app().set_result_and_step(ActionState.DOWN_INCREASE,
                                                              self.get_state(),
                                                              self.moves)
                    self.get_start_point(True)
                    return

                else:

                    # App.get_running_app().response = ActionState.CHANGE_POSSESSION
                    # env._state = self.get_state()
                    # env.step(self.moves)
                    App.get_running_app().set_result_and_step(ActionState.CHANGE_POSSESSION,
                                                              self.get_state(),
                                                              self.moves)
                    self.reset()
                    return

            for x in self.players:
                # run plays
                if not self.run_play:
                    if self.noughts_attack:
                        if x.possession and x.center_y > self.start_point[1] + self.TD * 3:
                            self.run_play = True

                    else:
                        if x.possession and x.center_y < self.start_point[1] - self.TD * 3:
                            self.run_play = True

                        else:
                            self.run_play = False
                # out of bounds
                if x.possession and (x.x < self.OoB or x.right > self.width - self.OoB):

                    for y in self.gamers:
                        y.moving = False
                        y.destination = []
                    Clock.unschedule(self.update)
                    if self.down < 4:

                        self.down = self.down + 1
                        self.get_start_point(True)
                        # App.get_running_app().response = ActionState.DOWN_INCREASE
                        # env._state = self.get_state()
                        # env.step(self.moves)
                        if self.start_point[1] > start_dist:
                            reward = ball.center_y - self.start_point[1]
                        else:
                            reward = ball.center_y - start_dist
                        App.get_running_app().set_result_and_step(ActionState.DOWN_INCREASE,
                                                                  self.get_state(),
                                                                  self.moves, reward=reward)

                    else:

                        # App.get_running_app().response = ActionState.CHANGE_POSSESSION
                        # env._state = self.get_state()
                        # env.step(self.moves)
                        App.get_running_app().set_result_and_step(ActionState.CHANGE_POSSESSION,
                                                                  self.get_state(),
                                                                  self.moves)
                        self.reset()

                # touch downs
                elif self.noughts_attack:
                    if x.possession and x.y > self.top - self.TD - x.width/2 and isinstance(x, Nought):
                        Clock.unschedule(self.update)
                        self.game_ended = True
                        # App.get_running_app().response = ActionState.TOUCHDOWN
                        # env._state = self.get_state()
                        # env.step(self.moves)
                        App.get_running_app().set_result_and_step(ActionState.TOUCHDOWN,
                                                                  self.get_state(),
                                                                  self.moves)
                        self.reset()

            # stop loop if everyone stopped moving

            if not any(x.moving for x in self.gamers):
                self.in_play = False
                Clock.unschedule(self.update)
                # App.get_running_app().response = ActionState.VALID_MOVE
                # env._state = self.get_state()
                # env.step(self.moves)
                if any(x.possession for x in self.players):
                    if self.start_point[1] > start_dist:
                        reward = ball.center_y - self.start_point[1]
                    else:
                        reward = ball.center_y - start_dist
                else:
                    reward = None
                App.get_running_app().set_result_and_step(ActionState.VALID_MOVE,
                                                          self.get_state(),
                                                          self.moves, reward=reward)

    def set_up(self, dt):
        self.in_play = False
        self.run_play = False
        start_dist = self.width * 0.25
        for x in self.gamers:
            x.moving = False
            x.destination = []
            if isinstance(x, Player):
                x.possession = False
        if self.noughts_attack:
            if self.start_point[1] > self.top - start_dist:
                self.start_point[1] = self.top - start_dist
        if not self.noughts_attack:
            if self.start_point[1] < start_dist:
                self.start_point[1] = start_dist
        parent = App.get_running_app().gm
        nought1 = parent.ids.nought1
        nought2 = parent.ids.nought2
        nought3 = parent.ids.nought3
        nought4 = parent.ids.nought4
        nought5 = parent.ids.nought5
        cross1 = parent.ids.cross1
        cross2 = parent.ids.cross2
        cross3 = parent.ids.cross3
        cross4 = parent.ids.cross4
        cross5 = parent.ids.cross5
        ball = App.get_running_app().ball
        dist = nought1.width * 1.67
        dist2 = nought1.width * 5

        if self.noughts_attack:

            nought1.pos = self.width / 2 - nought1.width / 2, \
                          self.start_point[1] - nought1.height / 2

            nought2.pos = self.width / 2 - nought2.width / 2 - dist, \
                self.start_point[1] - nought2.height / 2 + dist

            nought3.pos = self.width / 2 - nought3.width / 2 + dist, \
                self.start_point[1] - nought3.height / 2 + dist

            nought4.pos = nought4.width * 2, \
                self.start_point[1] - nought3.height / 2 + dist

            nought5.pos = self.width - nought5.width * 2.5, \
                self.start_point[1] - nought3.height / 2 + dist

            cross1.pos = self.width / 2 - cross1.width / 2, \
                self.start_point[1] + dist2 - cross1.height / 2

            cross2.pos = self.width / 2 - cross2.width / 2 - dist, \
                self.start_point[1] + dist2 - cross2.height / 2

            cross3.pos = self.width / 2 - cross3.width / 2 + dist, \
                self.start_point[1] + dist2 - cross3.height / 2

            cross4.pos = cross4.width * 2, \
                self.start_point[1] + dist2 - cross3.height / 2

            cross5.pos = self.width - cross4.width * 2.5, \
                self.start_point[1] + dist2 - cross3.height / 2

            ball.pos = self.width / 2 - nought1.width / 2, \
                self.start_point[1] - nought1.height / 2

            nought1.possession = True

    def get_start_point(self, set_up):
        rel_height = self.width * 0.0625
        big_rel_height = self.width * 0.25
        ball = App.get_running_app().ball

        if self.noughts_attack:
            new_start_point = [ball.center_x, ball.center_y - (App.get_running_app().gm.ids.nought1.width * 2)]
            if rel_height < new_start_point[1] < self.top-big_rel_height:
                self.start_point = new_start_point

            elif new_start_point[1] <= rel_height:
                self.start_point = [new_start_point[0], rel_height]

            elif new_start_point[1] >= self.top - big_rel_height:
                self.start_point = [new_start_point[0], self.top - big_rel_height]

        if set_up:
            self.set_up(0)


class Player(Widget):
    drawing = BooleanProperty(False)
    moving = BooleanProperty(False)
    destination = ListProperty([])
    direction = ListProperty([])
    possession = BooleanProperty(False)
    thrower = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(Player, self).__init__(**kwargs)

    def calculate_direction(self):
        if self.destination:
            if self.destination[0] <= 0 + self.width / 1.6:
                self.destination[0] = 0 + self.width / 1.6
            if self.destination[0] >= App.get_running_app().pitch.width - self.width / 1.6:
                self.destination[0] = App.get_running_app().pitch.width - self.width / 1.6
            if self.destination[1] <= 0 + self.width / 1.6:
                self.destination[1] = 0 + self.width / 1.6
            if self.destination[1] >= App.get_running_app().pitch.top - self.width / 1.6:
                self.destination[1] = App.get_running_app().pitch.top - self.width / 1.6

            direction = Vector(self.destination) - Vector(self.center)
            self.direction = direction.normalize() * self.parent.width * 0.004

    def catch(self, ball):
        players = [x for x in App.get_running_app().pitch.children if isinstance(x, Player)]
        pitch = App.get_running_app().pitch
        env = App.get_running_app().env
        if not self.thrower:
            # if not any(x.possession for x in players):
                if self.collide_widget(ball) and (ball.moving or pitch.fumble):
                    self.possession = True
                    ball.destination = []
                    ball.moving = False
                    if pitch.trick_play:
                        pitch.trick_play = False
                    else:
                        pitch.run_play = True
                        pitch.fumble = False
                    for x in players:
                        x.thrower = False

                    if pitch.noughts_attack:
                        if isinstance(self, Cross):

                            #env.response = ActionState.CHANGE_POSSESSION
                            #env._state = pitch.get_state()
                            App.get_running_app().set_result_and_step(ActionState.CHANGE_POSSESSION,
                                                                      pitch.get_state(),
                                                                      pitch.moves)
                            pitch.reset()
                            #env.step(pitch.moves)

                    else:
                        if isinstance(self, Nought):
                            # pitch.run_play = False
                            pitch.noughts_attack = True

                    return True
                else:
                    return False

    def tackle(self):

        pitch = App.get_running_app().pitch
        start_dist = pitch.width * 0.25
        env = App.get_running_app().env
        ball = App.get_running_app().ball
        for x in pitch.players:

            if self.collide_widget(x) and x != self:

                if ((x.__class__ != self.__class__) and x.possession) or \
                        ((x.__class__ != self.__class__) and self.possession):
                    #
                    # x.moving = False
                    # x.destination = []
                    # self.moving = False
                    # self.destination = []
                    for y in pitch.players:
                        y.moving = False
                        y.destination = []
                    pitch.run_play = False
                    pitch.in_play = False
                    Clock.unschedule(pitch.update)
                    if pitch.down < 4:
                        pitch.down += 1

                        # env.response = ActionState.DOWN_INCREASE
                        # env._state = pitch.get_state()
                        # env.step(pitch.moves)
                        if pitch.start_point[1] > start_dist:
                            reward = ball.center_y - pitch.start_point[1]
                        else:
                            reward = ball.center_y - start_dist
                        App.get_running_app().set_result_and_step(ActionState.DOWN_INCREASE,
                                                                  pitch.get_state(),
                                                                  pitch.moves, reward=reward)
                        pitch.get_start_point(True)
                    else:
                        # env.response = ActionState.CHANGE_POSSESSION
                        # env._state = self.get_state()
                        # env.step(pitch.moves)
                        App.get_running_app().set_result_and_step(ActionState.CHANGE_POSSESSION,
                                                                  pitch.get_state(),
                                                                  pitch.moves)
                        pitch.reset()

                else:

                    x.moving = True
                    direction = (Vector(self.center) - Vector(x.center)).normalize()
                    self.destination = Vector(self.center) + direction * self.width
                    self.calculate_direction()
                    x.destination = Vector(x.center) + direction * - self.width
                    x.calculate_direction()

    def update(self, dt):

        if self.possession:
            App.get_running_app().ball.center = Vector(self.center) + [self.width * 0.43, 0]
        else:

            self.catch(App.get_running_app().ball)

        if self.moving:
            pitch = App.get_running_app().pitch
            self.tackle()
            if self.moving:
                if Vector(self.center).distance(self.destination) > dp(3):
                    self.center = Vector(self.direction) + self.center

                else:
                    self.center = self.destination
                    self.destination = []
                    self.direction = []
                    self.moving = False
                    self.thrower = False


class Cross(Player):

    def __init__(self, **kwargs):
        super(Player, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 0.003)


class Nought(Player):
    def __init__(self, **kwargs):
        super(Player, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 0.003)


class Ball(Widget):
    destination = ListProperty([])
    moving = BooleanProperty(False)
    drawing = BooleanProperty(False)
    direction = ListProperty([])
    angle = NumericProperty(0)

    def __init__(self, **kwargs):
        super(Ball, self).__init__(**kwargs)
        App.get_running_app().ball = self
        Clock.schedule_interval(self.update, 0.003)

    def calculate_direction(self):
        if self.destination:
            if self.destination[0] <= 0 + self.width / 1.6:
                self.destination[0] = 0 + self.width / 1.6
            if self.destination[0] >= App.get_running_app().pitch.width - self.width / 1.6:
                self.destination[0] = App.get_running_app().pitch.width - self.width / 1.6
            if self.destination[1] <= 0 + self.width / 1.6:
                self.destination[1] = 0 + self.width / 1.6
            if self.destination[1] >= App.get_running_app().pitch.top - self.width / 1.6:
                self.destination[1] = App.get_running_app().pitch.top - self.width / 1.6
            direction = Vector(self.destination) - Vector(self.center)
            self.direction = direction.normalize() * self.parent.width * 0.008

    def update(self, dt):

        if self.moving:

            self.angle = (degrees(atan2(self.destination[1] - self.center_y, self.destination[0] - self.center_x))
                          % 360) - 90 if self.destination else 0

            if Vector(self.center).distance(self.destination) > dp(4):
                self.center = Vector(self.direction) + self.center

            else:
                self.center = self.destination
                self.destination = []
                self.direction = []
                self.moving = False
                for x in App.get_running_app().pitch.children:
                    if isinstance(x, Player):
                        x.thrower = False


class ActionState(Enum):
    VALID_MOVE = 1
    ILLEGAL_MOVE = 2
    CHANGE_POSSESSION = 3
    DOWN_INCREASE = 4
    TOUCHDOWN = 5
    GAME_COMPLETE = 6


class NoughtEnv(PyEnvironment):

    def __init__(self, game):
        self._action_spec = array_spec.BoundedArraySpec((22,), np.float32, minimum=-0.5, maximum=1, name='moves')
        self._observation_spec = array_spec.BoundedArraySpec((13, 4), np.float32, minimum=0, maximum=1, name='field')
        self._episode_ended = False
        self.game = game
        self._state = [[0.9, 0.18759375, 0., 0.],
                       [0.125, 0.18759375, 0., 0.],
                       [0.5835, 0.18759375, 0., 0.],
                       [0.4165, 0.18759375, 0., 0.],
                       [0.5, 0.18759375, 0., 0.],
                       [0.9, 0.0939375, 0., 0.],
                       [0.125, 0.0939375, 0., 0.],
                       [0.5835, 0.0939375, 0., 0.],
                       [0.4165, 0.0939375, 0., 0.],
                       [0.5, 0.04696875, 1., 0.],
                       [0.5215, 0.04696875, 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 0., 0., 0.]]

        self.response = ActionState.VALID_MOVE
        self.count = 0
        App.get_running_app().env = self
    #
    # def action_spec(self):
    #     return self._action_spec

    def action_spec(self):

        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        print(f'===\n ended. resetting\n===')

        self.count = 0
        self._state = [[0.9, 0.18759375, 0., 0.],
                       [0.125, 0.18759375, 0., 0.],
                       [0.5835, 0.18759375, 0., 0.],
                       [0.4165, 0.18759375, 0., 0.],
                       [0.5, 0.18759375, 0., 0.],
                       [0.9, 0.0939375, 0., 0.],
                       [0.125, 0.0939375, 0., 0.],
                       [0.5835, 0.0939375, 0., 0.],
                       [0.4165, 0.0939375, 0., 0.],
                       [0.5, 0.04696875, 1., 0.],
                       [0.5215, 0.04696875, 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 0., 0., 0.]]

        self.response = ActionState.VALID_MOVE
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):

        self.response = self.game.response
        self._state = self.game.state
        self.count += 1
        # if self._episode_ended:
        #     return self.reset()

        if self.count >= 30:
            Clock.schedule_once(App.get_running_app().train_env, 0)
            self._episode_ended = True
            return ts.termination(np.array(self.game.state, dtype=np.float32), -90)

        if self.game.response == ActionState.VALID_MOVE:
            Clock.schedule_once(App.get_running_app().train_env, 0)
            return ts.transition(np.array(self.game.state, dtype=np.float32), self.game.reward)

        if self.game.response == ActionState.DOWN_INCREASE:
            self.game.episode_time = 0
            Clock.schedule_once(App.get_running_app().train_env, 0)
            return ts.transition(np.array(self.game.state, dtype=np.float32), self.game.reward)

        if self.game.response == ActionState.CHANGE_POSSESSION:
            Clock.schedule_once(App.get_running_app().train_env, 0)
            self._episode_ended = True
            return ts.termination(np.array(self.game.state, dtype=np.float32), -100)

        if self.response == ActionState.TOUCHDOWN:
            Clock.schedule_once(App.get_running_app().train_env, 0)
            self._episode_ended = True
            return ts.termination(np.array(self.game.state, dtype=np.float32), 100)


class IMAgent(SacAgent):

    def __init__(self,
                 env: TFPyEnvironment,
                 observation_spec: TensorSpec = None,
                 action_spec: TensorSpec = None,
                 reward_fn: Callable = lambda time_step: time_step.reward,
                 action_fn: Callable = lambda action: action,
                 name: str = 'IMAgent',
                 replay_buffer_max_length: int = 10000,
                 learning_rate: float = 1e-5,
                 training_batch_size: int = 30,
                 training_parallel_calls: int = 3,
                 training_prefetch_buffer_size: int = 3,
                 training_num_steps: int = 2,
                 **dqn_kwargs):

        self._env = env
        self._reward_fn = reward_fn
        self._name = name
        self._observation_spec = observation_spec or self._env.observation_spec()

        self._action_spec = action_spec or self._env.action_spec()

        self._action_fn = action_fn
        self.actor_network = self._build_actor_net()
        self.critic_network = self._build_critic_net()
        act_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        crit_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        env_ts_spec = self._env.time_step_spec()
        time_step_spec = TimeStep(
            step_type=env_ts_spec.step_type,
            reward=env_ts_spec.reward,
            discount=env_ts_spec.discount,
            observation=self.actor_network.input_tensor_spec
        )

        super().__init__(time_step_spec,
                         self._action_spec,
                         self.critic_network,
                         self.actor_network,
                         act_optimizer,
                         crit_optimizer,
                         alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                             learning_rate=learning_rate),
                         **dqn_kwargs)

        self._policy_state = self.policy.get_initial_state(
            batch_size=self._env.batch_size)
        self._rewards = []

        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=self.collect_data_spec,
            batch_size=self._env.batch_size,
            max_length=replay_buffer_max_length)

        self._training_batch_size = training_batch_size
        self._training_parallel_calls = training_parallel_calls
        self._training_prefetch_buffer_size = training_prefetch_buffer_size
        self._training_num_steps = training_num_steps

    def _build_actor_net(self):
        fc_layer_params = (50, 200, 50)

        network = ActorDistributionNetwork(
            self._observation_spec,
            self._action_spec,
            fc_layer_params=fc_layer_params)

        network.create_variables()
        network.summary()

        return network

    def _build_critic_net(self):

        network = CriticNetwork(
            (self._observation_spec, self._action_spec))

        network.create_variables()
        network.summary()

        return network

    def reset(self):
        self._policy_state = self.policy.get_initial_state(
            batch_size=self._env.batch_size
        )
        self._rewards = []

    def _augment_time_step(self, time_step: TimeStep) -> TimeStep:

        reward = self._reward_fn(time_step)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        if reward.shape != time_step.reward.shape:
            reward = tf.reshape(reward, time_step.reward.shape)

        observation = self._observation_fn(time_step.observation)

        return TimeStep(
            step_type=time_step.step_type,
            reward=reward,
            discount=time_step.discount,
            observation=observation
        )

    def _observation_fn(self, observation: tf.Tensor) -> tf.Tensor:
        """
            Takes a tensor with specification self._env.observation_spec
            and extracts a tensor with specification self._observation_spec.

            For example, consider an agent within an NxN maze environment.
            The env could expose the entire NxN integer matrix as an observation
            but we would prefer the agent to only see a 3x3 window around their
            current location. To do this we can override this method.

            This allows us to have different agents acting in the same environment
            with different observations.
        """
        return observation

    def train_iteration(self) -> LossInfo:
        experience, info = self._replay_buffer.get_next(
            sample_batch_size=self._training_batch_size,
            num_steps=self._training_num_steps
        )

        train = self.train(experience)
        return train


def cross_reward_fn(ts: TimeStep) -> float:
    if ts.reward == -9:
        return ts.reward
    else:
        return ts.reward * -1


class FootballApp(App):
    ball = ObjectProperty()
    pitch = ObjectProperty()
    gm = ObjectProperty()
    gs = ObjectProperty()
    env = ObjectProperty()
    noughts = ObjectProperty()
    crosses = ObjectProperty()
    response = None
    state = None
    reward = None
    n_actions = None
    c_actions = None
    time_step = None
    next_step = None
    training_episode = 0
    episodes = 0
    episode_time = 0
    episode_clock_ticking = False
    training_checkpoint = int

    def build(self):

        self.env = NoughtEnv(self)
        self.env = TFPyEnvironment(self.env)
        n_arr_spec = BoundedTensorSpec.from_spec(array_spec.BoundedArraySpec((12,), np.float32, minimum=0, maximum=1.2))
        c_arr_spec = BoundedTensorSpec.from_spec(array_spec.BoundedArraySpec((10,), np.float32, minimum=0, maximum=1.2))
        self.noughts = IMAgent(self.env, action_spec=n_arr_spec, name='n')
        self.crosses = IMAgent(self.env, action_spec=c_arr_spec, reward_fn=cross_reward_fn, name='c')
        print('initial reset:')
        self.env.reset()
        Clock.schedule_once(self.train_env, 12)
        return GameS()

    def train_env(self, dt):

        self.time_step = self.env.current_time_step()
        if not self.time_step.is_last():
            self.n_actions = self.noughts.policy.action(self.time_step, self.noughts._policy_state)
            n_moves = np.array(self.n_actions[0]).reshape(12)
            self.c_actions = self.crosses.policy.action(self.time_step, self.crosses._policy_state)
            c_moves = np.array(self.c_actions[0]).reshape(10)
            pitch_action = np.concatenate((n_moves, c_moves))
            pitch_action = pitch_action.reshape((1, 22))
            actions = self.n_actions.replace(action=pitch_action)
            self.pitch.moves = actions
            self.pitch.set_cross_moves(pitch_action)

        else:
            self.episodes += 1
            print(f'episodes completed: {self.episodes}')

            if self.episodes % 1000 != 0:
                self.pitch.reset()
                print('new episode reset:')
                self.env.reset()
                self.crosses.reset()
                self.noughts.reset()
                Clock.schedule_once(self.train_env, 0)
            else:
                print('training')
                self.training_checkpoint = self.episodes * 3
                self.train_agents(0)

    def set_result_and_step(self, response, state, policy_step, reward=None):
        Clock.unschedule(self.pitch.tick_move_clock)
        self.reward = reward / 100 if reward else 0
        self.response = response
        self.state = state
        if reward:
            print(self.reward)
        try:
            self.next_step = self.env.step(policy_step)
            self.collect_data()
        except ValueError:
             print('error: restarting episode')
             self.pitch.reset()
             self.env.reset()
             self.crosses.reset()
             self.noughts.reset()
             Clock.schedule_once(self.train_env, 0)

    def collect_data(self):
        n_traj = trajectory.from_transition(self.time_step,
                                            self.n_actions,
                                            self.next_step)
        c_traj = trajectory.from_transition(self.crosses._augment_time_step(self.time_step),
                                            self.c_actions,
                                            self.crosses._augment_time_step(self.next_step))
        self.noughts._replay_buffer.add_batch(n_traj)
        self.crosses._replay_buffer.add_batch(c_traj)

    def train_agents(self, dt):

        n_train = self.noughts.train_iteration()
        c_train = self.noughts.train_iteration()
        self.training_episode += 1
        if self.training_episode == self.training_checkpoint:
            PolicySaver(self.noughts.policy).save('models3/noughts_policy' + str(self.episodes))
            PolicySaver(self.crosses.policy).save('models3/crosses_policy' + str(self.episodes))
            print(f'saving {self.training_checkpoint}')
            n_check = tf.train.Checkpoint(actor_net=self.noughts.actor_network, value_net=self.noughts.critic_network)
            c_check = tf.train.Checkpoint(actor_net=self.crosses.actor_network, value_net=self.crosses.critic_network)
            n_check.save('models3/noughts_check' + str(self.training_episode))
            c_check.save('models3/crosses_check' + str(self.training_episode))
            print('saved')
            self.pitch.reset()
            print('training reset:')
            self.env.reset()
            self.crosses.reset()
            self.noughts.reset()
            Clock.schedule_once(self.train_env, 0)
            self.training_episode = 0
        else:
            Clock.schedule_once(self.train_agents, 0)


App = FootballApp().run()
