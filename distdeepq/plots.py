import numpy as np
import matplotlib.pyplot as plt
# from distdeepq.static import build_z
import tensorflow as tf


class PlotMachine:

    def __init__(self, dist_params, nb_actions, action_set=None):

        # self.z, self.dz = build_z(numpy=True, **dist_params)

        plt.ion()

        self.fig, self.ax = plt.subplots()

        # self.bars = [self.ax.bar(self.z, np.ones_like(self.z)*0.25, self.dz * 0.9) for _ in range(nb_actions)]

        if action_set is not None:
            plt.legend(action_set, loc='upper left')

        self.sess = tf.get_default_session()
        self.pi_out = tf.get_default_graph().get_tensor_by_name("distdeepq/q_func/Mul:0")
        self.sigma_out = tf.get_default_graph().get_tensor_by_name("distdeepq/q_func/mul:0")
        self.mu_out = tf.get_default_graph().get_tensor_by_name("distdeepq/q_func/split:2")
        #distdeepq/q_func/sparsemax/TensorArrayStack/TensorArrayGatherV3:0
        #distdeepq/q_func/softmax:0

    def make_pdf(self, obs):
        return self.sess.run(self.pi_out, {"distdeepq/observation:0": obs})[0], \
               self.sess.run(self.sigma_out, {"distdeepq/observation:0": obs})[0], \
               self.sess.run(self.mu_out, {"distdeepq/observation:0": obs})[0]

    def plot_distribution(self, obs):
        pi, sigma, mu= self.make_pdf(obs)

        tfd = tf.contrib.distributions

        gm = tfd.MixtureSameFamily(
            mixture_distribution=pi,
            components_distribution=tfd.Normal(
                loc=mu,
                scale=tf.sqrt(sigma)
            )
        )

        x = np.linspace(-110., -90., int(1e4), dtype=np.float32)
        plt.plot(x, gm.prob(x).eval())

        # for rects, sample in zip(self.bars, pdf_act):
        #     for rect, y in zip(rects, sample):
        #         rect.set_height(y)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

