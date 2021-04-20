import numpy as np
import sys

from inputs import InputData
sys.path.append(InputData.utils_path)
from data_generator import DataGenerator

import prode as pro

@pro.utils.timing
def main():
    dataGen = DataGenerator(InputData)
    
    data_train, data_valid = dataGen.training_data()
    
    if InputData.train_cases > 0:
        data_test = dataGen.testing_data()

    data = pro.data.OpDataSet(
        x_train=[data_train[1], data_train[2]],
        y_train=data_train[3],
        x_valid=[data_train[1], data_train[2]],
        y_valid=data_valid[3])

            # T_tr, u_tr, t_tr, y_tr = get_data(idx_train)
            # T_te, u_te, t_te, y_te = get_data(idx_valid)
            # plt.semilogx(t_i[idx_min:idx_max,:], y_i[idx_min:idx_max,:])
            # marker_style = dict(marker='o', lw=0.5, linestyle='none', fillstyle='none', markersize=5)
            # plt.plot(t_tr, y_tr, c='k', **marker_style)
            # plt.plot(t_te, y_te, c='r', **marker_style)
            # plt.show()


    net = pro.net.FuncOpNN(
        InputData.u_dim,
        InputData.y_dim,
        InputData.G_dim,
        InputData.layer_size_branch,
        InputData.layer_size_trunk,
        InputData.activation,
        InputData.kernel_initializer,
        regularization=InputData.regularization,
        dropout_rate=InputData.dropout_rate,
        batch_norm=InputData.batch_norm)

#     def ode_system(x, y):
#         """ODE system.
#         dy1/dx = y2
#         dy2/dx = -y1
#         """
#         y1, y2 = y[:, 0:1], y[:, 1:]
#         dy1_x = dde.grad.jacobian(y, x, i=0)
#         dy2_x = dde.grad.jacobian(y, x, i=1)
#         return [dy1_x - y2, dy2_x + y1]

#     def boundary(_, on_initial):
#         return on_initial

#     def func(x):
#         """
#         y1 = sin(x)
#         y2 = cos(x)
#         """
#         return np.hstack((np.sin(x), np.cos(x)))

#     geom = dde.geometry.TimeDomain(0, 10)
#     ic1 = dde.IC(geom, np.sin, boundary, component=0)
#     ic2 = dde.IC(geom, np.cos, boundary, component=1)
#     data = dde.data.PDE(
#         geom, ode_system, [ic1, ic2], 35, 2, solution=func, num_test=100
#     )

#     layer_size = [1] + [50] * 3 + [2]
#     activation = "tanh"
#     initializer = "Glorot uniform"
#     net = dde.maps.FNN(layer_size, activation, initializer)

#     model = dde.Model(data, net)
#     model.compile("adam", lr=0.001, metrics=["l2 relative error"])
#     losshistory, train_state = model.train(epochs=20000)

#     dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
