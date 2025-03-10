import deepxde as dde
from dde_PINN import NN
dde.config.set_default_float("float64")

def get_dde_PINN_data(geom, mu, num_domain, num_boundary, num_test):
    bc = dde.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary, component=0)

    data = dde.data.PDE(geom, NN(mu).pde,
                        bc,
                        num_domain = num_domain,
                        num_boundary = num_boundary,
                        num_test = num_test)
    return data

"""

L = 2.0   # Length of rectangle (R) 
l = 2.0   # Width of rectangle (R)

# We choose l=L to have a square 

# Bounds of x, y

x_lower = -L/2.0
x_upper =  L/2.0
y_lower = -l/2.0
y_upper =  l/2.0
R = 0.25  # Radius of circle (S)

# Center of the circle

Cx = 0.0
Cy = 0.0
outer = dde.geometry.Rectangle(xmin=(x_lower,y_lower), xmax=(x_upper,y_upper))
inter = dde.geometry.Disk([Cx, Cy], R)
geom = outer

def is_on_boundary(x):
    return geom.on_boundary(x)

eps_pinn_train = 1.15
num_domain = 5000
num_boundary = 3500
num_test = 2500
data = get_dde_PINN_data(geom, eps_pinn_train,num_domain, num_boundary, num_test)
xy_train = data.train_x  # training data
boundary_mask = np.array([is_on_boundary(x) for x in xy_train])
interior_mask = ~boundary_mask
xy_resid = xy_train[interior_mask]
BC_xy = xy_train[boundary_mask]
"""