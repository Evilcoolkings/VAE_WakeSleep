import numpy as np

Z_mu = np.asarray([[1, 0.2],[0.4, 0.5]])
Z_logvar = np.asarray([[1, 0.2],[0.4, 0.5]])
Z = np.asarray([[1, 0.2],[0.5, 0.5]])
Z_var = np.exp(0.5*np.log(Z_logvar))

a = -0.5*(Z - Z_mu)**2/(Z_var**2) - Z_logvar
print(a)
