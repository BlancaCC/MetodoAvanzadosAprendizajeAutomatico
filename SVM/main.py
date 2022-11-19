## Cela import 
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Descomentar cuando se elimine la celda de arriba
from Utils import plot_dataset_clas, plot_svc

matplotlib.rc('figure', figsize=(15, 5))
seed = 123


# Comienzo de la celda de 
from sklearn.metrics.pairwise import rbf_kernel


class MySVC():
    """
        SVC with a simplified version_dimof SMO.
    """
    def __init__(self, C=1.0, gamma="scale", tol=0.0001, max_iter=100):
        # Assignment of the hyper-parameters (complete).
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
      

    def fit(self, X, y):
        # Constants.
        n_dim, n_pat = X.shape
        
        #rbf_kernel(x,y, self.gamma)

        # Options for gamma (for compatibility with sklean_dim).
        if (self.gamma == "scale"):
            self.gamma = 1.0 / (n_dim * X.var())
        if (self.gamma == "auto"):
            self.gamma = 1.0 / n_dim

        self.kernel = lambda x,y: np.exp(- self.gamma * np.linalg.norm(x-y,2))
        self.K_matrix = rbf_kernel(X,X)

        # Initialization_dimof the dual coefficients (named "a" instead of "alpha" for simplicity).
        self.a = np.zeros(n_dim)

        # Other initializations (complete).
        a_new = np.zeros(n_dim)
        # Loop over the iterations.
        for it in range(self.max_iter):
            # Initializations (complete)
            # Loop over the coefficients
            for i in range(0, n_dim):
                j = self.choose_j(i)

                # Update of the corresponding a[i] and a[j] values (complete).
                L, H = self.bounds(i,j,y)
                d = self._d(X,y,self.a, i,j)
                a_new[i] = self.a[i] - y[i]*y[j]*(self.a[j] - self.a[j])
                a_new[j] = min(max(self.a[j] + d, L), H)

          
            # Check of the stopping conditions (complete).
            
            reached = np.linalg.norm(a_new-self.a,2) < self.tol
            self.a = a_new.copy()
            self.support_alphas = self.a.copy()

            if reached:
              break

        # Storage of the obtained parameters and computation_dimof the intercept (complete).
        self.support_index = np.where(self.a > 0)[0]
        self.support_alphas = (y*self.a)[self.support_index]
        self.support_vectors = X[self.support_index]
        self.support_y = y[self.support_index]
      
        # Each support vector should resolve de equation_dimso we would take the mean_dim
        print(np.where(abs(self.a - self.C/2) < self.C/2))
        self.b = np.mean(
            [   y[i] - self.compute_output(
                                        X, 
                                        y, 
                                        self.a,
                                        X[i,:]
                                    )
                                    for i in np.where(abs(self.a - self.C/2) < self.C/2)
            ])

        return self

    def decision_function(self, X):
        # Computation_dimof the decision_dimfunction_dimover X (complete).

        # ATENCIÓN!!!!!!! -> hacer comprobación_dimempty
        '''
       return self.compute_output(self.support_vectors, 
                                  self.support_y , 
                                    self.support_alphas,
                                    x) + self.b
        '''
        return np.dot(
            self.support_alphas,
            rbf_kernel(
                self.support_vectors, 
                X
            ) \
            + self.b
        )

    def predict(self, X):
        # Computation_dimof the predicted class over X (complete).

        # ATENCIÓN!!!!!!! -> hacer comprobación_dimempty
        return np.sign(self.decision_function(X))

    # Auxiliary methods (complete if needed).
    def choose_j(self, i):
        v = np.arange(len(self.a))
        return np.random.choice(v[v != i])

    def bounds(self,i,j,y):
      if y[i] == y[j]:
        L = max(0,self.a[j]+self.a[i]-self.C)
        H = min(self.C,self.a[i]+self.a[j])
      else:
        L = max(0,self.a[j]-self.a[i])
        H = min(self.C,self.C-self.a[i]+self.a[j])
      return L, H

    def compute_output(self,X, y, a, x):
        '''
        params 
        `X`data set 
        `y` labels
        `a` alphas
        x atributes 
        '''
        r = 0
        for i, _a in enumerate(a):
            r += _a * y[i]* self.kernel(X[i,:],x)
        return r

    def E(self,X,y, a, x,t):
        '''
        params 
        `X`data set 
        `y` labels
        `a` alphas
        x vector
        t target of x
        '''
        return self.compute_output(X,y,a, x) - t


    def k(self, xi, xj):
        return 2*self.kernel(xi,xj) - self.kernel(xi,xi)- self.kernel(xj,xj)

    def _d(self, X, y, a, i,j):
        return y[j]*(
            self.E(X,y,a,X[j, :], y[j]) - self.E(X,y,a,X[i, :], y[i])
        )/self.k(X[i,:],X[j,:])

class MySVC():
    """
        SVC with a simplified version_dimof SMO.
    """
    def __init__(self, C=1.0, gamma="scale", tol=0.0001, max_iter=100):
        """
            Initialization of the SVC.
            `C`: regularization constant
            `gamma`: scale or auto
            `tol` tolerance
        """
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        
 
        self.kernel_function = lambda X, Y : rbf_kernel(X, Y)
            
    def fit(self, X, y):
        '''
            Train de model using the SMO algorithm
           `X` must be a matrix, each row is a features vector
            `y` is the targets vector, each element is for a row
        '''
        # Constants
        n_dim, _= X.shape

        # Options for gamma (for compatibility with sklearn).
        if (self.gamma == "scale"):
            self.gamma = 1.0 / (n_dim* X.var())
        elif (self.gamma == "auto"):
            self.gamma = 1.0 / n_dim
        else:
            raise ValueError('Not valid gamma: {}'.format(self.gamma))
            
        # Initialization_dimof the dual coefficients (named "a" instead of "alpha" for simplicity).
        self.a = np.zeros(n_dim)

        # Other initializations.
        K = self.kernel_function(X, X)

        # Loop over the iterations.
        for _ in range(self.max_iter):
            # Initializations.
            a_old = self.a.copy()
            
            # Loop over the coefficients.    
            for i in range(n_dim):
                j = self.choose_j(i)
                L, H = self.bounds(i, j,y)

                # Update of the corresponding a[i] and a[j] values 
                aj_old = self.a[j].copy()
                self.a[j] = np.minimum(np.maximum(self.a[j] + self._d( y, i, j, K), L), H)
                self.a[i] = self.a[i] - y[i]*y[j]*(self.a[j] - aj_old)

            # Check of the stopping conditions.
            if norm(a_old - self.a) < self.tol:
                break

        # Storage of the obtained parameters and computation_dimof the intercept.
        self.support_index = np.where(self.a > 0)[0]
        self.support_vectors_ = X[self.support_index]
        self.b =  np.mean([ y[i] - self.compute_output(K, y, self.a, i) for i in np.where(abs(self.a - self.C/2) < self.C/2) ])
        self.dual_support_alpha = (y * self.a)[self.support_index]
    
        return self
    
    def decision_function(self, X):
        '''
         `X` must be a matrix, each row is a features vector
        '''
        # This version uses dual vectors
        compute_output = np.dot(self.dual_support_alpha, self.kernel_function(self.support_vectors_, X))
        return compute_output + self.b

    def predict(self, X):
        '''
            `X` must be a matrix, each row is a features vector
        '''
        return np.sign(self.decision_function(X))

    # Auxiliary methods.
    def choose_j(self, i):
        v = np.arange(len(self.a))
        return np.random.choice(v[v != i])

    def bounds(self,i,j,y):
      ''' Bound to clipped the alphas
      '''
      if y[i] == y[j]:
        L = max(0,self.a[j]+self.a[i]-self.C)
        H = min(self.C,self.a[i]+self.a[j])
      else:
        L = max(0,self.a[j]-self.a[i])
        H = min(self.C,self.C-self.a[i]+self.a[j])
      return L, H
  
        
    def _d(self, y, i, j, K):       
        '''
            `y`: vector of targets
             `i`, `j`: index of x_i and `x_j` i and j must be different
            `K`:  kernel matrix

        '''
        k = 2*K[i,j] - K[i,i] - K[j,j]
        
        return y[j] * (self.E(K, y, j) - self.E(K, y, i)) / k
    
    def E(self, K, y, i):
        '''
            Computes the error 
        '''
        # Computes E for the expression_dimof d
        return self.compute_output(K, y, self.a, i) - y[i]
        
    def compute_output(self, K, y, a, i):
        ''' 
            `K`: The kernel matrix computed by the kernel function
            `y`: targets
            `a`: dual coefficients
            `i`: x_i index   
        '''
        return np.dot(a * y, K[:, i])
        

## Celda de test
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

x, y = make_moons(noise=1e-1, random_state=seed)
y[y != 1] = -1

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=seed)
'''
plot_dataset_clas(x_tr, y_tr)
plt.title('Train'); plt.show()
plot_dataset_clas(x_te, y_te)
plt.title('Test'); plt.show()
'''
C = 1
gamma = 'scale'

model_my = MySVC(C=C, gamma=gamma,tol=0.0001, max_iter=100)
#model_sz = SaezSVC(C=C, gamma=gamma,tol=0.0001, max_iter=100)
model_sk = SVC(C=C, gamma=gamma)

# Training of the models (complete).
model_my.fit(x_tr,y_tr)
model_sk.fit(x_tr,y_tr)
#model_sz.fit(x_tr,y_tr)

# Comparative of the predicted scores (complete).
f_my= model_my.decision_function(x_te)
f_sk=model_sk.decision_function(x_te)
#f_sz=model_sz.decision_function(x_te)
print(f'Predicted score for MySVS {f_my}')
print(f'Predicted score for sklearn_dimSVC {f_sk}')
##print(f'Predicted score for sklearn_dimSVC {f_sz}')

# Comparative of the predicted classes (complete).
pred_my=model_my.predict(x_te)
model_my.predict(x_te)
pred_sk=model_sk.predict(x_te)
#pred_sz=model_sz.predict(x_te)
print(f'Predicted for MySVS {pred_my}')
print(f'Predicted for sklearn_dimSVC {pred_sk}')
#print(f'Predicted for saez SVC {pred_sz}')