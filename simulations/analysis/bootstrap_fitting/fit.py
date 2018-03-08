import numpy as np

raw_data = np.loadtxt("gammaMma.dat")

sample_length = 326
num_samples = len(raw_data)/sample_length


data = raw_data.reshape(num_samples, sample_length).T

K = 2000


bootstrap_sample_means = np.zeros(shape=(K, data.shape[1]))

#sample_length = 5

for k in range(K):

    idx = tuple(np.random.choice(np.arange(sample_length), size=sample_length, replace=True))

    #now pick the same idx rows for all columns in data; again one step
    bootstrap_sample = data[idx,:]

    #and get an entry for the collection of bootstrap means
    bootstrap_sample_means[k,:] = np.mean(bootstrap_sample, axis=0)



bootstrap_means_mean = np.zeros_like(bootstrap_sample_means)
bootstrap_means_mean[tuple(np.arange(K)),:] = np.mean(bootstrap_sample_means, axis=0)


###
##### calculate matrices needed for fitting

diff = bootstrap_sample_means - bootstrap_means_mean

#calculate (frozen) covariance matrix
covMat = np.zeros(shape=(num_samples, num_samples))
for i in range(num_samples):
    for j in range(num_samples):
        covMat[i,j] = np.mean( diff[:,i] * diff[:,j] )

#calculate inverse covariance matrix
Cinv = np.linalg.inv(covMat)


#define matrix Z

#        1 L[0]
#        1 L[1]
#Zmat =  1 L[2]
#        . .
#        . .
#        . .

L = np.arange(num_samples)
# actually: L[i] = i-th (log) lattice length

Zmat = np.zeros(shape=(num_samples, 2))
for i in range(num_samples):
    Zmat[tuple(np.arange(2)),:] = np.array([1, L[i]])


#further matrices used for fitting below
tMat1 = np.matmul(np.matmul(Zmat.T, Cinv), Zmat)
tMat2 = np.linalg.inv(tMat1)

TMatrix = np.matmul(np.matmul(tMat2, Zmat.T), Cinv)

#####
###



###
##### do bootstrap fits

#container for bootstrap fit parameters
Tbeta = np.zeros(shape=(K, 2))

bootstrap_sample_means = np.zeros(shape=(K, data.shape[1]))

for b in range(K):

    idx = tuple(np.random.choice(np.arange(sample_length), size=sample_length, replace=True))

    #now pick the same idx rows for all columns in data; again one step
    bootstrap_sample = data[idx,:]

    #and get an entry for the collection of bootstrap means
    bootstrap_sample_means[b,:] = np.mean(bootstrap_sample, axis=0)

for b in range(K):

    #calculate estimator Tbeta for coefficients beta
    Tbeta[b,:] = np.matmul(TMatrix, bootstrap_sample_means[b,:])


#####
###


###
##### averaging of fit parameters and plotting

# y = beta1 + x*beta2

beta1 = np.mean(Tbeta[:,0])
deltaBeta1 = np.std(Tbeta[:,0])

beta2 = np.mean(Tbeta[:,1])
deltaBeta2 = np.std(Tbeta[:,1])

print(str(beta1) + " +- " + str(deltaBeta1))
print(str(beta2) + " +- " + str(deltaBeta2))
