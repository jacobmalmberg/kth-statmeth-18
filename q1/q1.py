import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma

def generator(seed, N, M, K, W, alpha_bg, alpha_mw):
    # Data generator.
    # Input: seed: int, N: int, M: int, K: int, W: int, alpha_bg: numpy array with shape(K), alpha_mw: numpy array with shape(K)
    # Output: D: numpy array with shape (N,M), R_truth: numpy array with shape(N), theta_bg: numpy array with shape (K), theta_mw: numpy array with shape (W,K)

    np.random.seed(seed)        # Set the seed

    D = np.zeros((N,M))         # Sequence matrix of size NxM
    R_truth = np.zeros(N)       # Start position of magic word of each sequence

    theta_bg = np.zeros(K)      # Categorical distribution parameter of background distribution
    theta_mw = np.zeros((W,K))  # Categorical distribution parameter of magic word distribution

    # YOUR CODE:
    # Generate D, R_truth, theta_bg, theta_mw. Please use the specified data types and dimensions.

    alphabet = [x for x in range(1, K+1)];

    for i in range(R_truth.shape[0]):
        R_truth[i] = np.random.randint(low = 0, high = M-W+1)

    theta_bg = np.random.dirichlet(alpha_bg) # draw theta from a dirchlet prior
    for i in range(W):
        theta_mw[i] = np.random.dirichlet(alpha_mw)


    for n in range(N):
        magic_word_index = 0
        for m in range(M):
            if m in range(int(R_truth[n]), int(R_truth[n])+W): # draw from magic word disitributions
                #print(m)
                s = np.random.multinomial(1, theta_mw[magic_word_index])
                D[n,m] = alphabet[np.where(s == 1)[0].item()]
                magic_word_index += 1
            else:
                s = np.random.multinomial(1, theta_bg)
                D[n,m] = alphabet[np.where(s == 1)[0].item()]

    return D, R_truth, theta_bg, theta_mw

def margLikelihood_magic(alpha, totalcount, n_counts):
    K = n_counts.shape[0]
    J = n_counts.shape[1]
    marginal_likelihood = np.zeros(J)  # our output

    term1 = math.lgamma(np.sum(alpha)) - math.lgamma(totalcount + np.sum(alpha))
    for j in range(J):
        term2 = 0
        for k in range(K):
            term2 = math.lgamma(n_counts[k,j] + alpha[k]) - math.lgamma(alpha[k]) + term2
        marginal_likelihood[j] = term2 + term1
    return marginal_likelihood

def margLikelihood_bg(alpha, totalcount, n_counts):
    K = n_counts.shape[0]
    marginal_likelihood = 1 # our output


    term1 = math.lgamma(np.sum(alpha)) - math.lgamma(totalcount + np.sum(alpha))
    term2 = 0
    for k in range(K):
        term2 = math.lgamma(n_counts[k] + alpha[k]) - math.lgamma(alpha[k]) + term2
    marginal_likelihood = term2 + term1

    return marginal_likelihood


def full_conditional(marg_like_magic, marg_like_bg):
    term1 = 0
    for j in range(marg_like_magic.shape[0]):
        term1 = marg_like_magic[j] + term1
    product = term1 + marg_like_bg
    return product

def gibbs(D, alpha_bg, alpha_mw, num_iter, W):
    # Gibbs sampler.
    # Input: D: numpy array with shape (N,M),  alpha_bg: numpy array with shape(K), alpha_mw: numpy array with shape(K), num_iter: int
    # Output: R: numpy array with shape(num_iter, N)
    M = D.shape[1]
    K = alpha_bg.shape[0]

    N = D.shape[0]
    R = np.zeros((num_iter, N)) # Store samples for start positions of magic word of each sequence
    # YOUR CODE:
    # Implement gibbs sampler for start positions.

    for i in range(N):
        R[0][i] = np.random.randint(low = 0, high = M-W+1)

    Ntotal = N*W
    Btotal = N*(M-W)

    Ncounts = np.zeros((K, W)) # sum all columns in magic words
    Bcounts = np.zeros(K)

    for iter in range(1, num_iter):        # for every iteration
        R_temp = R[iter-1]
        for n in range(N):              # for each sequence
            full_conditional_vec = np.zeros(M-W+1)
            for pos in range(M-W+1):      # for each possible position for r_n
                Ncounts = np.zeros((K, W)) # sum all columns in magic words
                Bcounts = np.zeros(K)
                R_temp[n] = pos
                #print('R_temp: ', R_temp)
                for sequence in range(N):
                    j = 0
                    for m in range(M):
                        symbol = D[sequence,m]
                        if m in range(int(R_temp[sequence]), int(R_temp[sequence])+W):  # check whether the symbol is part of magic word
                            Ncounts[int(symbol)-1, j] += 1
                            j += 1
                        else:
                            Bcounts[int(symbol)-1] += 1
                #print('Ncounts: ', Ncounts)
                #input()
                marg_like_bg = margLikelihood_bg(alpha_bg, Btotal, Bcounts)
                marg_like_magic = margLikelihood_magic(alpha_mw, Ntotal, Ncounts)
                #print('marg_like_bg: ', marg_like_bg)
                #print('marg_like_magic: ', marg_like_magic)
                full_conditional_vec[pos] = full_conditional(marg_like_magic, marg_like_bg)


            # här ska vi sampla från full_conditional_vec
            p = np.exp(full_conditional_vec - np.max(full_conditional_vec))
            p = p/ np.sum(p)
            R_temp[n] = np.argmax(np.random.multinomial(1, p))

        R[iter] = R_temp
        #if iter > 100:
        #    print('R[iter]: ', R[iter])
        #input()


            #print(full_conditional_vec)
        #    input()

    #print('Ncounts: ', Ncounts)
    #print('Bcounts: ', Bcounts)
    #print('R', R[-1])

    #histogram(R)
    return R

def histogram(R, bins=6):
    bins = [x for x in range(bins)]
    y = R[100:-1,0]
    #p.histogram(y,bins)
    plt.hist(y, bins)
    plt.show()



def plot_convergence(sequence_vec, truth, seed, show=False, chain=False, sequence=False):
    """to plot convergence for one sequence"""
    truth_vec = [truth]*len(sequence_vec) #make it into a long vector for plot
    if chain: ##plot the chain stuff
        plt.figure(sequence) #which figure to use
        plt.plot(sequence_vec, label='start position samples, chain =' +str(int(chain)))
        #plt.plot(truth_vec, label='truth = ' +str(int(truth)) + ", chain = "+str(int(chain)), alpha =0.7)
        plt.legend(loc='upper left')
        #plt.title("truth is " +str(int(truth)) +", seed is " +str(seed))
        if show:
            #num_chains=chain+1 #how many chains were run
            plt.title("Sequence "+str(sequence) +". Convergence plot for "+str(chain) +" chains. Seed = " +str(seed))
            plt.plot(truth_vec, label='truth = ' +str(int(truth)), alpha =0.7)
            plt.legend(loc='upper left')
            #plt.show()
            plt.ylim(-1, 6)
            plt.savefig('sequence_'+str(sequence)+'.png')

    else:
        plt.plot(sequence_vec, label='start position samples')
        plt.plot(truth_vec, label='truth = ' +str(int(truth)), alpha =0.7)
        plt.legend(loc='upper left')
        plt.title("truth is " +str(int(truth)) +", seed is " +str(seed))
        if show:
            plt.show()


def multiple_runs_multiple_sequences(seed, N, M, K, W, num_iter, alpha_bg, alpha_mw):
    """multiple chains, for convergence etc"""
    """all sequences"""

    chains = 3
    D, R_truth, theta_bg, theta_mw = generator(seed, N, M, K, W, alpha_bg, alpha_mw) #yes it should be here
    r_list=list()
    for i in range(chains):
        R = gibbs(D, alpha_bg, alpha_mw, num_iter, W)
        r_list.append(R)

    for k in range(chains):
        for j in range(N):
            if k != (chains-1):
                plot_convergence(r_list[k][:,j], R_truth[j], seed, chain=(k+1), sequence=j)
            else:
                plot_convergence(r_list[k][:,j], R_truth[j], seed, show=True, chain=(k+1), sequence=j) #only show the final plot


def multiple_runs_convergence(seed, N, M, K, W, num_iter, alpha_bg, alpha_mw):
    """multiple chains, for convergence.
    one sequence
    """

    chains = 3
    seq = 1 #which sequence to plot for
    D, R_truth, theta_bg, theta_mw = generator(seed, N, M, K, W, alpha_bg, alpha_mw) #yes it should be here

    for i in range(chains):
        R = gibbs(D, alpha_bg, alpha_mw, num_iter, W)
        print("chain ", i)
        print("all samples for sequence " +str(seq))
        print(R[:,seq])
        print("\n")
        if i != (chains-1):
            plot_convergence(R[:,seq], R_truth[seq], seed, chain=(i+1))
        else:
            plot_convergence(R[:,seq], R_truth[seq], seed, show=True, chain=i) #only show the final plot


def multiple_runs_accuracy(seed, N, M, K, W, num_iter, alpha_bg, alpha_mw):
    """multiple chains, for accuracy.
    """

    chains = 3
    seq = 1 #which sequence to plot for
    D, R_truth, theta_bg, theta_mw = generator(seed, N, M, K, W, alpha_bg, alpha_mw) #yes it should be here
    theLag = 10 # if we want lag, otherwise set to 0

    for i in range(chains):
        print("chain ", i)
        R = gibbs(D, alpha_bg, alpha_mw, num_iter, W)

        measure_accuracy(R, R_truth, seed, lag=theLag)
        print("\n"*5)

def measure_accuracy(R, R_truth, seed, lag=False):
    """to measure accuracy"""
    """using lag if you want"""
    from collections import Counter
    sequences=len(R_truth)

    correct=0
    burnin=50 #discard first 50 samples
    guess_vect=list()

    for i in range(sequences):
        if lag!=False:
            data = Counter(R[50::lag,i])
        else:
            data =Counter(R[50:,i])

        most_common=int(data.most_common(1)[0][0]) #our guess is the one most frequent
        guess_vect.append(most_common)
        if most_common == int(R_truth[i]):
            print("Correct", data.most_common(1))
            correct+=1
        else:
            print("Wrong", data.most_common(1), "truth was", R_truth[i] )


    print("Seed " + str(seed) +". Accuracy was " + str(100*correct/sequences) +"%")
    print("\nStart positions (truth): ")
    print(R_truth)

    print("\nStart positions most guessed: ")
    print(guess_vect)







def main():
    seed = 123 #123 standard

    N = 20
    M = 10
    K = 4
    W = 5
    alpha_bg = np.array([12,7,3,1]) #np.ones(K)
    alpha_mw = np.ones(K) * 0.9
    num_iter = 500
    #multiple_runs(seed, N, M, K, W, num_iter, alpha_bg, alpha_mw)
    #multiple_runs_multiple_sequences(seed, N, M, K, W, num_iter, alpha_bg, alpha_mw)
    multiple_runs_accuracy(seed, N, M, K, W, num_iter, alpha_bg, alpha_mw)


    # print("Parameters: ", seed, N, M, K, W, num_iter)
    # print(alpha_bg)
    # print(alpha_mw)
    #
    # # Generate synthetic data.
    # D, R_truth, theta_bg, theta_mw = generator(seed, N, M, K, W, alpha_bg, alpha_mw)
    # print("\nSequences: ")
    # print(D)
    # print("\nStart positions (truth): ")
    # print(R_truth)
    #
    # # Use D, alpha_bg and alpha_mw to infer the start positions of magic words.
    # R = gibbs(D, alpha_bg, alpha_mw, num_iter, W)
    # print("\nStart positions (sampled): ")
    # print(R[-1,:])

    #measure_accuracy(R, R_truth, seed)

    #print(R[:,1])
    # #plot_convergence(R[:,1], R_truth[1], seed)
    # #plot_convergence(R[:,2], R_truth[2], seed, show=True)
    # multiple_runs(seed, N, M, K, W, num_iter, alpha_bg, alpha_mw)

    #print(R[1,:])

    # YOUR CODE:
    # Analyze the results. Check for the convergence.

if __name__ == '__main__':
    main()
