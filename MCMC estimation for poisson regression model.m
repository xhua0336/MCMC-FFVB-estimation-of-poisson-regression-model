rng(2020) % Fix the random seed
x_j = rand(100,4) % Generate covariates Xij ~ U(0,1) i = 100, j =4
o = ones(100,1) 
x = [o x_j] %Generate a design matrix i=100, j =5 with first column all one
beta = [1,-0.2, 0.4, 0.1, -0.7].' %
y = poissrnd(exp(x*beta))% a data set of n = 100 observations

[B,~,stats] = glmfit(x_j,y,'poisson')
Sigma = stats.covb % set Matrix Sigma
k = @(beta) exp(-(beta.'*beta/200)+ sum(y.*(x*beta) - exp(x*beta))) % function to compute the kernel k(beta)
N_iter = 10000; % number of interations 
N_burnin = 2000; % number of burnins 
N = N_iter+N_burnin; % total number of MCMC iterations 

markov_chain = zeros(N,5)
beta_initial = [1,-0.2, 0.4, 0.1, -0.7] % starting value
markov_chain(1,:) = beta_initial
n = 1
while n < N
    epsilon = mvnrnd(zeros(5,1),Sigma)
    proposal = markov_chain(n,:)+epsilon
    alpha = min(k(proposal.')/k(markov_chain(n,:).'),1)
    u = rand
    if u <alpha
        markov_chain(n+1, :) = proposal
    else
        markov_chain(n+1,:) = markov_chain(n,:)
    end
    n = n+1
end
%plot the beta 
subplot(2,3,1)
plot(markov_chain(:,1))
title('beta_0')  

subplot(2,3,2)
plot(markov_chain(:,2))
title('beta_1')

subplot(2,3,3)
plot(markov_chain(:,3))
title('beta_2')

subplot(2,3,4)
plot(markov_chain(:,4))
title('beta_3')

subplot(2,3,5)
plot(markov_chain(:,5))
title('beta_4')

a = ones(N - N_burnin,1)
%posterior mean and variance estimate
pme_beta0 = mean(markov_chain(N_burnin+1:N,1))% posterior mean for beta0
pve_beta0 = mean((markov_chain(N_burnin+1:N,1) - a*pme_beta0).^2)%posterior variance for beta0

pme_beta1 = mean(markov_chain(N_burnin+1:N,2))% posterior mean for beta1
pve_beta1 = mean((markov_chain(N_burnin+1:N,2) - a*pme_beta1).^2)%posterior variance for beta1

pme_beta2 = mean(markov_chain(N_burnin+1:N,3))% posterior mean for beta2
pve_beta2 = mean((markov_chain(N_burnin+1:N,3) - a*pme_beta2).^2)%posterior variance for beta2

pme_beta3 = mean(markov_chain(N_burnin+1:N,4))% posterior mean for beta3
pve_beta3 = mean((markov_chain(N_burnin+1:N,4) - a*pme_beta3).^2)%posterior variance for beta3

pme_beta4 = mean(markov_chain(N_burnin+1:N,5))% posterior mean for beta4
pve_beta4 = mean((markov_chain(N_burnin+1:N,5) - a*pme_beta4).^2)%posterior variance for beta4

x_new = [1.8339,-2.2588,0.8622,0.3188] %Given new observation
posterior_mean_estimate = [pme_beta0,pme_beta1,pme_beta2,pme_beta3,pme_beta4]
predictive_mean = exp(posterior_mean_estimate(1) + x_new*posterior_mean_estimate(2:5).')%calculate predictive_mean 

