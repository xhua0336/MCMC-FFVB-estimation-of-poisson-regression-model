% Implement FFVB with control variate and adaptive learning
clear all
rng(2020) % Fix the random seed
%======================model setting====================================
x_j = rand(100,4) % Generate covariates Xij ~ U(0,1) i = 100, j =4
o = ones(100,1) 
x = [o x_j] %Generate a design matrix i=100, j =5 with first column all one
beta_ini = [1,-0.2, 0.4, 0.1, -0.7].' %
y = poissrnd(exp(x*beta_ini))% a data set of n = 100 observations
mu_hp = 0; sigma2_hp = 100; % prior for beta

%=========================VB setting====================================
d = 10;  % length of variational parameter lambda
S = 2000;  % number of Monte Carlo samples used to estimate the gradient and LB
beta1_adap_weight = 0.9; % adaptive learning weight
beta2_adap_weight = 0.9; % adaptive learning weightmax_iter = 2000;
max_iter = 2000;
patience_max = 10;
t_w = 100; %window

%learning rate:
eps0 = 0.001; 
tau_threshold = max_iter/2;

%===========================Initialization================================
[B,~,stats] = glmfit(x_j,y,'poisson')
Sigma = stats.covb % set Matrix Sigma
lambda = [beta_ini;diag(Sigma)]; % initial lambda
lambda_best = lambda; %to store the optimal lambda

h_lambda = zeros(S,1); % function h_lambda

grad_log_q_lambda = zeros(S,d);
grad_log_q_times_h = zeros(S,d);

mu_mu = lambda(1:5); sigma2_mu = lambda(6:10);
%% Estimate the gradient of LB
parfor s = 1:S    
        % generate beta_s
        beta = mvnrnd(mu_mu,sigma2_mu.*eye(5)).'
        
        grad_log_q_lambda(s,:) = [(beta - mu_mu)./(sigma2_mu.^2); -1*(ones(5,1)./sigma2_mu) + (beta - mu_mu).^2./(sigma2_mu.^3)]
        h_lambda(s) = (-5/2)*log(2*pi) -log(10)-(beta.'*beta/200) + sum(y.*(x*beta) - exp(x*beta) - log(factorial(y))) - (-5/2*log(2*pi) - sum(log(sigma2_mu.^2) - sum(ones(5,1)./(2*sigma2_mu.^2).*(beta - mu_mu).^2)))    
        grad_log_q_times_h(s,:) = grad_log_q_lambda(s,:)*h_lambda(s);
end

grad_LB = mean(grad_log_q_times_h)';

%-------Initilize learning rate -----------------
g_adaptive = grad_LB ; v_adaptive = g_adaptive.^2
g_bar_adaptive = g_adaptive ; v_bar_adaptive = v_adaptive

%-----------calclulate the control variate c------
    cv = zeros(1,d); 
for i = 1:d
    aa = cov(grad_log_q_times_h(:,i),grad_log_q_lambda(:,i));
    cv(i) = aa(1,2)/aa(2,2);
end

iter = 1;
stop = false;
LB = 0; LB_bar = 0; patience = 0;
%% ==========================start iteration================================
while ~stop        
    mu_mu = lambda(1:5); sigma2_mu = lambda(6:10);
    h_lambda = zeros(S,1); % function h_lambda
    grad_log_q_lambda = zeros(S,d);
    grad_log_q_times_h = zeros(S,d); % (gradient of log_q) x h_lambda(.) 
    grad_log_q_times_h_cv = zeros(S,d); % control variate version: (gradient of log_q) x (h_lambda(.)-c) 
    
    parfor s = 1:S    
        % generate beta_s
        beta = mvnrnd(mu_mu,sigma2_mu.*eye(5)).'
      
        grad_log_q_lambda(s,:) = [(beta - mu_mu)./(sigma2_mu.^2); -1*(ones(5,1)./sigma2_mu) + (beta - mu_mu).^2./(sigma2_mu.^3)]
        h_lambda(s) = (-5/2)*log(2*pi) -log(10)-(beta.'*beta/200) + sum(y.*(x*beta) - exp(x*beta) - log(factorial(y))) - (-5/2*log(2*pi) - sum(log(sigma2_mu.^2) - sum(ones(5,1)./(2*sigma2_mu.^2).*(beta - mu_mu).^2)))   
        grad_log_q_times_h(s,:) = grad_log_q_lambda(s,:)*h_lambda(s);
        grad_log_q_times_h_cv(s,:) = grad_log_q_lambda(s,:).*(h_lambda(s)-cv);
    end
    
    %calclulate the control variate c
    cv = zeros(1,d); 
    for i = 1:d
        aa = cov(grad_log_q_times_h(:,i),grad_log_q_lambda(:,i));
        cv(i) = aa(1,2)/aa(2,2);
    end
    
    grad_LB = mean(grad_log_q_times_h_cv)';
    
    g_adaptive = grad_LB; v_adaptive = g_adaptive.^2; 
    g_bar_adaptive = beta1_adap_weight*g_bar_adaptive+(1-beta1_adap_weight)*g_adaptive;
    v_bar_adaptive = beta2_adap_weight*v_bar_adaptive+(1-beta2_adap_weight)*v_adaptive;

    %compute stepsize = min(eps0,eps0*tau_threshold/iter) and update
    if iter>=tau_threshold
        stepsize = eps0*tau_threshold/iter;
    else
        stepsize = eps0;
    end
    
    lambda = lambda+stepsize*g_bar_adaptive./sqrt(v_bar_adaptive);
        
    %Compute the lower bound estimate
    LB(iter) = mean(h_lambda);
    
    %If t≥tW : compute the moving averaged lower bound
    if iter>=t_w
        LB_bar(iter-t_w+1) = mean(LB(iter-t_w+1:iter));
        LB_bar(iter-t_w+1)
    end
       
    if (iter>t_w)
        if (LB_bar(iter-t_w+1)>=max(LB_bar))
            lambda_best = lambda;
            patience = 0;
        else
            patience = patience+1;
        end
    end
    
    if (patience>patience_max)||(iter>max_iter) stop = true; 
    end 
        
    iter = iter+1;
 
end
lambda = lambda_best;
mu_mu = lambda(1:5); sigma2_mu = lambda(6:10);
sigma5 = diag(sigma2_mu)
%% 
%plot the lower bound
plot(LB_bar)
title('Lower Bound') 

%estimate beta
beta_posterior = zeros(S,5)
for s = 1:S    
   beta_posterior(s,:) = mvnrnd(mu_mu, sigma5)
end

beta_posterior_mean = mean(beta_posterior,1)
a = ones(S,5)
beta_posterior_variance = mean((beta_posterior - beta_posterior_mean(ones(1,S),:)).^2, 1)%posterior variance for beta

%calculate predictive mean
x_new = [1.8339,-2.2588,0.8622,0.3188] %Given new observation
predictive_mean = exp(beta_posterior_mean(1) + x_new*beta_posterior_mean(2:5).')%calculate predictive_mean 