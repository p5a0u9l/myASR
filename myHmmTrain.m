function [prior, a_ij, mu, sigma] = myHmmTrain(data, initialStateDist, a_ij, mu, sigma)

%% iniz
max_iter = 200;
thresh = 5e-4;
nTrain = length(data);
N = size(data{1}, 1);
M = length(initialStateDist);

expNumTrans = zeros(M, M);
expNumVisits = zeros(M, 1);

gammaSum = zeros(M, 1);
m = zeros(N, M);
op = zeros(N, N, M);
logLikSum = 0;
prevLogLik = -inf;
converged = 0;
iterCount = 1;
prior = initialStateDist;

%% Convergence loop
while ~converged
   %% E step
   logLikSum = 0;
   for iTrain=1:nTrain
      x = data{iTrain};
      T = size(x, 2);
      B = obsProbs(x, mu, sigma);
      [~, ~, gamma,  curLogLik, xiSum] = forwardbackward(prior, a_ij, B);
      logLikSum = logLikSum +  curLogLik;
      
      expNumTrans = expNumTrans + xiSum; % sum(xi,3);
      expNumVisits = expNumVisits + gamma(:,1);
      gammaSum = gammaSum + sum(gamma, 2);
      
      for i = 1:M
         w = gamma(i, :); % w(t) = w(i,k,t,l)
         wobs = x .* repmat(w, [N 1]); % wobs(:,t) = w(t) * obs(:,t)
         m(:, i) = m(:, i) + sum(wobs, 2); % m(:) = sum_t w(t) obs(:,t)
         op(:, :, i) = op(:, :, i) + wobs * x'; % op(:,:) = sum_t w(t) * obs(:,t) * obs(:,t)'
      end
   end
   
   %% M step
   prior = normalise(expNumVisits);
   a_ij = mk_stochastic(expNumTrans);
   [mu, sigma] = updateGaussian(gammaSum, m, op);
      
   %% Check for convergence
   d_loglik = abs(logLikSum - prevLogLik);
   avg_loglik = (abs(logLikSum) + abs(prevLogLik))/2;
   if (d_loglik / avg_loglik) < thresh, converged = 1; end

   %% Reloop
   prevLogLik = logLikSum;
   if (iterCount == max_iter), dbstop in hmm_em at 62; end
   iterCount =  iterCount + 1;
   if mod(iterCount, 10) == 0
      fprintf('\t\titeration %d, loglik = %f, threshCheck %f \n', iterCount, logLikSum, d_loglik / avg_loglik);
   end
end

function [mu, sigma] = updateGaussian(gammaSum, m, op)
[Y, M] = size(m);

for i = 1:M
   mu(:, i) = m(:, i) / gammaSum(i);
end
for i = 1:M
   SS = op(:,:,i)/gammaSum(i)  - mu(:,i)*mu(:,i)';
   sigma(:, i) = diag(SS);
end
