function [alpha, beta, gamma, loglik, xiSum] = forwardbackward(prior, a_ij, B)
%% Iniz.
[M, T] = size(B);
scale = ones(1, T);
loglik = 0;

alpha = zeros(M, T);
beta = zeros(M, T);
gamma = zeros(M, T);
xiSum = zeros(M, M);


%% Forwards
t = 1;
alpha(:, 1) = prior .* B(:,t);
[alpha(:, t), scale(t)] = normalise(alpha(:, t));

for t=2:T
   m = a_ij' * alpha(:, t-1);
   alpha(:, t) = m.* B(:, t);
   [alpha(:,t), scale(t)] = normalise(alpha(:,t));
end
if any(scale == 0)
   loglik = -inf;
else
   loglik = sum(log(scale));
end

%% Backwards
beta(:, T) = ones(M, 1);
gamma(:, T) = normalise(alpha(:, T).* beta(:, T));

for t = T - 1:-1:1
   b = beta(:, t + 1).* B(:, t + 1);
   beta(:,t) = a_ij * b;
   beta(:,t) = normalise(beta(:,t));
   gamma(:,t) = normalise(alpha(:,t) .* beta(:,t));
   xiSum = xiSum + normalise((a_ij .* (alpha(:,t) * b')));
end

function [M, z] = normalise(A, dim)
if nargin < 2
  z = sum(A(:));
  s = z + (z == 0);
  M = A / s;
elseif dim == 1 % normalize each column
  z = sum(A);
  s = z + (z==0);
  M = A ./ repmatC(s, size(A,1), 1);
else
  z = sum(A,dim);
  s = z + (z == 0);
  L = size(A, dim);
  d = length(size(A));
  v = ones(d, 1);
  v(dim) = L;
  c = repmat(s, v');
  M = A./c;
end