function B = obsProbs(x, mu, sigma)
[N, M] = size(mu);
[~, T] = size(x);

B = zeros(M, T);
for j=1:M
   denom = (2*pi)^(N/2)*sqrt(prod(sigma(:, j))); 
   D = sqdist(x, mu(:, j), inv(diag(sigma(:, j))))';
   B(j, :, :) = exp(-0.5*D)/denom;
end
