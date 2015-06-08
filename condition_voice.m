function y = condition_voice(x, fs)
n = 40;
initBuffer = floor(0.1*fs); % ms
tailBuffer = ceil(0.2*fs); % ms
headroom = 0.1; % percent

% normalize
x = x/((1 - headroom)*max(abs(x)));

sigma = var(x);
xMag = abs(x);

% est. start/stop of speech, add pre and tail buffers
s0 = find(xMag > n*sigma, 1) - initBuffer;
sF = find(xMag > n*sigma, 1, 'last') + tailBuffer;

if s0 < 0, s0 = 1; end

y = x(s0:sF); % return trimmed, normalized speech