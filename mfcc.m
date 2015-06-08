%% MFCC
function [c, dc] = mfcc(s, C, B, M)
% Inputs
% s - the speech signal
% C - the # of cepstral coefficients
% B - the # of mel-scale filter banks
% M - the order of MFCC derivative deltas

debug = 0;
%% Setup and Iniz.

% function handles
hz2mel = @(hz)(1125*log(1 + hz/700));
mel2hz = @(mel)(700*exp(mel/1125) - 700);
lifter = @(N, L)(1 + 0.5*L*sin(pi*(0:N-1)/L));

% Parameters
nTotal = length(s); % # samples in speech signal
tFrame = 25; % time per frame (ms)
tShift = 10; % time per shift
alpha = 0.97; % preemphasis coefficient
L = 22; % cepstral sine lifter parameter
fs = 8e3;  % signal sample rate
fL = 300; % lower Mel frequency limit
fH = 3700;                  % upper Mel frequency limit
N = round(fs*tFrame/1e3);   % # sample in one frame
nShift = round(fs*tShift/1e3);  % # sample in a shift
T = round(1 + (nTotal - N)/nShift); % # frames of speech
K = 2^(nextpow2(N)+1);  % # FFT bins

% precompute DCT matrix
b = 1:B;
for n = 0:C - 1
   DCT(n+1, :) = sqrt(2/B)*cos(pi*n*(b - 0.5)/B);
end

% preemphasis
s = filter([1 -alpha], 1, s);

% cepstral lifter 
lift = lifter(T, L).';

% generate frames
s = [s; zeros(N + (T - 1)*nShift, 1)];
x = zeros(N, T);
idx = zeros(N, T);
for t = 1:T
   idx(:, t) = (1:N) + (t - 1)*nShift;
   x(:, t) = s(idx(:, t)).*hamming(N);
end

% Mel scale filter bank
f = linspace(0, fs/2, K/2 + 1); % 
mel = hz2mel(f);
fc = mel2hz(hz2mel(fL) + (0:B + 1) * ((hz2mel(fH)-hz2mel(fL))/(B + 1)));
ms_fb = melScaleFilterBank(f, fc);  % (nBank) x (L/2 + 1)

%% Processing
% |fft|^2
x = fft(x, K);
x = x(1:K/2+1, :); % keep 0 to fs/2
xMag = abs(x); % (L/2 + 1) x (nFrame)

% log energy of filter outputs
xLog = log(ms_fb*xMag); % (nBank) x (nFrame)

% cosine transform
xDct = DCT*xLog; %   (nBank) x (nFrame)
    
% lifter
c = (diag(lift)*(xDct.')).';

% Deltas
n = (1:M)';
dc = zeros(C, T - 2*M); % drop 1st M and last M frames to accomodate difference
for ic = 1:C
   for t = (1 + M):(T - M)
      dc(ic, t - M) = ((c(ic, t+n) - c(ic, t - n))*n)/(2*sum(n.^2));
   end
end

c = c(:, (1 + M):(T - M)); 

%% Plots
if debug; figure(1); plot(ms_fb.'); title('Triangle Filter Bank'); end; 
if debug; figure(2); imagesc(10*log10(xMag)); title('$|FFT|^2$ [dB]'); end; 
if debug; figure(3), imagesc(xLog.'); xlabel('Frame'); title('Mel-Filtered Log Energy'); end
% if debug; figure(3), plot(x, '.-'); if debug;title(sprintf('Cosine Transform, Frame %d', iFr)); end
if debug; figure(4), imagesc(c.'); xlabel('Frame'); colormap hot; title(sprintf('MFCCs')); colorbar; end
if debug; distfig('Rows', 2, 'Cols', 2); pause(0.1); end

%% Functions
function H = melScaleFilterBank(f, fc)
% implements 6.150 of Huang et al text
N = length(f);
M = length(fc) - 2;
H = zeros(M, N);
for m = 1:M
   k = and(f >= fc(m), f <= fc(m + 1));
   H(m, k) = (f(k) - fc(m))/(fc(m + 1) - fc(m));
   k = f >= fc(m + 1) & f <= fc(m + 2);
   H(m, k) = (fc(m + 2)-f(k))/(fc(m + 2)-fc(m + 1));
end