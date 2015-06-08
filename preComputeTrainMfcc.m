function preComputeTrainMfcc(list)
%% PRECOMPUTEALLMFCC
global baseline obsMat
list = filterList(list, 'train_.*_.*');

%% Iterate over training list and generate features
for i = 1:length(list)
   wavName = list{i};
   fprintf('preComputeAllMfcc %d of %d\n', i, length(list));
   if baseline
      obsMat.(wavName) = altMfcc(wavName);
   else
      obsMat.(wavName) = myMfcc(wavName);
   end
end

function y = altMfcc(wavName)
%% COMPUTEMFCC
global wavMat debug
% load waveform
C = 12;  % MFCC order
% B = 26;  % mel-scale filter bank order
% M = 2;   % MFCC Delta order

fs = 8e3;
if debug; fprintf('Begin feature extraction for %s...\n', wavName);  end
s = wavMat.(wavName);
nTotal = length(s); % # samples in speech signal
tFrame = 25; % time per frame (ms)
tShift = 10; % time per shift
fL = 300; % lower Mel frequency limit
fH = 3700;                  % upper Mel frequency limit
N = round(fs*tFrame/1e3);   % # sample in one frame
nShift = round(fs*tShift/1e3);  % # sample in a shift
T = round(1 + (nTotal - N)/nShift); % # frames of speech

% Compute cepstral coefficients and delta coefficients
[y, ~] = melcepst(s, fs, '0d', C, 26, N, nShift, fL/fs, fH/fs);

function y = myMfcc(wavName)
%% COMPUTEMFCC
global wavMat plots debug
% load waveform
C = 13;  % MFCC order
B = 26;  % mel-scale filter bank order
M = 2;   % MFCC Delta order
fs = 8e3;
if debug; fprintf('Begin feature extraction for %s...\n', wavName);  end
s = wavMat.(wavName);
% Compute cepstral coefficients and delta coefficients
[CC, DC] = mfcc(s, C, B, M);
y = [CC; DC];

% Plots
titleStr = regexprep(wavName, '_', ' ');
if plots(1); figure(1); plot((1:length(s))/fs, s);   xlabel('time [s]');  title(['Waveform: ', titleStr]); snapnow; end
if plots(2); figure(2); imagesc((y)); colorbar; colormap hot; caxis([-100, 100]); xlabel('Frame'); ylabel('Feature'); title(['MFCCs and Deltas: ', titleStr]); snapnow; end
