%% functions
function s = record_voice(time, cli)
y = audiorecorder(8000, 16, 1); % creates the record object.
if cli
   fprintf('\nRecording in 3 ')
   for i = 1:3
      pause(1)
      fprintf('%d ', 3 - i)
   end
end
recordblocking(y, time); % records 4 seconds of speech
s = getaudiodata(y); % extracts the signal as a vector
if cli, fprintf('Got it.\n\n'); end