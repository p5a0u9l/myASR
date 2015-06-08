function filt = filterList(list, filter) % because matlab doesn't do list comprehensions
%% FILTERLIST
i = 0;
for n = 1:length(list)
   if regexp(list{n}, filter)
      i = i + 1;
      filt{i} = list{n};
   end
end
