% This script will call the following local optimization method.
%
% The objective function will write x to x_file, and then make x_done_file to
% let persistent APOSMM know it can read in x_file. It then waits until
% y_done_file exists before reading in y_file and deleting y_done_file.
%
% After the localopt run is finished, the optimum is written to opt_file.
%
% x0, and the 5 filenames involved must be given when invoking the script.

xopt = fminsearch(@(x)wrapper_obj_fun(x,x_file,y_file,x_done_file,y_done_file),x0)
dlmwrite(opt_file, xopt, 'delimiter', ' ', 'precision', 16)
dlmwrite([opt_file '_flag'], 1)  % This assume xopt is a local min, not just the last point in the run
exit
