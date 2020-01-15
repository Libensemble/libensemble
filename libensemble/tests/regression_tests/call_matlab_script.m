% This script will call the following local optimization method.
%
% The objective function will write x to x_file, and then make x_done_file to
% let persistent APOSMM know it can read in x_file. It then waits until
% y_done_file exists before reading in y_file and deleting y_done_file.
%
% After the localopt run is finished, the optimum is written to opt_file.
%
% x0, and the 5 filenames involved must be given when invoking the script.

xopt = fminsearch(@(x)fun(x,x_file,y_file,x_done_file,y_done_file),x0)
dlmwrite(opt_file, xopt, 'delimiter', ' ', 'precision', 16)
exit

function y = fun(x,x_file,y_file,x_done_file,y_done_file)
    % This is the objective function used to communicate x-values and receive
    % y-values.

    % Write x to x_file and then make x_done_file
    dlmwrite(x_file, x, 'delimiter', ' ', 'precision', 16)
    dlmwrite(x_done_file, 1)

    % Wait until y_done_file appears
    while exist(y_done_file,'file')==0
        pause(0.01)
    end

    % Read in y and delete y_done_file
    y = dlmread(y_file)
    delete(y_done_file)
end
