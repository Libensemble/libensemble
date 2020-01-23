function y = wrapper_obj_fun(x,x_file,y_file,x_done_file,y_done_file)
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
