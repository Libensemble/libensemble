for n = 2:7
    mins = 10;
    nonfeas = [];
    feas = [];
    for i = 1:4000;
        str = ['lmin_' int2str(n) '_' sprintf('%04d',i) '_' int2str(mins)];
        A = load(str);
        if any(any(A(:,1:end-1) < 0)) || any(any(A(:,1:end-1) > 1))
            nonfeas = [nonfeas; i];
%             delete(str);
        else
            feas = [feas; i];
        end
    end
    dlmwrite(['nonfeas_' int2str(n) '_' int2str(mins)],nonfeas)
    dlmwrite(['feas_' int2str(n) '_' int2str(mins)],feas)
end
