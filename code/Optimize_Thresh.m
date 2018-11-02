function [x,fval,exitflag,output,population,score] = Optimize_Thresh(nvars,lb,ub,intcon,PopulationSize_Data,InitialPopulationMatrix_Data)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = optimoptions('ga');
%% Modify options setting
options = optimoptions(options,'PopulationSize', PopulationSize_Data);
options = optimoptions(options,'InitialPopulationMatrix', InitialPopulationMatrix_Data);
options = optimoptions(options,'Display', 'iter');
[x,fval,exitflag,output,population,score] = ...
ga(@Loss,nvars,[],[],[],[],lb,ub,[],intcon,options);
