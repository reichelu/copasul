function [y bv glob loc syl] = copasul_preproc(f0,glob,loc,syl,config)


%% time values to sample indices to synchronize input
f0(:,1)=round(f0(:,1)*config.fs);
glob(:,1:2) = round(glob(:,1:2)*config.fs);
glob(1,1)=max(1,glob(1,1));

%% if not provided assign the same file idx to all gobal segmetns 
if size(glob,2)<3
    glob = [glob ones(size(glob,1),1)];
end

%% remove fileIdx column from loc if given
if (size(loc,2)>1 && max(mod(loc(:,end),1))==0)
    loc=loc(:,1:end-1);
end

%% if not or only provided temporal zero
% -> use segment midpoint, resp. symmetrical window around
if size(loc,2)==2     % no midpoint -> center
    loc = [loc mean(loc')'];
elseif size(loc,2)==1 % midpoint only -> symmetrical window
    loc = [loc-config.lwl loc loc+config.lwl];
end

%% find errorneous nuclei (=0) and replace them by segment midpoint
zi=find(loc(:,3)==0);
loc(zi,3)=mean(loc(zi,1:2)')';

loc = round(loc*config.fs);
loc(1,1)=max(1,loc(1,1));
syl = round(syl*config.fs);

%% zero padding to ensure start at sample 1 and end at final glob
while f0(1,1)>1
    f0=[f0(1,1)-1 0; f0];
end
while f0(end,1)<glob(end,end)
    f0=[f0; f0(end,1)+1 0];
end


%% deprecated: all in one:
%[y bv nprc] = fu_f0_preproc(f0,config);

%% preproc f0; if file idx in 3rd col, seperate preproc for each file
if size(f0,2)<3
    f0=[f0 ones(size(f0,1),1)];
end
y=[];
bv=[];
nprc={};
j=1;
for i=min(f0(:,3)):max(f0(:,3));
    f0seg = f0(find(f0(:,3)==i),1:2);
    [yseg bvseg nprcseg] = fu_f0_preproc(f0seg,config);
    y=[y;yseg];
    bv=[bv;bvseg];
    nprc{j}=nprcseg;
    j=j+1;
end

%% remove samples (same as f0 indices)
y=y(:,2);

return
