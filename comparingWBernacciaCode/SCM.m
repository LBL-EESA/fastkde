%This is a version of Bernacchia's code obtained from
%http://abernacchi.user.jacobs-university.de/software.html on 1/7/14 @ 9:32 AM
%PST.  Comments were added by Travis A. O'Brien.

% The method does not work for integer (or uniformly spaced) data!!!
% It may give poor estimates of discontinuous or divergent density distributions
function [xvec denest]=SCM(xdata)

%*****************************
% Set some basic parameters
%*****************************
%The number of x-grid points
nx=1000;
%The number of test t-grid points
nt0=51;  % must be odd in order to contain zero!!!
%The number of full t-grid points
nt=1001; % must be odd in order to contain zero!!!

%Bounds on the proportion of frequency points that should contain useable ECF points
prop1=0.4;
prop2=0.6;
%The average proportion
prop=(prop1+prop2)/2;

%The length of the input dataset
ndata=length(xdata);

%*****************************
% Configure the x/t grids 
%*****************************
%Initially configure the x-grid to span the data with 'nx' evenly spaced points
xmax=max(xdata);
xmin=min(xdata);
xwidth=xmax-xmin;
dx=xwidth/(nx-1);
xvec=xmin:dx:xmax;

%tmax=pi/dx;
%dt=2*pi/xwidth;

%Calculate the corresponding frequency-space grid
tmax=pi/xwidth;
dt=2*tmax/(nt0-1);
tvec=-tmax:dt:tmax;

%**************************************************
% Calculate the ECF on a small subset of
% frequency points to determin the proper size
% of the frequency-space domain.
%**************************************************
%Repeat the ECF calculation until a proper frequency
% grid has been found
ok=0;
while(ok==0)

    %Initialize the ECF estimate
    ch_est=zeros(1,nt0);
    %Calculate the ECF as the test points
    for idata=1:ndata
        ch_est=ch_est+exp(i*tvec*xdata(idata));
    end
    %Normalize the ECF
    ch_est=ch_est./ndata;

    %Find the number of ECF points that are above the filter threshold
    nuse=length(find((abs(ch_est).^2)>(4*(ndata-1)/(ndata^2))));
    %nuse/nt0
    
    %Calculate whether the relative proportion of filter points
    %(the number of useable points over the total number of frequency points)
    %is between the proportions 'prop1' and 'prop2'
    if(nuse/nt0>prop1 && nuse/nt0<prop2)
        %If it is, set the OK flag, so that we exit this loop
        ok=1;
    else
        %If it isn't, expand the frequency domain by the average of prop1/prop2
        %(i.e. double the domain size)
        tmax=tmax*(nuse/nt0)/prop;
        %Recalculate the frequency grid before we repeat
        dt=2*tmax/(nt0-1);
        tvec=-tmax:dt:tmax;
    end
    
end

%*****************************
% Estimate the fourier-space
% optimal density
%*****************************
%Calculate the full frequency grid
dt=2*tmax/(nt-1);
tvec=-tmax:dt:tmax;

%Calculate the ECF on the full grid
ch_est=zeros(1,nt);
for idata=1:ndata
    ch_est=ch_est+exp(i*tvec*xdata(idata));
end
ch_est=ch_est./ndata;

%Find all indices that are above the filter threshold
indpos=find((abs(ch_est).^2)>(4*(ndata-1)/(ndata^2)));

dum=(abs(ch_est).^2).*(ndata^2)/2;
ftsq=1./(dum-(ndata-1)-sqrt(dum.*(dum-2*(ndata-1))));
%ftsq=ndata^2/(ndata-1)^2*(abs(ch_est).^2/2-(ndata-1)/ndata^2+sqrt(abs(ch_est).^2/2.*(abs(ch_est).^2/2-2*(ndata-1)/ndata^2)));
ch_inf=ndata.*ch_est(indpos)./(ndata-1+1./ftsq(indpos));

tvec2=tvec(indpos);

xwidth2=2*pi/dt;
if xwidth2<xwidth
    xwidth=xwidth2;
    dx=xwidth/(nx-1);
    xvec=(mean(xdata)-xwidth/2):dx:(mean(xdata)+xwidth/2);  % to be improved
end

denest=real((ch_inf*exp(-i*tvec2'*xvec))*dt/(2*pi));


