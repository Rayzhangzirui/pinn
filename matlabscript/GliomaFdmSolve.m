function [phi,uall,tall,u] = GliomaFdmSolve(Pwm, Pgm, Pcsf, Dw, rho, tfinal, ix, varargin)
    
    p.u0 = @(r) 0.1*exp(-0.1*r.^2);
    p.xdim = 2;
    p = parseargs(p, varargin{:});
    


    h = 1; % spacial resolution, mm (caption figure 1)
    epsilon = 3; % width of diffused domain
    Dg = Dw/10;
    D = Pwm*Dw + Pgm*Dg; % diffusion coefficients
    sz = [1 1 1];
    sz(1:numel(size(Pwm))) = size(Pwm); % even when Pwm is 2d, 3rd dim is 1

    % integer grid, horizontal is x, vertical y, only used to get u0
    [gx,gy,gz] = ndgrid(1:sz(1),1:sz(2),1:sz(3)); 


    %% finite difference operators
    flap = cat(3,[0 0 0;0 1 0; 0 0 0],[0 1 0;1 -6 1; 0 1 0],[0 0 0;0 1 0; 0 0 0])/h^2;
    fdx = [-1 0 1]/(2*h);
    fdy = fdx';
    fdz = reshape(fdx,1,1,3);
    operator = @(x,f) imfilter(x,f,'circular','same'); % periodic bc, so that in 2d flap is actually [1,-4, 1]

    Dx = @(x) operator(x,fdx);
    Dy = @(x) operator(x,fdy);
    Dz = @(x) operator(x,fdz);
    Lap = @(x) operator(x,flap);

    threshod = @(x) min(max(x,0),1); % threshold in [0,1]
    %% Solve Cahn-Hilliard equation to get diffused domain

    % if Pwm + Pgm > Pcsf or phi_threshold, then it's tissue 
    % take indicator function of tissue region as initial guess of CH equation
    phi_threshold = 0.1; 
    phi = double( Pwm+Pgm> max(phi_threshold,Pcsf)); 

    numiter = 100; % number of Cahn Hillard steps
    dt = h^4/(8*2*epsilon); % dt condition, see Glioma_ComputePFF_CahnHilliard.cpp
    tau = 1e-3; %  see CahnHillardOperator, tau = 0.001 (some small number, needed if psi(t=0) = {0,1})

    % Euler's method in time
    for i = 1:numiter
        fprintf('%g\n',i)

        lap_phi = Lap(phi); 
        u = (1/2)*phi.*(1-phi).*(1-2*phi)-epsilon^2*lap_phi;
        M = phi.*(1-phi) + tau;
        dxu = M.*Dx(u);
        dyu = M.*Dy(u);
        dzu = M.*Dz(u);
        dtphi = Dx(dxu)+Dy(dyu)+Dz(dzu);
        phi = phi + dt*dtphi;
        phi = threshod(phi);
    end
    
    
   %% Solve reaction diffusion 
    r = sqrt(h.^2*((gx-ix(1)).^2+(gy-ix(2)).^2+(gz-ix(3)).^2)); % distance
    u0 = p.u0(r);
%     u0 = exp(-0.01*r2.^2);

    dt0 = 0.99*h^2/(6*max(D(:))); % CFL condition, timestep
    % Euler's method in time
    u = u0;
    t = 0;
    tall = [t];
    uall = {u0};

    DxDphi = Dx(D.*phi);
    DyDphi = Dy(D.*phi);
    DzDphi = Dz(D.*phi);

    while t<tfinal

        fprintf('%g\n',t)
        if t+dt0<tfinal
            dt = dt0;
        else
            dt = tfinal-t;
        end

        dudt = DxDphi.*Dx(u) + DyDphi.*Dy(u) + DzDphi.*Dz(u) + D.*phi.* Lap(u) + rho*phi.*u.*(1-u);
        
        u = u + dt*dudt;
        u = threshod(u);
        t = t + dt;

        tall(end+1) = t;
        uall{end+1} = u;
    end
    
    uend = u;
    uall = cat(p.xdim+1,uall{:});
end
