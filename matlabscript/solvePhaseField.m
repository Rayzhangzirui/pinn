function phi = solvePhaseField(Pwm,Pgm,Pcsf,varargin)
% Solve Cahn-Hilliard equation to get diffused domain
    p.numiter = 100; % number of Cahn Hillard steps
    p.epsilon = 3; % width of diffused domain
    p = parseargs(p, varargin{:});
    h = 1; % spacial resolution, mm (caption figure 1)
    


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

    % if Pwm + Pgm > Pcsf or phi_threshold, then it's tissue 
    % take indicator function of tissue region as initial guess of CH equation
    phi_threshold = 0.1; 
    phi = double( Pwm+Pgm> max(phi_threshold,Pcsf)); 

    
    dt = h^4/(8*2*p.epsilon); % dt condition, see Glioma_ComputePFF_CahnHilliard.cpp
    tau = 1e-3; %  see CahnHillardOperator, tau = 0.001 (some small number, needed if psi(t=0) = {0,1})

    % Euler's method in time
    for i = 1:p.numiter
%         fprintf('%g ',i);

        lap_phi = Lap(phi); 
        u = (1/2)*phi.*(1-phi).*(1-2*phi)-p.epsilon^2*lap_phi;
        M = phi.*(1-phi) + tau;
        dxu = M.*Dx(u);
        dyu = M.*Dy(u);
        dzu = M.*Dz(u);
        dtphi = Dx(dxu)+Dy(dyu)+Dz(dzu);
        phi = phi + dt*dtphi;
        phi = threshod(phi);
    end
    

end