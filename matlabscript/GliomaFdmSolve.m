function fdmsol = GliomaFdmSolve(atlas, rho, tfinal, ix, varargin)
    
%     p.u0 = @(r) 0.1*exp(-0.1*r.^2);
    p.u0 = @(x,y,z) 0.1*exp(-0.1*((x-ix(1)).^2+(y-ix(2)).^2+(z-ix(3)).^2));
    p.xdim = 2;
    p.factor = 10;
    p = parseargs(p, varargin{:});
    
    h = 1; % spacial resolution, mm (caption figure 1)
    
    % integer grid, nd grid
    [gx,gy,gz,D,phi,P] = deal(atlas.gx, atlas.gy, atlas.gz, atlas.df, atlas.phi,atlas.P); 

    sz = [1 1 1];
    sz(1:numel(size(D))) = size(D); 


    %% finite difference operators
    flap = cat(3,[0 0 0;0 1 0; 0 0 0],[0 1 0;1 -6 1; 0 1 0],[0 0 0;0 1 0; 0 0 0])/h^2;
    fdx = [-1 0 1]'/(2*h);
    fdy = fdx';
    fdz = reshape(fdx,1,1,3);
    operator = @(x,f) imfilter(x,f,'circular','same'); % periodic bc, so that in 2d flap is actually [1,-4, 1]

    Dx = @(x) operator(x,fdx);
    Dy = @(x) operator(x,fdy);
    Dz = @(x) operator(x,fdz);
    Lap = @(x) operator(x,flap);

    threshod = @(x) min(max(x,0),1); % threshold in [0,1]
   
    
   %% Solve reaction diffusion 
%     r = sqrt(h.^2*((gx-ix(1)).^2+(gy-ix(2)).^2+(gz-ix(3)).^2)); % distance
%     u0 = p.u0(r);
    u0 = p.u0(gx,gy,gz);

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
%         imagesc(u);
%         fprintf('%g\n',t)
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

    
    uall = cat(p.xdim+1,uall{:});
    fprintf('finish GliomaFdmsolve\n');

    fdmsol.phi = phi;
    fdmsol.uall = uall;
    fdmsol.tall = tall;
    fdmsol.uend = u;
    fdmsol.DxPphi = Dx(P.*phi);
    fdmsol.DyPphi = Dy(P.*phi);
    fdmsol.DzPphi = Dz(P.*phi);

end
