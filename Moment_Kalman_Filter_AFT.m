clear
clc
close all

load("Control_HO_points_LL.mat")
load("System_HOM_30_points_LL.mat")
load("System_HOX_100_points_LL.mat")
sys_filename = ["System_HOM_10_points_LL.mat";"System_HOM_15_points_LL.mat";"System_HOM_20_points_LL.mat";"System_HOM_25_points_LL.mat";"System_HOM_30_points_LL.mat";];
sys_title = ["moment order = 10";"moment order = 15";"moment order = 20";"moment order = 25";"moment order = 30";];
sys_size = size(sys_filename, 1);
x_size = n_ensembles*dim_size;
p=3;

% Noise variables
% wv = 5e-3;
% vv = 5e-3;
wv = 1e-2;
vv = 1.5e-2;

% wvb = 5e-3;
% vvb = 5e-3;
wvb = 2e-2;
vvb = 2.5e-2;
% wvb = 0;
% vvb = 0;

% g_beta_w = @(x) (3-x+x.^2);
g_beta_w = @(x) ones(size(x));
% g_beta_v = @(x) (3.5+2*x-x.^2);
g_beta_v = @(x) ones(size(x));

g_deg = 5;
g_supp = linspace(0, 1, g_deg + 1);
g_w = g_beta_w(g_supp);
g_v = g_beta_v(g_supp);
g_Leg = zeros(g_deg + 1);
for iter=1:g_deg+1
    g_Leg_supp = legendre(iter-1,g_supp,'norm');
    g_Leg(iter, :) = g_Leg_supp(1,:);
end
g_coeff_w = linsolve((g_Leg)', g_w');
g_coeff_w(abs(g_coeff_w)<1e-5) = 0;
g_coeff_w = [g_coeff_w; zeros(m_size-g_deg-1,1)];
g_coeff_v = linsolve((g_Leg)', g_v');
g_coeff_v(abs(g_coeff_v)<1e-5) = 0;
g_coeff_v = [g_coeff_v; zeros(m_size-g_deg-1,1)];

G_beta_w = g_beta_w(omega_scl);
G_beta_v = g_beta_v(omega_scl);

% Measurements Variables
ny_min = 1;
ny_max = 1;
if ny_max<ny_min
    ny_hold = ny_min;
    ny_min = ny_max;
    ny_max = ny_hold;
end
%% Stochastic Process and Filter Performance
Axx = expm(Ax*dt);
Kxhist = zeros(n_ensembles*dim_size,sys_size);
trace_filter = zeros(iter_max, sys_size);
trace_meas = trace_filter;
trace_stoc = trace_filter;
error_x_stoc = trace_stoc;
error_x_filter = trace_stoc;
error_x_filter_2 = trace_stoc;
error_x_meas = trace_stoc;
error_stoc = zeros(m_size*dim_size, iter_max, sys_size);
error_meas = error_stoc;
error_filter = error_stoc;
cx=zeros(dim_size*n_ensembles, dim_size, iter_max+1, sys_size);
cxforward=zeros(dim_size*n_ensembles, dim_size, iter_max+1, sys_size);
cxm = zeros(dim_size*n_ensembles, dim_size*n_ensembles);

time_mom = zeros(1,sys_size);

x = x0x;
WVx = sqrt(linspace(wv^2, wv^2, n_ensembles+1))';
x_traj_dist = zeros(dim_size*n_ensembles,iter_max+1);
x_traj_dist(:,1) = x0x;
y_traj_dist = x_traj_dist;
for iter =1:iter_max
    [~, x_trajJ_fine] = adaptive_taylor_wng(p,Phi_x,Psi_p_x,[0 dt],[x;u((iter-1)*u_size+1:iter*u_size, :)], WVx, dim_size, wvb, G_beta_w); 
    x = x_trajJ_fine(end,:)'; % the end of this sequence is x[k+1]
    x = x(1:x_size);
    x_traj_dist(:,iter+1) = x;

    ny = randi([round(ny_min*n_ensembles) round(ny_max*n_ensembles)]);
    ens_id = sort(randperm(n_ensembles, ny))';
    ens_nid = setdiff(1:n_ensembles, ens_id);
    y_id = zeros(dim_size*ny,1);
    for iter2 = 1:dim_size
        y_id((iter2-1)*ny+1:iter2*ny) = ens_id*dim_size-(iter2-1);
    end
    y_id = sort(y_id);
    y_nid = setdiff(1:n_ensembles*dim_size', y_id);
    y = zeros(n_ensembles*dim_size,1);
    y(y_id) = x(y_id) + normrnd(0, vv, size(y_id)) + kron(G_beta_v', normrnd(0, vvb, dim_size, 1));
    if size(y_nid) >0
        y(y_nid) = z(y_nid);
    end
    y_traj_dist(:,iter+1) = y;
end

%% Kalman Filter in moment space
m_hist=1e6*ones(1,sys_size);
for iter_sys = 1:sys_size
    load(sys_filename(iter_sys))
    m_hist(iter_sys) = m_size;
    App = (expm(Ap*dt));
    App(abs(App)<1e-6) = 0;
    App = sparse(App);
% Discretized Moment Calculation Parameters
    prec_mom = 7.5*n_ensembles;
    P_k = zeros(m_size, size(omega_scl,2));
    P_kdim = zeros(m_size*dim_size, dim_size*size(omega_scl,2));
    ck = zeros(dim_size*m_size,1);
    for iter = 1:m_size
        P_kiter = legendre(iter-1, omega_scl, 'norm');
        P_k(iter, :) = P_kiter(1,:);
        for iter2=1:dim_size
            P_kdim(dim_size*(iter-1)+iter2,iter2:dim_size:end-dim_size+iter2) = P_kiter(1,:);
        end
        ck(dim_size*(iter-1)+1:dim_size*iter) = (iter)^2/((2*iter-1)*(2*iter+1));
    end
    normP_k = reshape(repmat(1/2:1:(2*m_size-1)/2,dim_size,1), m_size*dim_size,1);
    % typical method
    omega_prec = linspace(-1,1,prec_mom);
    Leg_poly = zeros(m_size, prec_mom);
    Leg_polydim = zeros(m_size*dim_size, dim_size*prec_mom);
    for iter = 0:m_size-1
        Leg_poly_iter = legendre(iter, omega_prec, 'norm');
        Leg_poly(iter+1,:) = Leg_poly_iter(1,:);
        for iter2=1:dim_size
            Leg_polydim(dim_size*(iter)+iter2,iter2:dim_size:end-dim_size+iter2) = Leg_poly_iter(1,:);
        end
    end

    z = x0x;
    xm = x0;
    xmnf = x0;
    x_traj_model = zeros(dim_size*n_ensembles, iter_max+1);
    x_traj_model(:,1) = x0x;
    x_traj_filter = zeros(dim_size*m_hist(iter_sys), iter_max+1);
    x_traj_filter(:,1) = x0;
    H_obs = sparse(eye(dim_size*m_size));
    c=zeros(dim_size*m_size, dim_size*m_size, iter_max+1);
    cnf = c;
    cnf(:, :, 1) = cov(diag(xm));
    cov_stoc = zeros(dim_size*m_size, dim_size*m_size, iter_max);
    cov_meas = zeros(dim_size*m_size, dim_size*m_size, iter_max);
    cov_filter = zeros(dim_size*m_size, dim_size*m_size, iter_max);

    Norm_scale = 1;
    K = zeros(dim_size*m_size);
    m_Ksize = min(m_size, 10);
    m_Kstepsize = ceil(10/2);
    m_Ksteps = ceil((m_size-10)/m_Kstepsize);

    App_K = App(1:dim_size*m_Ksize, 1:dim_size*m_Ksize);

    VV_2 = (2*vv^2/(n_ensembles-1)*eye(dim_size*m_size));
    WV_2 = (2*wv^2/(n_ensembles-1)*eye(dim_size*m_size));

    WV_2 = WV_2 + kron(g_coeff_w(1:m_size)*g_coeff_w(1:m_size)'*wvb^2, eye(dim_size));
    VV_2 = VV_2 + kron(g_coeff_v(1:m_size)*g_coeff_v(1:m_size)'*vvb^2, eye(dim_size));

    WVbkg = kron(G_beta_w'*G_beta_w, eye(dim_size))*wvb^2;
    VVbkg = kron(G_beta_v'*G_beta_v, eye(dim_size))*vvb^2;

    VV_2 = sqrtm(VV_2);
    WV_2 = sqrtm(WV_2);
    for iter = 1:iter_max
        x = x_traj_dist(:,iter+1);
        mx = initial_moment(m_size,reshape(x, dim_size, n_ensembles),omega_scl, dim_size, Leg_poly, prec_mom);
        
        tic
        Kx = zeros(n_ensembles*dim_size,1);
    
        [~, x_trajJ_fine] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[xmnf;u((iter-1)*u_size+1:iter*u_size, :)]); 
        xmnf = x_trajJ_fine(end,:)'; % the end of this sequence is x[k+1]
        xmnf = xmnf(1:m_size*dim_size);
        
        y = y_traj_dist(:,iter+1);
        my = initial_moment(m_size,reshape(y, dim_size, n_ensembles),omega_scl, dim_size, Leg_poly, prec_mom);

        cov_meas(:,:,iter) = VV_2*VV_2';

        xm = xmnf;
        d=my-H_obs*xm;% innovation

        chat=App*sparse(c(:,:,iter))*App'+WV_2*WV_2';

        K=(chat*H_obs')/(H_obs*chat*H_obs'+cov_meas(:,:,iter));% Kalman gain

        c(:,:,iter+1)=(eye(m_size*dim_size)-K)*chat;% covariance update
        xm=xm+K*d;% estimator update
        xm(1:dim_size) = xm(1:dim_size);
        x_traj_filter(:,iter+1) = xm;
        
        error_stoc(1:dim_size*m_size, iter, iter_sys) = abs(mx-xmnf);
        error_meas(1:dim_size*m_size, iter, iter_sys) = abs(mx-my);
        error_filter(1:dim_size*m_size, iter, iter_sys) = abs(mx-xm);
        cov_stoc(:,:,iter) = chat;
        cov_filter(:,:,iter) = c(:,:,iter+1);
        m_min = min(m_hist);
        trace_stoc(iter, iter_sys) = trace(cov_stoc(1:m_min*dim_size,1:m_min*dim_size,iter)*Norm_scale);
        trace_meas(iter, iter_sys) = trace(cov_meas(1:m_min*dim_size,1:m_min*dim_size,iter)*Norm_scale);
        trace_filter(iter, iter_sys) = trace(cov_filter(1:m_min*dim_size,1:m_min*dim_size,iter)*Norm_scale);
        time_mom(iter_sys) = time_mom(iter_sys) + toc;
        [~, x_trajJ_fine] = adaptive_taylor(p,Phi_x,Psi_p_x,[0 dt],[z;u((iter-1)*u_size+1:iter*u_size, :)]); 
        z = x_trajJ_fine(end,:)'; % the end of this sequence is x[k+1]
        z = z(1:x_size);        
        
        xmnf = xm;
        dy = (y-z);
        for iter2 = 1:n_ensembles
            Kx((dim_size)*(iter2-1)+1:dim_size*iter2) = linsolve(diag(P_kdim(:, dim_size*(iter2-1)+1:dim_size*iter2)'*d),P_kdim(:, dim_size*(iter2-1)+1:dim_size*iter2)'*K*d);
        end
        
        
        Kx(Kx<0) = zeros(sum(Kx<0,1),1);
        Kx(Kx>1) = ones(sum(Kx>1,1),1);
        cxforwardm = (Axx*cxm*Axx')+(wv^2*eye(dim_size*n_ensembles)+WVbkg);
        cxm = (eye(dim_size*n_ensembles)-diag(Kx))*(cxforwardm)-cxforwardm*diag(Kx)'+diag(Kx)*(cxforwardm+(vv^2)*eye(dim_size*n_ensembles)+VVbkg)*diag(Kx)';
        for iter2 = 1:n_ensembles        
            cxforward((dim_size)*(iter2-1)+1:dim_size*iter2, :, iter+1, iter_sys) = cxforwardm((dim_size)*(iter2-1)+1:dim_size*iter2,(dim_size)*(iter2-1)+1:dim_size*iter2);
            cx((dim_size)*(iter2-1)+1:dim_size*iter2, :, iter+1, iter_sys) = cxm((dim_size)*(iter2-1)+1:dim_size*iter2,(dim_size)*(iter2-1)+1:dim_size*iter2);
        end

        error_x_stoc(iter, iter_sys) = mean((vecnorm(reshape(abs(x-z), dim_size, n_ensembles), 2)));
        error_x_meas(iter, iter_sys) = mean((vecnorm(reshape(abs(x-y), dim_size, n_ensembles), 2)));
        z = z + Kx.*dy;
        error_x_filter(iter, iter_sys) = mean((vecnorm(reshape(abs(x-z), dim_size, n_ensembles), 2)));
        x_traj_model(:,iter+1) = z;
        
        Kxhist(:, iter_sys) = Kxhist(:, iter_sys) + (Kx);

        error_x_filter_2(iter, iter_sys) = mean((vecnorm(reshape(abs(x-P_kdim'*xm), dim_size, n_ensembles), 2)));
    end
    Kxhist(:, iter_sys) = Kxhist(:, iter_sys)./iter_max;
end
%% Stochastic Process and Filter Performance for state KF

x = x0x;
z = x0x;
x_traj_dist_KF = zeros(dim_size*n_ensembles,iter_max+1);
x_traj_dist_KF(:,1) = x0x;
x_traj_model_KF = x_traj_dist_KF;
x_traj_filter_KF = x_traj_dist_KF;
H_obs = eye(dim_size*n_ensembles);
cx_KF=zeros(dim_size*n_ensembles, dim_size*n_ensembles, iter_max+1, 'single');

error_x_stoc_KF = zeros(1,iter_max);
error_x_filter_KF = error_x_stoc_KF;
error_x_meas_KF = error_x_stoc_KF;

y = zeros(n_ensembles*dim_size, 1);
ac = 0*wv^2/2;
bc = 0*vv^2/2;
WVx = sqrt(linspace(wv^2-ac, wv^2+ac, n_ensembles+1))';
VVx = sqrt(linspace(vv^2-bc, vv^2+bc, n_ensembles))';
VVx = reshape(repmat(VVx, 1, dim_size)', dim_size*n_ensembles, 1);
Kxhist_KF = zeros(n_ensembles*dim_size,1);

Norm_scale = 1;

VV_2 = sqrtm((vv^2)*eye(dim_size*n_ensembles) + VVbkg);
WV_2 = sqrtm((wv^2)*eye(dim_size*n_ensembles) + WVbkg);
tic
for iter = 1:iter_max
    [~, x_trajJ_fine] = adaptive_taylor(p,Phi_x,Psi_p_x,[0 dt],[z;u((iter-1)*u_size+1:iter*u_size, :)]); 
    z = x_trajJ_fine(end,:)'; % the end of this sequence is x[k+1]
    z = z(1:x_size);
    x = x_traj_dist(:,iter+1);
    x_traj_dist_KF(:,iter+1) = x;
    y = y_traj_dist(:,iter+1);
    x_traj_model_KF(:,iter+1) = z;
    d=y-H_obs*z;% innovation
    chat=Axx*cx_KF(:,:,iter)*Axx'+WV_2*WV_2';
    K=(chat*H_obs')/(H_obs*chat*H_obs'+VV_2*VV_2');% Kalman gain
    cx_KF(:,:,iter+1)=(eye(n_ensembles*dim_size)-K)*chat;% covariance update
    error_x_stoc_KF(1,iter) = mean((vecnorm(reshape(abs(x-z), dim_size, n_ensembles), 2)));
    error_x_meas_KF(1,iter) = mean((vecnorm(reshape(abs(x-y), dim_size, n_ensembles), 2)));
    z=z+K*d;% estimator update
    error_x_filter_KF(1,iter) = mean((vecnorm(reshape(abs(x-z), dim_size, n_ensembles), 2)));
    x_traj_filter_KF(:,iter+1) = z;
        
    Kxhist_KF = Kxhist_KF + diag(K);
end
Kxhist_KF = Kxhist_KF./iter_max;
toc
%%
u_plot=reshape(u, u_size, iter_max);
fig=figure;
hold on
for iter=1:u_size
    plot(1:iter_max, u_plot(iter,:), '-','LineWidth',4);
end
set(gca,'FontSize',16)
legend({'nominal input 1', 'nominal input 2', 'nominal input 3'},'Location', 'southoutside', 'NumColumns', 3,'FontSize',16);
grid on
ylim([min(u_min) max(u_max)])
drawnow
frame = getframe(fig);
im = frame2im(frame);
[A2, map2] = rgb2ind(im, 256);
%%
rgbensemble = winter(n_ensembles);
figure;
tiledlayout(2,4, 'Padding', 'compact');
nexttile(1, [2 2]);
hold on
plot3(x_traj(n_ensembles/2*dim_size+1,:), x_traj(n_ensembles/2*dim_size+2,:), x_traj(n_ensembles/2*dim_size+3,:),  'LineWidth', 2, 'Color', [rgbensemble(1,:), 1])
grid on
xlim([min(x_traj(1:3:end-2,:),[],'all') 1.2*(max(x_traj(1:3:end-2,:),[],'all')-min(x_traj(1:3:end-2,:),[],'all'))+min(x_traj(1:3:end-2,:),[],'all')])
ylim([min(x_traj(2:3:end-1,:),[],'all') 1.2*(max(x_traj(2:3:end-1,:),[],'all')-min(x_traj(2:3:end-1,:),[],'all'))+min(x_traj(2:3:end-1,:),[],'all')])
zlim([min(x_traj(3:3:end,:),[],'all') 1.2*(max(x_traj(3:3:end,:),[],'all')-min(x_traj(3:3:end,:),[],'all'))+min(x_traj(3:3:end,:),[],'all')])
ax=gca;
ax.FontSize=24;
view([1 -1 0.5])

for iter=2:min(n_ensembles, size(x_traj,1)/dim_size)
    a = 1-0.8*(iter)/n_ensembles;
    scatter3(x_traj(3*iter-2,end), x_traj(3*iter-1,end),x_traj(3*iter,end) , 100, -1+2*iter/n_ensembles,'filled', 'LineWidth', 1, 'MarkerEdgeColor',[0 0 0], 'MarkerEdgeAlpha',a, 'MarkerFaceAlpha',a)
end
plot3(x_targetx(1,:), x_targetx(2,:), x_targetx(3,:), 'ro', 'MarkerSize',18, 'LineWidth', 2)
grid on
xlabel('State x_{1}', 'Rotation', -33)
ylabel('State x_{2}', 'Rotation', 50)
zlabel('State x_{3}')

nexttile(3, [2 2]);

p1 = plot3(x_targetx(1,:), x_targetx(2,:), x_targetx(3,:), 'ro', 'MarkerSize',18, 'LineWidth', 2);
hold on
p2 = plot3(x_traj(1,1), x_traj(2,1), x_traj(3,1), 'k.', 'MarkerSize',40, 'LineWidth', 2);
leg = legend([p2 p1], 'Initial state(x_{0})', 'Target state(x_{T})', 'Location', 'southoutside', 'NumColumns', 3, 'AutoUpdate','off');
leg.Layout.Tile = 'north';
plot3(x_traj(1,:), x_traj(2,:), x_traj(3,:),  'LineWidth', 2, 'Color', [rgbensemble(1,:), 1])
for iter=2:n_ensembles
    r = rgbensemble(iter,1);
    g = rgbensemble(iter,2);
    b = rgbensemble(iter,3);
    a = 1-0.8*(iter)/n_ensembles;
    plot3(x_traj(dim_size*iter-2,:), x_traj(dim_size*iter-1,:), x_traj(dim_size*iter,:),'LineWidth', 2, 'Color', [r, g, b, a])
end
plot3(x_targetx(1,:), x_targetx(2,:), x_targetx(3,:), 'ro', 'MarkerSize',18, 'LineWidth', 2);
grid on
xlim([min(x_traj(1:3:end-2,:),[],'all') 1.2*(max(x_traj(1:3:end-2,:),[],'all')-min(x_traj(1:3:end-2,:),[],'all'))+min(x_traj(1:3:end-2,:),[],'all')])
ylim([min(x_traj(2:3:end-1,:),[],'all') 1.2*(max(x_traj(2:3:end-1,:),[],'all')-min(x_traj(2:3:end-1,:),[],'all'))+min(x_traj(2:3:end-1,:),[],'all')])
zlim([min(x_traj(3:3:end,:),[],'all') 1.2*(max(x_traj(3:3:end,:),[],'all')-min(x_traj(3:3:end,:),[],'all'))+min(x_traj(3:3:end,:),[],'all')])
ax=gca;
ax.FontSize=24;
view([1 -1 0.5])

for iter=2:min(n_ensembles, size(x_traj,1)/dim_size)
    a = 1-0*(iter)/n_ensembles;
    scatter3(x_traj(3*iter-2,end), x_traj(3*iter-1,end),x_traj(3*iter,end) , 100, -1+2*iter/n_ensembles,'filled', 'LineWidth', 1, 'MarkerEdgeColor',[0 0 0], 'MarkerEdgeAlpha',a, 'MarkerFaceAlpha',a)
end
plot3(3.4, -0.5, 0.1, 'o', 'MarkerSize',16, 'LineWidth', 2, 'MarkerEdgeColor', [1, 0, 0]);
grid on
xlabel('State x_{1}', 'Rotation', -33)
ylabel('State x_{2}', 'Rotation', 50)
zlabel('State x_{3}')

colormap winter
cor = colorbar;
cor.Ticks = -0.95:0.195:1;
cor.TickLabels = ["-1" "-0.8" "-0.6" "-0.4" "-0.2" "0" "0.2" "0.4" "0.6" "0.8" "1"];
ylabel(cor,'Parameter \beta','FontSize',24);
hColourbar.Label.Position(1) = 3;

set(gcf, 'Position',  [200, 250, 950, 650])

%%
rgbensemble = winter(n_ensembles);
figure;
tillay = tiledlayout(2,4, 'Padding', 'compact');
nexttile(1, [2 2]);
hold on
plot3(x_traj(1,:), x_traj(2,:), x_traj(3,:),  'LineWidth', 2, 'Color', [rgbensemble(1,:), 1])
for iter=2:n_ensembles
    r = rgbensemble(iter,1);
    g = rgbensemble(iter,2);
    b = rgbensemble(iter,3);
    a = 1-0.8*(iter)/n_ensembles;
    plot3(x_traj(dim_size*iter-2,:), x_traj(dim_size*iter-1,:), x_traj(dim_size*iter,:),'LineWidth', 2, 'Color', [r, g, b, a])
end
grid on
xlim([min(x_traj(1:3:end-2,:),[],'all') 1.2*(max(x_traj(1:3:end-2,:),[],'all')-min(x_traj(1:3:end-2,:),[],'all'))+min(x_traj(1:3:end-2,:),[],'all')])
ylim([min(x_traj(2:3:end-1,:),[],'all') 1.2*(max(x_traj(2:3:end-1,:),[],'all')-min(x_traj(2:3:end-1,:),[],'all'))+min(x_traj(2:3:end-1,:),[],'all')])
zlim([min(x_traj(3:3:end,:),[],'all') 1.2*(max(x_traj(3:3:end,:),[],'all')-min(x_traj(3:3:end,:),[],'all'))+min(x_traj(3:3:end,:),[],'all')])
ax=gca;
ax.FontSize=24;
view([1 -1 0.5])

for iter=2:min(n_ensembles, size(x_traj,1)/dim_size)
    a = 1-0.8*(iter)/n_ensembles;
    scatter3(x_traj(3*iter-2,end), x_traj(3*iter-1,end),x_traj(3*iter,end) , 100, -1+2*iter/n_ensembles,'filled', 'LineWidth', 1, 'MarkerEdgeColor',[0 0 0], 'MarkerEdgeAlpha',a, 'MarkerFaceAlpha',a)
end
plot3(x_targetx(1,:), x_targetx(2,:), x_targetx(3,:), 'ro', 'MarkerSize',18, 'LineWidth', 2)
grid on
xlabel('State x_{1}', 'Rotation', -33)
ylabel('State x_{2}', 'Rotation', 50)
zlabel('State x_{3}')

nexttile(3, [2 2]);

p1 = plot3(x_targetx(1,:), x_targetx(2,:), x_targetx(3,:), 'ro', 'MarkerSize',15, 'LineWidth', 2);
hold on
p2 = plot3(x_traj_dist(1,1), x_traj_dist(2,1), x_traj_dist(3,1), 'k.', 'MarkerSize',40, 'LineWidth', 2);
leg = legend([p2 p1], 'Initial state(x_{0})', 'Target state(x_{T})', 'Location', 'southoutside', 'NumColumns', 3, 'AutoUpdate','off');
leg.Layout.Tile = 'north';
plot3(x_traj_dist(1,:), x_traj_dist(2,:), x_traj_dist(3,:),  'LineWidth', 2, 'Color', [rgbensemble(1,:), 1])
for iter=2:n_ensembles
    r = rgbensemble(iter,1);
    g = rgbensemble(iter,2);
    b = rgbensemble(iter,3);
    a = 1-0.8*(iter)/n_ensembles;
    plot3(x_traj_dist(dim_size*iter-2,:), x_traj_dist(dim_size*iter-1,:), x_traj_dist(dim_size*iter,:),'LineWidth', 2, 'Color', [r, g, b, a])
end
plot3(x_targetx(1,:), x_targetx(2,:), x_targetx(3,:), 'ro', 'MarkerSize',15, 'LineWidth', 2);
grid on
xlim([min(x_traj_dist(1:3:end-2,:),[],'all') 1.2*(max(x_traj_dist(1:3:end-2,:),[],'all')-min(x_traj_dist(1:3:end-2,:),[],'all'))+min(x_traj_dist(1:3:end-2,:),[],'all')])
ylim([min(x_traj_dist(2:3:end-1,:),[],'all') 1.2*(max(x_traj_dist(2:3:end-1,:),[],'all')-min(x_traj_dist(2:3:end-1,:),[],'all'))+min(x_traj_dist(2:3:end-1,:),[],'all')])
zlim([min(x_traj_dist(3:3:end,:),[],'all') 1.2*(max(x_traj_dist(3:3:end,:),[],'all')-min(x_traj_dist(3:3:end,:),[],'all'))+min(x_traj_dist(3:3:end,:),[],'all')])
ax=gca;
ax.FontSize=24;
view([1 -1 0.5])

for iter=2:min(n_ensembles, size(x_traj_dist,1)/dim_size)
    a = 1-0*(iter)/n_ensembles;
    scatter3(x_traj_dist(3*iter-2,end), x_traj_dist(3*iter-1,end),x_traj_dist(3*iter,end) , 100, -1+2*iter/n_ensembles,'filled', 'LineWidth', 1, 'MarkerEdgeColor',[0 0 0], 'MarkerEdgeAlpha',a, 'MarkerFaceAlpha',a)
end
plot3(3.4, -0.5, 0.1, 'o', 'MarkerSize',16, 'LineWidth', 2, 'MarkerEdgeColor', [1, 0, 0]);
grid on
xlabel('State x_{1}', 'Rotation', -33)
ylabel('State x_{2}', 'Rotation', 50)
zlabel('State x_{3}')

colormap winter
cor = colorbar;
cor.Ticks = -0.95:0.195:1;
cor.TickLabels = ["-1" "-0.8" "-0.6" "-0.4" "-0.2" "0" "0.2" "0.4" "0.6" "0.8" "1"];
ylabel(cor,'Parameter \beta','FontSize',24);
hColourbar.Label.Position(1) = 3;
plot3(x_targetx(1,:), x_targetx(2,:), x_targetx(3,:), 'ro', 'MarkerSize',15, 'LineWidth', 2)

hold on
ax2 = axes(tillay);
ax2.Layout.Tile = 2;
ax2.Layout.TileSpan = [1 1];
plot3(3.139, -0.013, 0.005, 'ro', 'MarkerSize',16, 'LineWidth', 2)
hold on
for iter=1:n_ensembles
    r = rgbensemble(iter,1);
    g = rgbensemble(iter,2);
    b = rgbensemble(iter,3);
    plot3(x_traj(dim_size*iter-2,end), x_traj(dim_size*iter-1,end), x_traj(dim_size*iter,end), 'o', 'MarkerSize',10, 'LineWidth', 1,'MarkerFaceColor', [r, g, b], 'MarkerEdgeColor', [0, 0, 0])
end
grid on
ax=gca;
ax.FontSize=13;
ax.XAxis.Exponent = 0;
ax.YAxis.Exponent = 0;
ax.ZAxis.Exponent = 0;

hold on
ax2 = axes(tillay);
ax2.Layout.Tile = 4;
ax2.Layout.TileSpan = [1 1];
plot3(2.6, -4, 1.3, 'ro', 'MarkerSize',16, 'LineWidth', 2)
hold on
for iter=1:n_ensembles
    r = rgbensemble(iter,1);
    g = rgbensemble(iter,2);
    b = rgbensemble(iter,3);
    plot3(x_traj_dist(dim_size*iter-2,end), x_traj_dist(dim_size*iter-1,end), x_traj_dist(dim_size*iter,end), 'o', 'MarkerSize',10, 'LineWidth', 1,'MarkerFaceColor', [r, g, b], 'MarkerEdgeColor', [0, 0, 0])
end
%     plot(x_traj(1:2:end-1,end), x_traj(2:2:end,end), 'g.', 'MarkerSize',10, 'LineWidth', 1)
grid on
ax=gca;
ax.FontSize=13;

set(gcf, 'Position',  [200, 250, 950, 650])

%%
axis_min_x = 0;
axis_max_x = iter_max;
axis_min_y = 0;
axis_max_y = [5e-2 repmat(5e-3, 1, 5)];
cum_divider = 1:1:ceil(iter_max/10);

figure
tiledlayout(2,3)
colororder([0 0 1; 0 0.75 0])
cumperformstoc = zeros(dim_size*m_size,1);
cumperformfilter = cumperformstoc;
cumperformmeas = cumperformstoc;
perform_moment = cumperformstoc;

leg_names = {'Estimation Average error', 'Filtered Average error', 'Measurement Average error'};
for iter = 1:6
    yyaxis left
    plot_stoc = cumsum(error_stoc(dim_size*iter-2,1:10:end, end))./cum_divider;
    plot_filter = cumsum(error_filter(dim_size*iter-2,1:10:end, end))./cum_divider;
    plot_meas = cumsum(error_meas(dim_size*iter-2,1:10:end, end))./cum_divider;
    cumperformstoc(dim_size*iter-2) = plot_stoc(end);
    cumperformfilter(dim_size*iter-2) = plot_filter(end);
    cumperformmeas(dim_size*iter-2) = plot_meas(end);
    
    nexttile;
    plot(1:10:iter_max,plot_stoc,'-','MarkerSize',12,'Color', [1 0 1 1],'Linewidth',3);
    hold on
    plot(1:10:iter_max,plot_filter,'-','MarkerSize',12,'Color', [0 0 1 1],'Linewidth',3);
    plot(1:10:iter_max,plot_meas,'-','MarkerSize',12,'Color', [0 0 0 1],'Linewidth',3);
    grid; hold;xlabel('step(\tau)');ylabel('Average Error');
    perform_moment(dim_size*iter-2) = cumperformfilter(dim_size*iter-2)./min([cumperformstoc(dim_size*iter-2) cumperformmeas(dim_size*iter-2)]);
    xlim([axis_min_x axis_max_x])
    ylim([axis_min_y axis_max_y(iter)])
    title(['m_{' num2str(iter-1) '}(x_{1})'])
    set(gca,'FontSize',24)
    
    plot_x = 1:10:iter_max;
    plot_index = plot_filter./min([plot_stoc;plot_meas]);
    yyaxis right
    hold on
    ylabel('{e_{filter}}/{e_{best}}', 'Interpreter','tex');
    if plot_index(1)>1
        isred = 1;
        if iter == 6 && size(leg_names,2)==3
            leg_names(4) = {'Deteriorated Performance'};
            leg_names(5) = {'Improved Performance'};
        end
    else
        isred = 0;
        if iter == 6 && size(leg_names,2)==3
            leg_names(4) = {'Improved Performance'};
            leg_names(5) = {'Deteriorated Performance'};
        end
    end
    xa = 1;
    ya = plot_index(1);
    itera = 2;
    for iter2 = 2:size(plot_index, 2)
        if (isred == 0 && plot_index(iter2)>1) || (isred == 1 && plot_index(iter2)<=1)
            xa2 = interp1([plot_index(iter2-1) plot_index(iter2)], [plot_x(iter2-1) plot_x(iter2)], 1);
            ax = [xa plot_x(itera:iter2-1) xa2];
            ay = [ya plot_index(itera:iter2-1) 1];
            a = area(ax, ay);
            a.LineWidth = 2;
            a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
            a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
            a.FaceAlpha = 0.1;
            a.EdgeAlpha = 0.1;
            xa = xa2;
            ya = 1;
            itera = iter2;
            isred = 1-isred;
        end
    end
    ax = [xa plot_x(itera:end)];
    ay = [ya plot_index(itera:end)];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
    ylim([axis_min_y 1.25])
end
if isred == 0
    isred = 1;
    ax = [plot_x(end) plot_x(end)+1];
    ay = [0.01 0.01];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
else
    isred = 0;
    ax = [plot_x(end) plot_x(end)+1];
    ay = [0.01 0.01];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
end
leg = legend(leg_names, 'Orientation', 'Horizontal','NumColumns',3,'FontSize',24);
leg.Layout.Tile = 'north';
set(gcf, 'Position',  [200, 250, 1350, 750])

figure
tiledlayout(2,3)
colororder([0 0 1; 0 0.75 0])
leg_names = {'Estimation Average error', 'Filtered Average error', 'Measurement Average error'};
for iter = 1:6
    yyaxis left
    plot_stoc = cumsum(error_stoc(dim_size*iter-1,1:10:end, end))./(1:1:ceil(iter_max/10));
    plot_filter = cumsum(error_filter(dim_size*iter-1,1:10:end, end))./(1:1:ceil(iter_max/10));
    plot_meas = cumsum(error_meas(dim_size*iter-1,1:10:end, end))./(1:1:ceil(iter_max/10));
    cumperformstoc(dim_size*iter-1) = plot_stoc(end);
    cumperformfilter(dim_size*iter-1) = plot_filter(end);
    cumperformmeas(dim_size*iter-1) = plot_meas(end);
    
    nexttile;
    plot(1:10:iter_max,plot_stoc,'-','MarkerSize',12,'Color', [1 0 1 1],'Linewidth',3);
    hold on
    plot(1:10:iter_max,plot_filter,'-','MarkerSize',12,'Color', [0 0 1 1],'Linewidth',3);
    plot(1:10:iter_max,plot_meas,'-','MarkerSize',12,'Color', [0 0 0 1],'Linewidth',3);
    grid; hold;xlabel('step(\tau)');ylabel('Average Error');
    perform_moment(dim_size*iter-1) = cumperformfilter(dim_size*iter-1)./min([cumperformstoc(dim_size*iter-1) cumperformmeas(dim_size*iter-1)]);
    xlim([axis_min_x axis_max_x])
    ylim([axis_min_y axis_max_y(iter)])
    title(['m_{' num2str(iter-1) '}(x_{2})'])
    set(gca,'FontSize',24)
    
    plot_x = 1:10:iter_max;
    plot_index = plot_filter./min([plot_stoc;plot_meas]);
    yyaxis right
    hold on
    ylabel('{e_{filter}}/{e_{best}}', 'Interpreter','tex');
    if plot_index(1)>1
        isred = 1;
        if iter == 6 && size(leg_names,2)==3
            leg_names(4) = {'Deteriorated Performance'};
            leg_names(5) = {'Improved Performance'};
        end
    else
        isred = 0;
        if iter == 6 && size(leg_names,2)==3
            leg_names(4) = {'Improved Performance'};
            leg_names(5) = {'Deteriorated Performance'};
        end
    end
    xa = 1;
    ya = plot_index(1);
    itera = 2;
    for iter2 = 2:size(plot_index, 2)
        if (isred == 0 && plot_index(iter2)>1) || (isred == 1 && plot_index(iter2)<=1)
            xa2 = interp1([plot_index(iter2-1) plot_index(iter2)], [plot_x(iter2-1) plot_x(iter2)], 1);
            ax = [xa plot_x(itera:iter2-1) xa2];
            ay = [ya plot_index(itera:iter2-1) 1];
            a = area(ax, ay);
            a.LineWidth = 2;
            a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
            a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
            a.FaceAlpha = 0.1;
            a.EdgeAlpha = 0.1;
            xa = xa2;
            ya = 1;
            itera = iter2;
            isred = 1-isred;
        end
    end
    ax = [xa plot_x(itera:end)];
    ay = [ya plot_index(itera:end)];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
    ylim([axis_min_y 1.25])
end
if isred == 0
    isred = 1;
    ax = [plot_x(end) plot_x(end)+1];
    ay = [0.01 0.01];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
else
    isred = 0;
    ax = [plot_x(end) plot_x(end)+1];
    ay = [0.01 0.01];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
end
leg = legend(leg_names, 'Orientation', 'Horizontal','NumColumns',3,'FontSize',24);
leg.Layout.Tile = 'north';
set(gcf, 'Position',  [200, 250, 1350, 750])

figure
tiledlayout(2,3)
colororder([0 0 1; 0 0.75 0])
leg_names = {'Estimation Average error', 'Filtered Average error', 'Measurement Average error'};
for iter = 1:6
    yyaxis left
    plot_stoc = cumsum(error_stoc(dim_size*iter,1:10:end, end))./(1:1:ceil(iter_max/10));
    plot_filter = cumsum(error_filter(dim_size*iter,1:10:end, end))./(1:1:ceil(iter_max/10));
    plot_meas = cumsum(error_meas(dim_size*iter,1:10:end, end))./(1:1:ceil(iter_max/10));
    cumperformstoc(dim_size*iter) = plot_stoc(end);
    cumperformfilter(dim_size*iter) = plot_filter(end);
    cumperformmeas(dim_size*iter) = plot_meas(end);
    
    nexttile;
    plot(1:10:iter_max,plot_stoc,'-','MarkerSize',12,'Color', [1 0 1 1],'Linewidth',3);
    hold on
    plot(1:10:iter_max,plot_filter,'-','MarkerSize',12,'Color', [0 0 1 1],'Linewidth',3);
    plot(1:10:iter_max,plot_meas,'-','MarkerSize',12,'Color', [0 0 0 1],'Linewidth',3);
    grid; hold;xlabel('step(\tau)');ylabel('Average Error');
    perform_moment(dim_size*iter) = cumperformfilter(dim_size*iter)./min([cumperformstoc(dim_size*iter) cumperformmeas(dim_size*iter)]);
    xlim([axis_min_x axis_max_x])
    ylim([axis_min_y axis_max_y(iter)])
    title(['m_{' num2str(iter-1) '}(x_{3})'])
    set(gca,'FontSize',24)
    
    plot_x = 1:10:iter_max;
    plot_index = plot_filter./min([plot_stoc;plot_meas]);
    yyaxis right
    hold on
    ylabel('{e_{filter}}/{e_{best}}', 'Interpreter','tex');
    if plot_index(1)>1
        isred = 1;
        if iter == 6 && size(leg_names,2)==3
            leg_names(4) = {'Deteriorated Performance'};
            leg_names(5) = {'Improved Performance'};
        end
    else
        isred = 0;
        if iter == 6 && size(leg_names,2)==3
            leg_names(4) = {'Improved Performance'};
            leg_names(5) = {'Deteriorated Performance'};
        end
    end
    xa = 1;
    ya = plot_index(1);
    itera = 2;
    for iter2 = 2:size(plot_index, 2)
        if (isred == 0 && plot_index(iter2)>1) || (isred == 1 && plot_index(iter2)<=1)
            xa2 = interp1([plot_index(iter2-1) plot_index(iter2)], [plot_x(iter2-1) plot_x(iter2)], 1);
            ax = [xa plot_x(itera:iter2-1) xa2];
            ay = [ya plot_index(itera:iter2-1) 1];
            a = area(ax, ay);
            a.LineWidth = 2;
            a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
            a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
            a.FaceAlpha = 0.1;
            a.EdgeAlpha = 0.1;
            xa = xa2;
            ya = 1;
            itera = iter2;
            isred = 1-isred;
        end
    end
    ax = [xa plot_x(itera:end)];
    ay = [ya plot_index(itera:end)];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
    ylim([axis_min_y 1.25])
end
if isred == 0
    isred = 1;
    ax = [plot_x(end) plot_x(end)+1];
    ay = [0.01 0.01];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
else
    isred = 0;
    ax = [plot_x(end) plot_x(end)+1];
    ay = [0.01 0.01];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
end
leg = legend(leg_names, 'Orientation', 'Horizontal','NumColumns',3,'FontSize',24);
leg.Layout.Tile = 'north';
set(gcf, 'Position',  [200, 250, 1350, 750])

for iter=7:m_size
    for iter2 = 1:dim_size
        plot_stoc = cumsum(error_stoc(dim_size*iter+1-iter2,21:10:end, end))./(3:1:ceil(iter_max/10));
        plot_filter = cumsum(error_filter(dim_size*iter+1-iter2,21:10:end, end))./(3:1:ceil(iter_max/10));
        plot_meas = cumsum(error_meas(dim_size*iter+1-iter2,21:10:end, end))./(3:1:ceil(iter_max/10));
        cumperformstoc(dim_size*iter+1-iter2) = plot_stoc(end);
        cumperformfilter(dim_size*iter+1-iter2) = plot_filter(end);
        cumperformmeas(dim_size*iter+1-iter2) = plot_meas(end);
        perform_moment(dim_size*iter+1-iter2) = cumperformfilter(dim_size*iter+1-iter2)./min([cumperformstoc(dim_size*iter+1-iter2) cumperformmeas(dim_size*iter+1-iter2)]);
    end
end

plot_id = [1 1 1 m_size m_size m_size];
figure
tiledlayout(2,3, 'Padding', 'Compact')
colororder([0 0 1; 0 0.75 0])
leg_names = {'Estimation Average error', 'Filtered Average error', 'Measurement Average error'};
for iter=1:6
    yyaxis left
    if mod(iter, 3) == 1
        plot_stoc = cumsum(error_stoc(dim_size*plot_id(iter)-2,1:10:end, end))./(1:1:ceil(iter_max/10));
        plot_filter = cumsum(error_filter(dim_size*plot_id(iter)-2,1:10:end, end))./(1:1:ceil(iter_max/10));
        plot_meas = cumsum(error_meas(dim_size*plot_id(iter)-2,1:10:end, end))./(1:1:ceil(iter_max/10));

        nexttile(iter);
    plot(1:10:iter_max,plot_stoc,'-','MarkerSize',12,'Color', [1 0 1 1],'Linewidth',3);
    hold on
    plot(1:10:iter_max,plot_filter,'-','MarkerSize',12,'Color', [0 0 1 1],'Linewidth',3);
    plot(1:10:iter_max,plot_meas,'-','MarkerSize',12,'Color', [0 0 0 1],'Linewidth',3);
    grid; hold;xlabel('step(\tau)');ylabel('Average Error');
    perform_moment(dim_size*iter) = cumperformfilter(dim_size*iter)./min([cumperformstoc(dim_size*iter) cumperformmeas(dim_size*iter)]);
    xlim([axis_min_x axis_max_x])
    ylim([axis_min_y axis_max_y(min(plot_id(iter), 6))])
    title(['m_{' num2str(plot_id(iter)-1) '}(x_{1})'])
    set(gca,'FontSize',24)
    
    plot_x = 1:10:iter_max;
    plot_index = plot_filter./min([plot_stoc;plot_meas]);
    yyaxis right
    hold on
    ylabel('{e_{filter}}/{e_{best}}', 'Interpreter','tex');
    if plot_index(1)>1
        isred = 1;
    else
        isred = 0;
    end
    xa = 1;
    ya = plot_index(1);
    itera = 2;
    for iter2 = 2:size(plot_index, 2)
        if (isred == 0 && plot_index(iter2)>1) || (isred == 1 && plot_index(iter2)<=1)
            xa2 = interp1([plot_index(iter2-1) plot_index(iter2)], [plot_x(iter2-1) plot_x(iter2)], 1);
            ax = [xa plot_x(itera:iter2-1) xa2];
            ay = [ya plot_index(itera:iter2-1) 1];
            a = area(ax, ay);
            a.LineWidth = 2;
            a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
            a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
            a.FaceAlpha = 0.1;
            a.EdgeAlpha = 0.1;
            xa = xa2;
            ya = 1;
            itera = iter2;
            isred = 1-isred;
        end
    end
    ax = [xa plot_x(itera:end)];
    ay = [ya plot_index(itera:end)];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
    ylim([axis_min_y 1.25])
    elseif mod(iter, 3) == 2
        plot_stoc = cumsum(error_stoc(dim_size*plot_id(iter)-1,1:10:end, end))./(1:1:ceil(iter_max/10));
        plot_filter = cumsum(error_filter(dim_size*plot_id(iter)-1,1:10:end, end))./(1:1:ceil(iter_max/10));
        plot_meas = cumsum(error_meas(dim_size*plot_id(iter)-1,1:10:end, end))./(1:1:ceil(iter_max/10));

        nexttile(iter);
    plot(1:10:iter_max,plot_stoc,'-','MarkerSize',12,'Color', [1 0 1 1],'Linewidth',3);
    hold on
    plot(1:10:iter_max,plot_filter,'-','MarkerSize',12,'Color', [0 0 1 1],'Linewidth',3);
    plot(1:10:iter_max,plot_meas,'-','MarkerSize',12,'Color', [0 0 0 1],'Linewidth',3);
    grid; hold;xlabel('step(\tau)');ylabel('Average Error');
    perform_moment(dim_size*iter) = cumperformfilter(dim_size*iter)./min([cumperformstoc(dim_size*iter) cumperformmeas(dim_size*iter)]);
    xlim([axis_min_x axis_max_x])
    ylim([axis_min_y axis_max_y(min(plot_id(iter), 6))])
    title(['m_{' num2str(plot_id(iter)-1) '}(x_{2})'])
    set(gca,'FontSize',24)
    
    plot_x = 1:10:iter_max;
    plot_index = plot_filter./min([plot_stoc;plot_meas]);
    yyaxis right
    hold on
    ylabel('{e_{filter}}/{e_{best}}', 'Interpreter','tex');
    if plot_index(1)>1
        isred = 1;
    else
        isred = 0;
    end
    xa = 1;
    ya = plot_index(1);
    itera = 2;
    for iter2 = 2:size(plot_index, 2)
        if (isred == 0 && plot_index(iter2)>1) || (isred == 1 && plot_index(iter2)<=1)
            xa2 = interp1([plot_index(iter2-1) plot_index(iter2)], [plot_x(iter2-1) plot_x(iter2)], 1);
            ax = [xa plot_x(itera:iter2-1) xa2];
            ay = [ya plot_index(itera:iter2-1) 1];
            a = area(ax, ay);
            a.LineWidth = 2;
            a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
            a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
            a.FaceAlpha = 0.1;
            a.EdgeAlpha = 0.1;
            xa = xa2;
            ya = 1;
            itera = iter2;
            isred = 1-isred;
        end
    end
    ax = [xa plot_x(itera:end)];
    ay = [ya plot_index(itera:end)];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
    ylim([axis_min_y 1.25])
    else
        plot_stoc = cumsum(error_stoc(dim_size*plot_id(iter),1:10:end, end))./(1:1:ceil(iter_max/10));
        plot_filter = cumsum(error_filter(dim_size*plot_id(iter),1:10:end, end))./(1:1:ceil(iter_max/10));
        plot_meas = cumsum(error_meas(dim_size*plot_id(iter),1:10:end, end))./(1:1:ceil(iter_max/10));
        
        nexttile(iter);
    plot(1:10:iter_max,plot_stoc,'-','MarkerSize',12,'Color', [1 0 1 1],'Linewidth',3);
    hold on
    plot(1:10:iter_max,plot_filter,'-','MarkerSize',12,'Color', [0 0 1 1],'Linewidth',3);
    plot(1:10:iter_max,plot_meas,'-','MarkerSize',12,'Color', [0 0 0 1],'Linewidth',3);
    grid; hold;xlabel('step(\tau)');ylabel('Average Error');
    perform_moment(dim_size*iter) = cumperformfilter(dim_size*iter)./min([cumperformstoc(dim_size*iter) cumperformmeas(dim_size*iter)]);
    xlim([axis_min_x axis_max_x])
    ylim([axis_min_y axis_max_y(min(plot_id(iter), 6))])
    title(['m_{' num2str(plot_id(iter)-1) '}(x_{3})'])
    set(gca,'FontSize',24)
    
    plot_x = 1:10:iter_max;
    plot_index = plot_filter./min([plot_stoc;plot_meas]);
    yyaxis right
    hold on
    ylabel('{e_{filter}}/{e_{best}}', 'Interpreter','tex');
    if plot_index(1)>1
        isred = 1;
        if iter == 6 && size(leg_names,2)==3
            leg_names(4) = {'Deteriorated Performance'};
            leg_names(5) = {'Improved Performance'};
        end
    else
        isred = 0;
        if iter == 6 && size(leg_names,2)==3
            leg_names(4) = {'Improved Performance'};
            leg_names(5) = {'Deteriorated Performance'};
        end
    end
    xa = 1;
    ya = plot_index(1);
    itera = 2;
    for iter2 = 2:size(plot_index, 2)
        if (isred == 0 && plot_index(iter2)>1) || (isred == 1 && plot_index(iter2)<=1)
            xa2 = interp1([plot_index(iter2-1) plot_index(iter2)], [plot_x(iter2-1) plot_x(iter2)], 1);
            ax = [xa plot_x(itera:iter2-1) xa2];
            ay = [ya plot_index(itera:iter2-1) 1];
            a = area(ax, ay);
            a.LineWidth = 2;
            a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
            a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
            a.FaceAlpha = 0.1;
            a.EdgeAlpha = 0.1;
            xa = xa2;
            ya = 1;
            itera = iter2;
            isred = 1-isred;
        end
    end
    ax = [xa plot_x(itera:end)];
    ay = [ya plot_index(itera:end)];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
    ylim([axis_min_y 1.25])
    end
end
if isred == 0
    isred = 1;
    ax = [plot_x(end) plot_x(end)+1];
    ay = [0.01 0.01];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
else
    isred = 0;
    ax = [plot_x(end) plot_x(end)+1];
    ay = [0.01 0.01];
    a = area(ax, ay);
    a.LineWidth = 2;
    a.FaceColor = [isred*0.75 (1-isred)*0.75 0];
    a.EdgeColor = [isred*0.75 (1-isred)*0.75 0];
    a.FaceAlpha = 0.1;
    a.EdgeAlpha = 0.1;
end

leg = legend(leg_names, 'Orientation', 'Horizontal','NumColumns',3,'FontSize',24);
leg.Layout.Tile = 'north';
set(gcf, 'Position',  [200, 250, 1350, 750])
%%
cumperformstoc = zeros(dim_size*m_size, 1);
cumperformfilter = cumperformstoc;
cumperformmeas = cumperformstoc;
perform_avg = mean(perform_moment);


figure
tiledlayout(1,3, 'Padding', 'Compact');
nexttile;
cum_trace_stoc = cumsum(reshape(trace_stoc(11:end,[1 3 5]),iter_max-10,3))'./(1:iter_max-10);
cum_trace_meas = cumsum(reshape(trace_meas(11:end,[1 3 5]),iter_max-10,3))'./(1:iter_max-10);
cum_trace_filter = cumsum(reshape(trace_filter(11:end,[1 3 5]),iter_max-10,3))'./(1:iter_max-10);
plot_iters = 11:iter_max;

plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_trace_stoc(1,:),'m-','Linewidth',3);
hold on
plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cumsum(reshape(trace_meas(11:end,1),iter_max-10,1))./(1:iter_max-10)','k-','Linewidth',3);
plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cumsum(reshape(trace_filter(11:end,1),iter_max-10,1))./(1:iter_max-10)','b-','Linewidth',3);
% 
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_trace_stoc(2,:),'m-','Linewidth',3);
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cumsum(reshape(trace_meas(11:end,3),iter_max-10,1))./(1:iter_max-10)','k-','Linewidth',3);
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cumsum(reshape(trace_filter(11:end,3),iter_max-10,1))./(1:iter_max-10)','b-','Linewidth',3);
% 
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_trace_stoc(3,:),'m-','Linewidth',3);
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cumsum(reshape(trace_meas(11:end,5),iter_max-10,1))./(1:iter_max-10)','k-','Linewidth',3);
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cumsum(reshape(trace_filter(11:end,5),iter_max-10,1))./(1:iter_max-10)','b-','Linewidth',3);
view([55 10])

Xsurf = [m_hist(1) m_hist(3) m_hist(5)];
Ysurf = 11:iter_max;
[X, Y] = meshgrid(Xsurf, Ysurf);
Z = cumsum(reshape(trace_filter(11:end,1),iter_max-10,1))./(1:iter_max-10)';
Z = [Z cumsum(reshape(trace_filter(11:end,3),iter_max-10,1))./(1:iter_max-10)'];
Z = [Z cumsum(reshape(trace_filter(11:end,5),iter_max-10,1))./(1:iter_max-10)'];
CO = [];
CO(:,:,1) = zeros(iter_max-10, 3); % red
CO(:,:,2) = zeros(iter_max-10, 3); % green
CO(:,:,3) = ones(iter_max-10, 3); % blue
s = surf(X,Y,Z, CO, 'FaceAlpha',0.2);
s.EdgeColor = 'none';

CO = [];
CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.8*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(1), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.25);
s.EdgeColor = 'none';

CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.6*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(3), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.15);
s.EdgeColor = 'none';

CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.4*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(5), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.05);
s.EdgeColor = 'none';

grid; hold on;xlabel('N');ylabel('step(\tau)', 'Rotation', 30); zlabel('Trace of covariance');
axis_top = max([cum_trace_meas(:,end) cum_trace_stoc(:,end)],[],'all');
axis_low = min(cum_trace_filter(:,end),[],'all');
axis([m_hist(1) m_hist(5) 100 iter_max axis_low-0.4*(axis_top-axis_low) axis_top+0.7*(axis_top-axis_low)]);

ax = gca;
ax.FontSize = 24;

set(gca, 'Ygrid', 'off')
set(gca, 'Xgrid', 'off')
set(gca, 'GridLineWidth', 2)
set(gca, 'GridAlpha', 0.6)
set(gcf, 'InvertHardcopy', 'off')

nexttile;
cum_error_stoc = cumsum(reshape([error_stoc(dim_size,11:end,1);error_stoc(dim_size,11:end,3);error_stoc(dim_size,11:end,5)]',iter_max-10,3))'./[1:iter_max-10;1:iter_max-10;1:iter_max-10];
cum_error_meas = cumsum(reshape([error_meas(dim_size,11:end,1);error_meas(dim_size,11:end,3);error_meas(dim_size,11:end,5)]',iter_max-10,3))'./[1:iter_max-10;1:iter_max-10;1:iter_max-10];
cum_error_filter = cumsum(reshape([error_filter(dim_size,11:end,1);error_filter(dim_size,11:end,3);error_filter(dim_size,11:end,5)]',iter_max-10,3))'./[1:iter_max-10;1:iter_max-10;1:iter_max-10];

plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_error_stoc(1,:),'m-','Linewidth',3);
hold on
plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_error_meas(1,:),'k-','Linewidth',3);
plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_error_filter(1,:),'b-','Linewidth',3);
% 
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_error_stoc(2,:),'m-','Linewidth',3);
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_error_meas(2,:),'k-','Linewidth',3);
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_error_filter(2,:),'b-','Linewidth',3);
% 
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_error_stoc(3,:),'m-','Linewidth',3);
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_error_meas(3,:),'k-','Linewidth',3);
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_error_filter(3,:),'b-','Linewidth',3);
view([55 10])

Xsurf = [m_hist(1) m_hist(3) m_hist(5)];
Ysurf = 11:iter_max;
[X, Y] = meshgrid(Xsurf, Ysurf);
Z = cumsum(reshape(error_filter(3,11:end,1),iter_max-10,1))./(1:iter_max-10)';
Z = [Z cumsum(reshape(error_filter(3,11:end,3),iter_max-10,1))./(1:iter_max-10)'];
Z = [Z cumsum(reshape(error_filter(3,11:end,5),iter_max-10,1))./(1:iter_max-10)'];
CO = [];
CO(:,:,1) = zeros(iter_max-10, 3); % red
CO(:,:,2) = zeros(iter_max-10, 3); % green
CO(:,:,3) = ones(iter_max-10, 3); % blue
s = surf(X,Y,Z, CO, 'FaceAlpha',0.2);
s.EdgeColor = 'none';

CO = [];
CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.8*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(1), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.25);
s.EdgeColor = 'none';

CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.6*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(3), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.15);
s.EdgeColor = 'none';

CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.4*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(5), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.05);
s.EdgeColor = 'none';

grid; hold on;xlabel('N');ylabel('step(\tau)', 'Rotation', 30); zlabel('Average Error for m_{0}(x_{3})');
axis_top = max([cum_error_meas(:,end) cum_error_stoc(:,end)],[],'all');
axis_low = min(cum_error_filter(:,end),[],'all');
axis([m_hist(1) m_hist(5) 100 iter_max axis_low-0.4*(axis_top-axis_low) axis_top+0.7*(axis_top-axis_low)]);

ax = gca;
ax.FontSize = 24;

set(gca, 'Ygrid', 'off')
set(gca, 'Xgrid', 'off')
set(gca, 'GridLineWidth', 2)
set(gca, 'GridAlpha', 0.6)
set(gcf, 'InvertHardcopy', 'off')

nexttile;
cum_error_stoc = cumsum(reshape([error_stoc(dim_size*(m_hist(1)),11:end,1);error_stoc(dim_size*(m_hist(3)),11:end,3);error_stoc(dim_size*(m_hist(5)),11:end,5)]',iter_max-10,3))'./[1:iter_max-10;1:iter_max-10;1:iter_max-10];
cum_error_meas = cumsum(reshape([error_meas(dim_size*(m_hist(1)),11:end,1);error_meas(dim_size*(m_hist(3)),11:end,3);error_meas(dim_size*(m_hist(5)),11:end,5)]',iter_max-10,3))'./[1:iter_max-10;1:iter_max-10;1:iter_max-10];
cum_error_filter = cumsum(reshape([error_filter(dim_size*(m_hist(1)),11:end,1);error_filter(dim_size*(m_hist(3)),11:end,3);error_filter(dim_size*(m_hist(5)),11:end,5)]',iter_max-10,3))'./[1:iter_max-10;1:iter_max-10;1:iter_max-10];

plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_error_stoc(1,:),'m-','Linewidth',3);
hold on
plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_error_meas(1,:),'k-','Linewidth',3);
plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_error_filter(1,:),'b-','Linewidth',3);
% 
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_error_stoc(2,:),'m-','Linewidth',3);
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_error_meas(2,:),'k-','Linewidth',3);
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_error_filter(2,:),'b-','Linewidth',3);
% 
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_error_stoc(3,:),'m-','Linewidth',3);
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_error_meas(3,:),'k-','Linewidth',3);
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_error_filter(3,:),'b-','Linewidth',3);
view([55 10])

Xsurf = [m_hist(1) m_hist(3) m_hist(5)];
Ysurf = 11:iter_max;
[X, Y] = meshgrid(Xsurf, Ysurf);
Z = cum_error_filter';
CO = [];
CO(:,:,1) = zeros(iter_max-10, 3); % red
CO(:,:,2) = zeros(iter_max-10, 3); % green
CO(:,:,3) = ones(iter_max-10, 3); % blue
s = surf(X,Y,Z, CO, 'FaceAlpha',0.2);
s.EdgeColor = 'none';

CO = [];
CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.8*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(1), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.25);
s.EdgeColor = 'none';

CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.6*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(3), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.15);
s.EdgeColor = 'none';

CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.4*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(5), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.05);
s.EdgeColor = 'none';

grid; hold on;xlabel('N');ylabel('step(\tau)', 'Rotation', 30); zlabel('Average Error for m_{N-1}(x_{3})');
axis_top = max([cum_error_meas(:,end) cum_error_stoc(:,end)],[],'all');
axis_low = min(cum_error_filter(:,end),[],'all');
axis([m_hist(1) m_hist(5) 100 iter_max axis_low-0.4*(axis_top-axis_low) axis_top+0.7*(axis_top-axis_low)]);
ax = gca;
ax.FontSize = 24;

set(gca, 'Ygrid', 'off')
set(gca, 'Xgrid', 'off')
set(gca, 'GridLineWidth', 2)
set(gca, 'GridAlpha', 0.6)
set(gcf, 'InvertHardcopy', 'off')
leg = legend('Model Estimation', 'Measurement Estimation', 'Filter Estimation', 'NumColumns', 3,'Location', 'north', 'FontSize', 24, 'TextColor', 'k');
leg.Layout.Tile = 'north';
set(gcf, 'Position',  [200, 250, 1150, 650])

FVx = zeros(1, n_ensembles);
MVx = zeros(1, n_ensembles);
XVx = zeros(1, n_ensembles);

cxxcum = cumsum(cx_KF(dim_size*(iter2-1)+1:dim_size*iter2,:,1:end),3);
for iter2=1:n_ensembles
    cxcum = cumsum(cx(dim_size*(iter2-1)+1:dim_size*iter2,:,1:end, end),3);
    cxxcum = cumsum(cx_KF(dim_size*(iter2-1)+1:dim_size*iter2,dim_size*(iter2-1)+1:dim_size*iter2,1:end),3);
    FVx(iter2) = trace(cxcum(:,:,end))/(iter_max);
    MVx(iter2) = trace(Axx(dim_size*(iter2-1)+1:dim_size*iter2,dim_size*(iter2-1)+1:dim_size*iter2)*cxcum(:,:,end-1)*Axx(dim_size*(iter2-1)+1:dim_size*iter2,dim_size*(iter2-1)+1:dim_size*iter2)'/(iter_max-1)+(WVx(iter2)^2+wvb^2)*eye(dim_size));
    XVx(iter2) = trace(cxxcum(:,:,end))/(iter_max);
end

%%
figure
tillay = tiledlayout(1,2, 'Padding', 'Compact');

cum_error_x_stoc = cumsum(reshape(error_x_stoc(11:end,[1 3 5]),iter_max-10,3))'./[1:iter_max-10;1:iter_max-10;1:iter_max-10];
cum_error_x_meas = cumsum(reshape(error_x_meas(11:end,[1 3 5]),iter_max-10,3))'./[1:iter_max-10;1:iter_max-10;1:iter_max-10];
cum_error_x_filter = cumsum(reshape(error_x_filter(11:end,[1 3 5]),iter_max-10,3))'./[1:iter_max-10;1:iter_max-10;1:iter_max-10];
cum_error_x_KF = cumsum(error_x_filter_KF(11:end))./(1:iter_max-10);
nexttile
hold on
plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_error_x_stoc(1,:),'m-','Linewidth',3);
hold on
plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_error_x_meas(1,:),'k-','Linewidth',3);
plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_error_x_filter(1,:),'b-','Linewidth',3);
plot3(m_hist(1)*ones(iter_max-10,1),plot_iters,cum_error_x_KF,'r:','Linewidth',5);
% 
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_error_x_stoc(2,:),'m-','Linewidth',3);
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_error_x_meas(2,:),'k-','Linewidth',3);
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_error_x_filter(2,:),'b-','Linewidth',3);
plot3(m_hist(3)*ones(iter_max-10,1),plot_iters,cum_error_x_KF,'r:','Linewidth',5);
% 
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_error_x_stoc(3,:),'m-','Linewidth',3);
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_error_x_meas(3,:),'k-','Linewidth',3);
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_error_x_filter(3,:),'b-','Linewidth',3);
plot3(m_hist(5)*ones(iter_max-10,1),plot_iters,cum_error_x_KF,'r:','Linewidth',5);

view([55 10])

Xsurf = [m_hist(1) m_hist(3) m_hist(5)];
Ysurf = 11:iter_max;
[X, Y] = meshgrid(Xsurf, Ysurf);
Z=cum_error_x_filter';
CO = [];
CO(:,:,1) = zeros(iter_max-10, 3); % red
CO(:,:,2) = zeros(iter_max-10, 3); % green
CO(:,:,3) = ones(iter_max-10, 3); % blue
s = surf(X,Y,Z, CO, 'FaceAlpha',0.2);
s.EdgeColor = 'none';

CO = [];
CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.8*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(1), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.25);
s.EdgeColor = 'none';

CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.6*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(3), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.15);
s.EdgeColor = 'none';

CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.4*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(5), 2, 2), [11 11; iter_max iter_max], [0 1;0 1], CO, 'FaceAlpha',0.05);
s.EdgeColor = 'none';

grid; hold on;xlabel('N');ylabel('step(\tau)'); zlabel('Average Error for ensemble states');
axis_top = max([cum_error_x_stoc;cum_error_x_meas],[],'all');
axis_low = min(cum_error_x_KF,[],'all');
axis([m_hist(1) m_hist(5) 100 iter_max axis_low-0.2*(axis_top-axis_low) axis_top+0.2*(axis_top-axis_low)]);
ax = gca;
ax.FontSize = 24;

set(gca, 'Ygrid', 'off')
set(gca, 'Xgrid', 'off')
set(gca, 'GridLineWidth', 2)
set(gca, 'GridAlpha', 0.6)
set(gcf, 'InvertHardcopy', 'off')

mean(error_x_meas(101:end,:))
mean(error_x_filter(101:end,:))
mean(error_x_meas_KF(101:end))
mean(error_x_stoc_KF(101:end))
mean(error_x_filter_KF(101:end))

FVx = zeros(5, n_ensembles);
MVx = zeros(5, n_ensembles);
XVx = zeros(5, n_ensembles);

iter_start = 1;
for iter2=1:n_ensembles
    cxxcum = cumsum(cx_KF(dim_size*(iter2-1)+1:dim_size*iter2,dim_size*(iter2-1)+1:dim_size*iter2,iter_start:end),3);
    for iter=1:5
        cxcum = cumsum(cx(dim_size*(iter2-1)+1:dim_size*iter2,:,iter_start:end, iter),3);
        FVx(iter, iter2) = trace(cxcum(:,:,end))/(iter_max+1-iter_start);
        MVx(iter, iter2) = trace(Axx(dim_size*(iter2-1)+1:dim_size*iter2,dim_size*(iter2-1)+1:dim_size*iter2)*cxcum(:,:,end-1)*Axx(dim_size*(iter2-1)+1:dim_size*iter2,dim_size*(iter2-1)+1:dim_size*iter2)'/(iter_max+1-iter_start)+(WVx(iter2)^2+wvb^2)*eye(dim_size));
        XVx(iter, iter2) = trace(cxxcum(:,:,end))/(iter_max+1-iter_start);
    end
end

x_traj_error = abs(x_traj_model - x_traj_dist);
x_traj_error = cumsum(x_traj_error')'./(1:iter_max+1);
nexttile
hold on
plot3(m_hist(1)*ones(n_ensembles,1),omega,MVx(1,:),'m-','Linewidth',3);
hold on
plot3(m_hist(1)*ones(n_ensembles,1),omega,dim_size*(VVx(1:n_ensembles).^2+vvb^2),'k-','Linewidth',3);
plot3(m_hist(1)*ones(n_ensembles,1),omega,FVx(1,:),'b-','Linewidth',3);
plot3(m_hist(1)*ones(n_ensembles,1),omega,XVx(1,:),'r:','Linewidth',5);
% 
plot3(m_hist(3)*ones(n_ensembles,1),omega,MVx(3,:),'m-','Linewidth',3);
plot3(m_hist(3)*ones(n_ensembles,1),omega,dim_size*(VVx(1:n_ensembles).^2+vvb^2),'k-','Linewidth',3);
plot3(m_hist(3)*ones(n_ensembles,1),omega,FVx(3,:),'b-','Linewidth',3);
plot3(m_hist(3)*ones(n_ensembles,1),omega,XVx(3,:),'r:','Linewidth',5);
% 
plot3(m_hist(5)*ones(n_ensembles,1),omega,MVx(5,:),'m-','Linewidth',3);
plot3(m_hist(5)*ones(n_ensembles,1),omega,dim_size*(VVx(1:n_ensembles).^2+vvb^2),'k-','Linewidth',3);
plot3(m_hist(5)*ones(n_ensembles,1),omega,FVx(5,:),'b-','Linewidth',3);
plot3(m_hist(5)*ones(n_ensembles,1),omega,XVx(5,:),'r:','Linewidth',5);

view([55 10])

Xsurf = [m_hist(1) m_hist(3) m_hist(5)];
Ysurf = omega;
[X, Y] = meshgrid(Xsurf, Ysurf);
Z = [FVx(1,:)' FVx(3,:)' FVx(5,:)'];
CO = [];
CO(:,:,1) = zeros(n_ensembles, 3); % red
CO(:,:,2) = zeros(n_ensembles, 3); % green
CO(:,:,3) = ones(n_ensembles, 3); % blue
s = surf(X,Y,Z, CO, 'FaceAlpha',0.2);
s.EdgeColor = 'none';

CO = [];
CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.8*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(1), 2, 2), [min(omega) min(omega); max(omega) max(omega)], [0 1;0 1], CO, 'FaceAlpha',0.25);
s.EdgeColor = 'none';

CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.6*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(3), 2, 2), [min(omega) min(omega); max(omega) max(omega)], [0 1;0 1], CO, 'FaceAlpha',0.15);
s.EdgeColor = 'none';

CO(:,:,1) = 0*ones(2, 2); % red
CO(:,:,2) = 0.4*ones(2, 2); % green
CO(:,:,3) = 0*ones(2, 2); % blue
s = surf(repmat(m_hist(5), 2, 2), [min(omega) min(omega); max(omega) max(omega)], [0 1;0 1], CO, 'FaceAlpha',0.05);
s.EdgeColor = 'none';

axis_top = max([dim_size*(VVx(1:n_ensembles).^2+vvb^2)';MVx],[],'all');
axis_low = min(XVx,[],'all');
grid; hold on;xlabel('N');ylabel('\beta', Interpreter='tex'); zlabel('Trace of error covariance (\tau = T)', Interpreter='tex');
axis([m_hist(1) m_hist(5) min(omega) max(omega) axis_low-0.2*(axis_top-axis_low) axis_top+0.2*(axis_top-axis_low)]);
ax = gca;
ax.FontSize = 24;

set(gca, 'Ygrid', 'off')
set(gca, 'Xgrid', 'off')
set(gca, 'GridLineWidth', 2)
set(gca, 'GridAlpha', 0.6)
set(gcf, 'InvertHardcopy', 'off')
leg = legend('Model Estimation', 'Measurement Estimation', 'Moment Filter Estimation', 'Classical Filter Estimation', 'NumColumns', 2,'Location', 'north', 'FontSize', 24, 'TextColor', 'k');
leg.Layout.Tile = 'north';
set(gcf, 'Position',  [200, 250, 950, 650])

%%
figure
mean_filter = mean(error_x_filter_KF(101:end));
plot(m_hist, mean(error_x_filter(101:end,:)), 'b-', m_hist, repmat(mean_filter, 1, size(m_hist,2)), 'r-', 'LineWidth', 3)
hold on
mean_stoc = mean(error_x_stoc(101:end,:));
mean_meas = repmat(mean(error_x_meas_KF(101:end)), 1, size(m_hist,2));
plot(m_hist, mean_stoc, 'm-', m_hist, mean_meas, 'k-', 'LineWidth', 3)
grid; hold;xlabel('Moment Order'); ylabel('Mean Error')
leg = legend('Moment KF', 'True KF', 'Model', 'Measurement','Location', 'southoutside', 'NumColumns', 4,'FontSize',24);
axis_top = max([mean_stoc mean_meas]);
axis([min(m_hist), max(m_hist), mean_filter-0.2*(axis_top-mean_filter) axis_top+0.2*(axis_top-mean_filter)])
ax = gca;
ax.FontSize = 24;
set(gcf, 'Position',  [200, 200, 770, 500])
%%

%sum(abs(Kxhist(1:dim_size:end-1,:)-Kxhist_KF(1:dim_size:end-1)))/n_ensembles
%sum(abs(Kxhist(2:dim_size:end, :)-Kxhist_KF(2:dim_size:end)))/n_ensembles
Performance = (1-(mean(error_x_filter(101:end,:))-mean(error_x_filter_KF(101:end)))./(mean(error_x_meas_KF(101:end))-mean(error_x_filter_KF(101:end))))';
Time = time_mom';
Moment_order = m_hist';

table(Moment_order, Time, Performance)
%% Discretization and Linearization scheme

function M=initial_moment(truncated_ind,X0,omega_list, dim_size, Leg_poly, prec_mom)
    % moment at time 0 (initial moment state)
    mk_record=zeros(dim_size*truncated_ind,1);
    if size(omega_list, 2)<prec_mom
        omega_list_interp = linspace(-1, 1, prec_mom);
        X0_interp = zeros(dim_size,prec_mom);
        for iter = 1:dim_size
            X0_interp(iter,:) = interp1(omega_list, X0(iter,:), omega_list_interp, 'nearest');
        end
        omega_list = omega_list_interp;
        X0 = X0_interp;
    end
    for k=1:truncated_ind
%         mk=trapz(omega_list,((omega_list.^(k)).*X0)');
        mk=trapz(omega_list,((Leg_poly(k,:)).*X0)');
        mk_record(dim_size*(k-1)+1:dim_size*k,1)=mk';
    end
    M=mk_record;
end

% Numerical integration with adaptive dt
function [t_store, x_trajJ] = adaptive_taylor(p,Phi,Psi_p,interval,x0)

xJ = x0;
x_trajJ = xJ;

err_tol = 10e-9;
dt_err_tol = 10e-6;

t = interval(1);
t_store = t;
dt_store = [];

while t < interval(2)-dt_err_tol && size(t_store, 2)<1e2
    T = num2cell(xJ);
    dt = min((err_tol*factorial(p)/norm(Psi_p(T{:}),inf))^(1/p),interval(2)-t);
    t = t+dt;
    t_store = [t_store t];
    dt_store = [dt_store, dt];
    
    xJ = Phi(dt,T{:});
    x_trajJ = [x_trajJ xJ];
    
end

t_store = transpose(t_store);
x_trajJ = transpose(x_trajJ);

end

function [t_store, x_trajJ] = adaptive_taylor_wng(p,Phi,Psi_p,interval,x0, wv, dim_size, wvb, G_beta_w)

xJ = x0;
x_trajJ = xJ;

err_tol = 10e-9;

t = interval(1);
t_store = t;
dt_store = [];

while t < interval(2)
    T = num2cell(xJ);
    dt = min((err_tol*factorial(p)/norm(Psi_p(T{:}),inf))^(1/p),interval(2)-t);
    t = t+dt;
    t_store = [t_store t];
    dt_store = [dt_store, dt];
    
    xJ = Phi(dt,T{:});
    x_trajJ = [x_trajJ xJ];
    
end

t_store = transpose(t_store);
x_trajJ = transpose(x_trajJ);

x_trajJ(end,1:dim_size*size(G_beta_w,2)) = x_trajJ(end,1:dim_size*size(G_beta_w,2)) + (normrnd(0, repmat(wv(1:size(G_beta_w, 2)),dim_size,1)))' + kron(G_beta_w', normrnd(0, wvb, dim_size, 1))';

end
