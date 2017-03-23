%
% Name :
%   ray_tracker_2d.m
%
% Purpose :
%   Identify the primary propagation mode between a HF transmitter and receiver. 
%   Calculate a ray trace and power profile along the path between the two stations.
%   This script is designed to be called from the python ray.py script.
%
%   Developed from example files provided with the pharlap ray tracing toolkit.


% Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variables are set by the calling python script. These variables are here
% for example/documentation purposes only.
%
%   UT              = [2017 2 2 21 53];     % UT - year, month, day, hour, minute
%   R12             = 28;                   % R12 index
%   speed_of_light  = 2.99792458e8;
%
%   ray_bearing     =  112.;
%   tx_lat          =   44.425;             % latitude of the start point of ray
%   tx_lon          = -121.238;             % longitude of the start point of ray
%   doppler_flag    = 1;                    % generate ionosphere 5 minutes later so that
%                                           % Doppler shift can be calculated
%   irregs_flag     = 0;                    % no irregularities - not interested in 
%                                           % Doppler spread or field aligned irregularities
%   kp              = 0;                    % kp not used as irregs_flag = 0. Set it to a 
%                                           % dummy value 
%
%   max_range       = 10000;                % maximum range for sampling the ionosphere (km)
%   num_range       = 201;                  % number of ranges (must be < 2000)
%
%   start_height    = 0;                    % start height for ionospheric grid (km)
%   height_inc      = 3;                    % height increment (km)
%   num_heights     = 200;                  % number of  heights (must be < 2000)
%
%   freq            = 14.042;               % frequency (MHz)
%   tol             = 1e-7;                 % ODE tolerance
%   nhops           = 3;
%   elevs           = [2:2:60];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

range_inc = max_range ./ (num_range - 1);   % range cell size (km)

% obtain ground range and azimuth of receiver from transmitter
[rx_range, rx_azm]  = latlon2raz(rx_lat, rx_lon, tx_lat, tx_lon);
rx_range            = rx_range / 1000.0;    % range now in km
ray_bearing         = rx_azm;               % assume omni-directional antnenna => no coning 

tic
% generate ionospheric, geomagnetic and irregularity grids
[iono_pf_grid, iono_pf_grid_5, collision_freq, irreg] = ...
    gen_iono_grid_2d(tx_lat, tx_lon, R12, UT, ray_bearing, ...
                     max_range, num_range, range_inc, start_height, ...
		     height_inc, num_heights, kp, doppler_flag, 'iri2016');
toc
 
% convert plasma frequency grid to  electron density in electrons/cm^3
iono_en_grid        = iono_pf_grid.^2   / 80.6164e-6;
iono_en_grid_5      = iono_pf_grid_5.^2 / 80.6164e-6;

% call raytrace for a fan of rays
% first call to raytrace so pass in the ionospheric and geomagnetic grids 
num_elevs   = length(elevs);
freqs       = freq.*ones(size(elevs));
[ray_data, ray_path_data] = ...
    raytrace_2d(tx_lat, tx_lon, elevs, ray_bearing, freqs, nhops, ...
             tol, irregs_flag, iono_en_grid, iono_en_grid_5, ...
	     collision_freq, start_height, height_inc, range_inc, irreg);

% Extract data from ray_data so it can be extracted by python. %%%%%%%%%%%%%%%%%
% Determine number of separate ray segments
rd_points   = 0;
for ii = 1:num_elevs
    rd_points = rd_points + length(ray_data(ii).lat);
end

% Create index vector of ray segments.
rd_id       = zeros(1,rd_points);
s_inx       = 1;
for ii = 1:num_elevs
    n       = length(ray_data(ii).lat);
    e_inx   = s_inx + n - 1;
    rd_id(s_inx:e_inx) = ii;
    s_inx   = e_inx + 1;
end

% Declare new non-stucture vectors for each field name.
ray_data_fieldnames = fieldnames(ray_data);
fns = ray_data_fieldnames;
for ii = 1:length(fns)
    expr = strcat('rd_',fns(ii),'= zeros(1,',num2str(rd_points),');');
    eval(expr{1});
end

% Populate the newly declared vectors.
s_inx = 1;
for ii = 1:num_elevs
    n_inx   = length(ray_data(ii).lat);
    e_inx   = s_inx+n_inx-1;
    
    for nn = 1:length(fns)
        if ~strcmp(fns(nn),'FAI_backscatter_loss') % We are not currently concerned with FAI's...

            lhs = strcat('rd_',fns(nn),'(',num2str(s_inx),':',num2str(e_inx),')');
            rhs = strcat('ray_data(',num2str(ii),').',fns(nn));
            expr = strcat(lhs{1},'=',rhs{1},';');
            eval(expr);
          % rd_lat(s_inx:e_inx) = ray_data(ii).lat            
        end
    end
    s_inx   = e_inx + 1;
end

%%
%% power calculations
%%
rd_D_Oabsorp    = zeros(1,length(rd_effective_range)) * nan;
rd_D_Xabsorp    = zeros(1,length(rd_effective_range)) * nan;
rd_fs_loss      = zeros(1,length(rd_effective_range)) * nan;

mm              = 0;
for ii = 1:length(ray_data)
    gnd_fs_loss     = 0;
    O_absorp        = 0;
    X_absorp        = 0;
    for kk = 1:length(ray_data(ii).ray_label)
        mm = mm + 1;
        if ray_data(ii).ray_label(kk) < 1
            continue;
        end
        ray_apogee          = rd_apogee(mm);
        ray_apogee_gndr     = rd_gnd_rng_to_apogee(mm);
        [ray_apogee_lat, ray_apogee_lon] = raz2latlon( ...
          ray_apogee_gndr, ray_bearing, tx_lat, tx_lon, 'wgs84');
        plasfrq_at_apogee   = rd_plasma_freq_at_apogee(mm);
        if kk == 1
            % calculate geo-mag splitting factor - assume that it
            % is the same for all hops (really need to calculate
            % separately for each hop)
            [del_fo, del_fx] = ...
                  gm_freq_offset(ray_apogee_lat, ray_apogee_lon, ...
                                 ray_apogee, ray_bearing, ...
                                 freq, plasfrq_at_apogee, UT);
        end
        elev        = rd_initial_elev(mm);
        O_absorp    = O_absorp + abso_bg(ray_apogee_lat,ray_apogee_lon,elev,freq + del_fo,UT,R12,1);
        X_absorp    = X_absorp + abso_bg(ray_apogee_lat,ray_apogee_lon,elev,freq + del_fx,UT,R12,0);

        rd_D_Oabsorp(mm)    = O_absorp;
        rd_D_Xabsorp(mm)    = X_absorp;

        if kk > 1
            fs_lat = rd_lat(mm-1);
            fs_lon = rd_lon(mm-1);
            % Forward ground-scattering loss.
            gnd_fs_loss = gnd_fs_loss + ground_fs_loss(fs_lat, fs_lon, elev, freq);
        end
        rd_fs_loss(mm)  = gnd_fs_loss;
    end
end

% one-way radar equation
wavelen     = speed_of_light ./ (freq .* 1e6);
pwr_tmp     = tx_power * (wavelen.^2 ./ (4.*pi)) ./ (4.*pi .* rd_effective_range.^2);

% ionospheric absorption terms for O and X modes
rd_rx_power_0_dB   = 10*log10(pwr_tmp) + gain_tx_db + gain_rx_db;
rd_rx_power_dB     = rd_rx_power_0_dB - rd_deviative_absorption - rd_fs_loss;
rd_rx_power_O_dB   = rd_rx_power_dB   - rd_D_Oabsorp;
rd_rx_power_X_dB   = rd_rx_power_dB   - rd_D_Xabsorp;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identify Ray Hitting Receiver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

srch_frequency          = [];
srch_elevation          = [];
srch_group_range        = [];
srch_deviative_absorption     = [];
srch_D_Oabsorp          = [];
srch_D_Xabsorp          = [];
srch_fs_loss            = [];
srch_eff_range          = [];
srch_phase_path         = [];
srch_ray_apogee         = [];
srch_ray_apogee_gndr    = [];
srch_plasfrq_at_apogee  = [];
srch_ray_hops           = [];
srch_del_freq_O         = [];
srch_del_freq_X         = [];
%srch_dgnd_dels          = [];

% loop over ray elevation 0.5 degree steps
srch_gnd_range   = zeros(num_elevs, nhops)*NaN;
srch_grp_range   = zeros(num_elevs, nhops)*NaN;
srch_labels      = zeros(num_elevs, nhops);

for el_idx=1:num_elevs
    for hop_idx = 1:ray_data(el_idx).nhops_attempted
        srch_gnd_range(el_idx, hop_idx) = ray_data(el_idx).ground_range(hop_idx);
        srch_grp_range(el_idx, hop_idx) = ray_data(el_idx).group_range(hop_idx);
        srch_labels(el_idx, hop_idx)    = ray_data(el_idx).ray_label(hop_idx);
    end
end

% find the "good rays" i.e. the rays which come to ground OK
srch_ray_good   = 0;
for hop_idx = 1:nhops
    srch_idx_goodray = find(srch_labels(:, hop_idx) == 1);
    if length(srch_idx_goodray) > 3
      
      % Find ray ground ranges which bracket the receiver ground range, do
      % raytracing with finer (0.05 deg) elevation grid within coarse 
      % braketing rays, and finally interpolate to find the ray elevations 
      % and group ranges (and their absorption losses), which will hit
      % the receiver.
      srch_els          = elevs(srch_idx_goodray);
      srch_gnd          = srch_gnd_range(srch_idx_goodray, hop_idx)';
      srch_grp          = srch_grp_range(srch_idx_goodray, hop_idx)';
      srch_dgrp_dels    = deriv(srch_grp, srch_els);
      srch_num          = length(srch_els);
      srch_grp_to_rx    = [];
      
      % loop over all good elevations
      for ii=1:srch_num-1
        
        % find the bracketing rays - ignore those whose rate of change of 
        % range  with elevation is too large as this indicates we are too
        % far into a cusp region to be reliable
        if ((srch_gnd(ii) >= rx_range & srch_gnd(ii+1) < rx_range) | ...
            (srch_gnd(ii) <= rx_range & srch_gnd(ii+1) > rx_range)) & ...
            (srch_els(ii+1) - srch_els(ii) < 2*elev_step) & ...
            (abs(srch_dgrp_dels(ii)) < 500) & (abs(srch_dgrp_dels(ii+1)) < 500)
          
          srch_el_step  = srch_els(ii+1) - srch_els(ii);
          fine_el_step  = srch_el_step ./ 5;
          fine_els      = [srch_els(ii) : fine_el_step : srch_els(ii+1)];
          fine_elevs    = [];
          fine_gnd      = [];
          fine_label    = [];
          
          % raytrace at fine elevation steps between bracketing rays
          freqs         = freq .* ones(size(fine_els));
          [fine_ray_data, fine_ray_path_data] = raytrace_2d(tx_lat, tx_lon, ...
                    fine_els, ray_bearing, freqs, hop_idx, tol, irregs_flag);
            
          for idx=1:6
            fine_elev = fine_els(idx);
            if fine_ray_data(idx).nhops_attempted == hop_idx
              fine_gnd      = [fine_gnd fine_ray_data(idx).ground_range(hop_idx)];
              fine_label    = [fine_label fine_ray_data(idx).ray_label(hop_idx)];
              fine_elevs    = [fine_elevs fine_elev];
            end
          end
          
          % interpolate to get elevation to launch ray to hit rx and
          % raytrace at this elevation to get all the other required
          % quantities 
          if (isempty(find(fine_label < 1)) & length(fine_label >=3))
            srch_elev_torx = interp1(fine_gnd, fine_elevs, rx_range, 'pchip');
            
            [srch_ray_data, srch_ray_path_data] = raytrace_2d(tx_lat, tx_lon, ...
                  srch_elev_torx, ray_bearing, freq, hop_idx, tol, irregs_flag);
            
            if srch_ray_data.ray_label == 1
              srch_elevation        = [srch_elevation srch_elev_torx];
              srch_group_range      = [srch_group_range srch_ray_data.group_range(hop_idx)];
              srch_phase_path       = [srch_phase_path srch_ray_data.phase_path(hop_idx)];
              srch_deviative_absorption   = ...
                  [srch_deviative_absorption srch_ray_data.deviative_absorption(hop_idx)];
              srch_eff_range = ...
                  [srch_eff_range srch_ray_data.effective_range(hop_idx)];
            
              srch_gnd_fs_loss      = 0;
              srch_O_absorp         = 0;
              srch_X_absorp         = 0;
              for kk = 1:hop_idx
                srch_ray_apogee  = srch_ray_data.apogee(kk);
                srch_ray_apogee_gndr  = srch_ray_data.gnd_rng_to_apogee(kk);
                [srch_ray_apogee_lat, srch_ray_apogee_lon] = raz2latlon( ...
                    srch_ray_apogee_gndr, ray_bearing, tx_lat, tx_lon, 'wgs84');
                srch_plasfrq_at_apogee = srch_ray_data.plasma_freq_at_apogee(kk);
                if kk == 1
                  % calculate geo-mag splitting factor - assume that it
                  % is the same for all hops (really need to calculate
                  % separately for each hop)
                  [srch_del_fo, srch_del_fx] = ...
                        gm_freq_offset(srch_ray_apogee_lat, srch_ray_apogee_lon, ...
                                       srch_ray_apogee, ray_bearing, ...
                                       freq, srch_plasfrq_at_apogee, UT);
                end
                srch_O_absorp = srch_O_absorp + abso_bg(srch_ray_apogee_lat, ...
                    srch_ray_apogee_lon, srch_elev_torx, freq + srch_del_fo, ...
                    UT, R12, 1);
                srch_X_absorp = srch_X_absorp + abso_bg(srch_ray_apogee_lat, ...
                    srch_ray_apogee_lon, srch_elev_torx, freq + srch_del_fx, ...
                    UT, R12, 0);

                if kk > 1
                  srch_fs_lat       = srch_ray_data.lat(kk-1);
                  srch_fs_lon       = srch_ray_data.lon(kk-1);
                  srch_gnd_fs_loss  = srch_gnd_fs_loss + ...
                    ground_fs_loss(srch_fs_lat, srch_fs_lon, srch_elev_torx, freq);
                end
              end
            
              srch_D_Oabsorp    = [srch_D_Oabsorp srch_O_absorp];
              srch_D_Xabsorp    = [srch_D_Xabsorp srch_X_absorp];
              srch_fs_loss      = [srch_fs_loss srch_gnd_fs_loss];
              srch_del_freq_O   = [srch_del_freq_O, srch_del_fo];
              srch_del_freq_X   = [srch_del_freq_X, srch_del_fx];
            
              srch_frequency    = [srch_frequency freq];
              srch_ray_hops     = [srch_ray_hops hop_idx];
              srch_ray_good     = 1;
            end  % of if label == 1
          end
        end   % of find bracketing rays
      end   % of loop over "good" elevations
    end   % of "if length(srch_idx_goodray) > 3"
end   % of loop over nhops
  
% Extract data from ray_data so it can be extracted by python. %%%%%%%%%%%%%%%%%
if srch_ray_good ~= 0
    % Determine number of separate ray segments
    srch_rd_points  = 0;
    srch_num_elevs  = length(srch_ray_data);
    for ii = 1:srch_num_elevs
        srch_rd_points = srch_rd_points + length(srch_ray_data(ii).lat);
    end

    % Create index vector of ray segments.
    srch_rd_id       = zeros(1,srch_rd_points);
    s_inx       = 1;
    for ii = 1:srch_num_elevs
        n       = length(srch_ray_data(ii).lat);
        e_inx   = s_inx + n - 1;
        srch_rd_id(s_inx:e_inx) = ii;
        s_inx   = e_inx + 1;
    end

    % Declare new non-stucture vectors for each field name.
    srch_ray_data_fieldnames = fieldnames(srch_ray_data);
    fns = srch_ray_data_fieldnames;
    for ii = 1:length(fns)
        expr = strcat('srch_rd_',fns(ii),'= zeros(1,',num2str(srch_rd_points),');');
        eval(expr{1});
    end

    % Populate the newly declared vectors.
    s_inx = 1;
    for ii = 1:srch_num_elevs
        n_inx   = length(srch_ray_data(ii).lat);
        e_inx   = s_inx+n_inx-1;
        
        for nn = 1:length(fns)
            if ~strcmp(fns(nn),'FAI_backscatter_loss') % We are not currently concerned with FAI's...

                lhs = strcat('srch_rd_',fns(nn),'(',num2str(s_inx),':',num2str(e_inx),')');
                rhs = strcat('srch_ray_data(',num2str(ii),').',fns(nn));
                expr = strcat(lhs{1},'=',rhs{1},';');
                eval(expr);
              % srch_rd_lat(s_inx:e_inx) = srch_ray_data(ii).lat            
            end
        end
        s_inx   = e_inx + 1;
    end

    %% one-way radar equation
    %pwr_tmp     = tx_power * (wavelen.^2 ./ (4.*pi)) ./ (4.*pi .* srch_eff_range.^2);
    %
    %%% ionospheric absorption terms for O and X modes
    %srch_rd_rx_power_0_dB  = 10*log10(pwr_tmp) + gain_tx_db + gain_rx_db;
    %srch_rd_rx_power_dB    = srch_rd_rx_power_0_dB - srch_deviative_absorption - srch_fs_loss;
    %srch_rd_rx_power_O_dB  = srch_rd_rx_power_dB   - srch_D_Oabsorp;
    %srch_rd_rx_power_X_dB  = srch_rd_rx_power_dB   - srch_D_Xabsorp;

    %%
    %% power calculations
    %%
    srch_rd_D_Oabsorp    = zeros(1,length(srch_rd_effective_range)) * nan;
    srch_rd_D_Xabsorp    = zeros(1,length(srch_rd_effective_range)) * nan;
    srch_rd_fs_loss      = zeros(1,length(srch_rd_effective_range)) * nan;

    mm              = 0;
    for ii = 1:length(srch_ray_data)
        gnd_fs_loss     = 0;
        O_absorp        = 0;
        X_absorp        = 0;
        for kk = 1:length(srch_ray_data(ii).ray_label)
            mm = mm + 1;
            if srch_ray_data(ii).ray_label(kk) < 1
                continue;
            end
            ray_apogee          = srch_rd_apogee(mm);
            ray_apogee_gndr     = srch_rd_gnd_rng_to_apogee(mm);
            [ray_apogee_lat, ray_apogee_lon] = raz2latlon( ...
              ray_apogee_gndr, ray_bearing, tx_lat, tx_lon, 'wgs84');
            plasfrq_at_apogee   = srch_rd_plasma_freq_at_apogee(mm);
            if kk == 1
                % calculate geo-mag splitting factor - assume that it
                % is the same for all hops (really need to calculate
                % separately for each hop)
                [del_fo, del_fx] = ...
                      gm_freq_offset(ray_apogee_lat, ray_apogee_lon, ...
                                     ray_apogee, ray_bearing, ...
                                     freq, plasfrq_at_apogee, UT);
            end
            elev        = srch_rd_initial_elev(mm);
            O_absorp    = O_absorp + abso_bg(ray_apogee_lat,ray_apogee_lon,elev,freq + del_fo,UT,R12,1);
            X_absorp    = X_absorp + abso_bg(ray_apogee_lat,ray_apogee_lon,elev,freq + del_fx,UT,R12,0);

            srch_rd_D_Oabsorp(mm)    = O_absorp;
            srch_rd_D_Xabsorp(mm)    = X_absorp;

            if kk > 1
                fs_lat = srch_rd_lat(mm-1);
                fs_lon = srch_rd_lon(mm-1);
                % Forward ground-scattering loss.
                gnd_fs_loss = gnd_fs_loss + ground_fs_loss(fs_lat, fs_lon, elev, freq);
            end
            srch_rd_fs_loss(mm)  = gnd_fs_loss;
        end
    end

    % one-way radar equation
    wavelen     = speed_of_light ./ (freq .* 1e6);
    pwr_tmp     = tx_power * (wavelen.^2 ./ (4.*pi)) ./ (4.*pi .* srch_rd_effective_range.^2);

    % ionospheric absorption terms for O and X modes
    srch_rd_rx_power_0_dB   = 10*log10(pwr_tmp) + gain_tx_db + gain_rx_db;
    srch_rd_rx_power_dB     = srch_rd_rx_power_0_dB - srch_rd_deviative_absorption - srch_rd_fs_loss;
    srch_rd_rx_power_O_dB   = srch_rd_rx_power_dB   - srch_rd_D_Oabsorp;
    srch_rd_rx_power_X_dB   = srch_rd_rx_power_dB   - srch_rd_D_Xabsorp;
end
