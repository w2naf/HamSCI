if __name__ == '__main__':
    import datetime
    import os
    import sys
    import pickle
    import copy 

    sTime       = datetime.datetime(2014,2,1)
    eTime       = datetime.datetime(2014,2,28)

    sTime       = datetime.datetime(2016,8,27)
    eTime       = datetime.datetime(2016,8,28)
    sTime       = datetime.datetime(2016,11,11)
    eTime       = datetime.datetime(2016,11,18)

#    sTime       = datetime.datetime(2016,10,1,0)
#    eTime       = datetime.datetime(2016,10,31,23,59, 59)

    sTime       = datetime.datetime(2016,11,1,0)
    sTime       = datetime.datetime(2016,10,1,0)
    eTime       = datetime.datetime(2016,11,2,0)
#    eTime       = datetime.datetime(2016,11,17,0)
#    eTime       = datetime.datetime(2016,11,30,0)
    eTime       = datetime.datetime(2016,12,1,0)

#    sTime       = datetime.datetime(2016,12,1,0)
##    eTime       = datetime.datetime(2016,12,5,0)
#    eTime       = datetime.datetime(2016,12,6,0)
    data_dir    = 'data/wspr' 

    #Select only stations within two lat/lon areas (near VT and NJIT)
    #   K2MFF 'FN20vr' (40.7429,-74.1770)
    #   KM4EGE 'EM97tf' 

    #For simplicity in this proof-of-concept application, only chose stations in the following gridsquares:
    #   FN20 and FN21 (or FN30 and FN31)
    #   Need to select from wider area for southern station 
    gridsq=['FN20', 'EM97']
    gridsq=['FN20', 'EM98']
    

    if str(sys.argv[1]) == 'archive': 
        df=store_data(sTime, eTime, gridsq)
        gridsq=['FN20', 'EM97']
        df=store_data(sTime, eTime, gridsq, df=df)
        import ipdb; ipdb.set_trace()


    #Test Code for VM
    print str(sys.argv[1])
    if str(sys.argv[1]) == 'usePickle': 
        import ipdb; ipdb.set_trace()
        p_dir='data/wspr/filtered_wspr'
        p_filename = 'wspr_'+gridsq[0]+'-'+gridsq[1]+'_'+sTime.strftime('%Y%m%d-')+eTime.strftime('%Y%m%d.p')
        p_filepath = os.path.join(p_dir,p_filename)
        print p_filepath
        with open(p_filepath,'rb') as fl:
            df_filt = pickle.load(fl)
        import ipdb; ipdb.set_trace()

    elif str(sys.argv[1]) == 'useFile': 
        p_dir='data/wspr/filtered_wspr'
        p_filename = 'wspr_'+gridsq[0]+'-'+gridsq[1]+'_'+sTime.strftime('%Y%m%d-')+eTime.strftime('%Y%m%d.csv')
        p_filepath = os.path.join(p_dir,p_filename)
        print p_filepath
        df_filt=pd.read_csv(p_filepath)
        import ipdb; ipdb.set_trace()
        df_filt['timestamp']=df_filt.timestamp.astype(datetime.datetime)

    #For two month long data runs
    elif str(sys.argv[1]) == 'useCSV': 
        p_dir='output/wspr/'
        p_filename = 'filtered_wspr_data_'+sTime.strftime('%Y%m')+'_'+gridsq[1]+'.csv'
        midTime=datetime.datetime(sTime.year, sTime.month+1, sTime.day)
        p_filename2 = 'filtered_wspr_data_'+midTime.strftime('%Y%m')+'_'+gridsq[1]+'.csv'
        p_filepath = os.path.join(p_dir,p_filename)
        p_filepath2 = os.path.join(p_dir,p_filename2)
        print p_filepath
        print p_filepath2
        df_filt=pd.read_csv(p_filepath)
        df_filt = pd.concat([df_filt,pd.read_csv(p_filepath2)])
        import ipdb; ipdb.set_trace()
        df_filt=wspr_lib.redefine_grid(df_filt, precision=4)
        df_filt=wspr_lib.find_hour(df_filt)
        df_filt['timestamp']=df_filt.timestamp.astype(datetime.datetime)

    #Original Code
    elif str(sys.argv[1]) == 'original':
#        import ipdb; ipdb.set_trace()
        #    df = wspr_lib.read_wspr(sTime,eTime,data_dir, overwrite=True)
        df = wspr_lib.read_wspr(sTime,eTime,data_dir)
#        import ipdb; ipdb.set_trace()
       
    #    #Find the pairs of stations with most links between them
    #    tx, num_tx, rx= find_pair(df, prefix='FN20', prefix2='EM95')

    #    #Found stations were KK4WJF and K3EA
    #    stations = ['KK4WJF', 'K3EA']
    #    stations = [tx[0], rx[0][0]]
    #    #Filter to only include links between two specified stations
    #    df_filt  =   wspr_lib.select_pair(df, stations)
        
        #Filter to only include links between stations in specific grid sqares
#        df_filt=wspr_lib.filter_grid_pair(df, gridsq, redef=True, precision=4) 
        df_filt=wspr_lib.filter_grid_pair(df, gridsq)
        df_filt=wspr_lib.redefine_grid(df_filt, precision=4)
#        #Redefine grid and filter by gridsquare
#        df=wspr_lib.redefine_grid(df, precision=4)
#        df_filt=wspr_lib.filter_grid_pair(df, gridsq, redef=False, precision=4)


    #    fig=plot_wspr_snr(df[df.power==30])
    df_filt = wspr_lib.dB_to_Watt(df_filt)

    #Plot figure 
#    fig=plot_wspr_snr(df_filt, by_pwr=False, legend=False)
    fig=plot_wspr_snr(df_filt, by_pwr=False, legend=True)
#    fig=plot_wspr_snr(df_filt, by_pwr=False, legend=False, raw_time=True)

    df_avg = snr_avg(df_filt)
    fig2=plot_avg_snr(df_avg, by_pwr=False, legend=False)

#  Need to write code to plot raw times 

    output_file= 'wspr_test'+'_'+'plot'+'_'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%Y%m%d_')+eTime.strftime('%Y%m%d')+'.png'
    output_file2= 'wspr_test'+'_'+'avg'+'_'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%Y%m%d_')+eTime.strftime('%Y%m%d')+'.png'

##    note = str(sys.argv[2])
#    if note:
#        output_file= 'wspr_test'+'_'+note+'_'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%Y%m%d_')+eTime.strftime('%Y%m%d')+'.png'
#    else:
#        output_file= 'wspr_test'+'_'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%Y%m%d_')+eTime.strftime('%Y%m%d')+'.png'
    output_dir=os.path.join('output', 'wspr')
    output_path=os.path.join(output_dir, output_file)
    output_path2=os.path.join(output_dir, output_file2)
#    output_path=os.path.join(output_dir, 'wspr_test'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%d%b%Y%H%MUT-')+eTime.strftime('%d%b%Y%H%MUT')+'.png')
    if not os.path.exists(output_path):
         try:    # Create the output directory, but fail silently if it already exists
             os.makedirs(output_dir) 
         except:
             pass

    fig.savefig(output_path)
    fig2.savefig(output_path2)
    import ipdb; ipdb.set_trace()

#   #Plot second 
#    df_filt=wspr_lib.filter_grid_pair(df, ['FN20', 'EM96']) 
#    gridsq=['FN20', 'EM96']
#    fig=plot_wspr_snr(df_filt)
#    output_dir=os.path.join('output', 'wspr')
#    output_path=os.path.join(output_dir, 'wspr_test'+gridsq[0]+'_'+gridsq[1]+'.png')
#    fig.savefig(output_path)
    import ipdb; ipdb.set_trace()
