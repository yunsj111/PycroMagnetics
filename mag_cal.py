import pymag as ppm
import mag_visualize as pmv
import magmatlib as mml
import time

def start_calculation(evol=None, name=None, 
                      total_step=1000000,
                      log_step=1000,
                      early_stop=True,
                      conv_error_criteria=5e-5,
                      conv_error_tolerance=10
                      verbose=0):

    # Set Logger
    mlogger = ppm.MagnetLogger(log_name=name)

    # Set start time
    start = time.perf_counter()

    # Start calculation loop
    for i in range(total_step):

        # Calculate one step of evolution
        evol.evolve()

        # Get end time
        end = time.perf_counter()

        
        if i%log_step == 0:
            # Get log
            total_mx = evol.magnet.mx.sum()
            total_my = evol.magnet.my.sum()
            total_mz = evol.magnet.mz.sum()
            mask_area = evol.magnet.mask.sum()
            mlogger(time=evol.time, dm=evol.dm, 
                    mean_mx=total_mx/mask_area, 
                    mean_my=total_my/mask_area, 
                    mean_mz=total_mz/mask_area)
            
            # Print log
            if verbose > 0:
                print('{}/{} time:{}'.format(i,total_step, end - start))
                print('current dm & tstep : {:e} & {:e}'.format(evol.dm, evol.tstep))
                print('  mean (mx, my, mz) : ({:e}, {:e}, {:e})'.format(total_mx/mask_area, 
                                                                        total_my/mask_area,
                                                                        total_mz/mask_area))

            # Save evolver
            ppm.save_Evolver(evol, name)
            
            # Set early stop
            if early_stop:
                times, dms, mean_mxs, mean_mys, mean_mzs = mlogger.load_log()
                dms = np.array(dms)
                dms = np.log(dms)/np.log(10)
                if ppm.is_conversion(dms, 
                                     criteria=conv_error_criteria, 
                                     min_iter=conv_error_tolerance):
                    print('============== Early Stop ==============')
                    break