import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import matplotlib.pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'Times New Roman'

def Draw_plots(N_rnn=None, 
               Fs = [],
               rec_rnn=None, 
               rec_msn=None, 
               rec_GP=None, 
               rec_th=None, 
               rec_inh=None, 
               rec_E=None, 
               rec_mem=None, 
               rec_output=None, 
               means=None, 
               trial=None, 
               test=None, 
               data_GT=[],
               LEIA_Pred=[],
               combine_flag=None, 
               save_flag=None,
               show_heat_map = None, 
               f_name="None",
               y_range=None,
               fig=None,
               ax1=None,
               ax2=None,
               ax3=None,
               ax4=None,
               ax5=None,
               ax6=None,
               ax7=None,
               ax8=None,
               ax9=None):
    
    if combine_flag == None:
        if N_rnn!=None and rec_rnn!=None:
            plot_rnn = np.zeros((len(rec_rnn), N_rnn))
            for i in range(len(rec_rnn)):
                plot_rnn[i,:] = rec_rnn[i][0].reshape(1,N_rnn)
            x_time = np.arange(0,len(rec_rnn),1)
            fig, ax = plt.subplots(1,1,figsize=(12,5))
            axins = inset_axes(ax,
                                width="5%",  # width = 5% of parent_bbox width
                                height="100%",  # height : 50%
                                loc='lower left',
                                bbox_to_anchor=(1.01, 0., 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0)
            ax.set_xlabel('time (a.u.)', fontsize=30)
            ax.set_ylabel('Units', fontsize=30)
            ax.tick_params(axis='y', labelsize=25)
            ax.tick_params(axis='x', labelsize=25)
            ax.set_title('RNN dynamics', fontsize=30, pad=20)
            im = ax.imshow(plot_rnn.T, cmap='plasma', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            ax.autoscale(False)
            cbar = fig.colorbar(im, cax=axins)
            cbar.ax.tick_params(labelsize=20)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()

        # RNN - MSN weights
        if len(Fs) != 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            im = ax.imshow(Fs)
            ax.set_title('RNN - MSN weights', fontsize=25)
            ax.set_xlabel('Receiving MSN Neurons', fontsize=20)
            ax.set_ylabel('Sending RNN Neurons', fontsize=20)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = fig.colorbar(im, cax=cax)
            clb.set_label('weight values', labelpad=17,
                            fontsize=15, rotation=270)
            plt.tight_layout()
            plt.show()

                # we need to extract everyhting for this to be plotted
        if rec_E != None:
            plot_E = np.zeros((1, len(rec_E)))
            for i in range(len(rec_E)):
                plot_E[0, i] = rec_E[i]


            fig, ax = plt.subplots(1,1,figsize=(7, 5))
            # plt.figure(figsize=(7, 5))
            # ax.plot(plot_I[0, :], label='Inhibitory neuron', linewidth=3)
            ax.plot(plot_E[0, :], label='Thalamus entropy', linewidth=3)
            ax.set_xlabel('Time', fontsize=20)
            ax.set_ylabel('Activity', fontsize=20)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            # ax.set_xticks(fontsize=15)
            # ax.set_yticks(fontsize=15)
            # plt.ylim(0,30)
            ax.set_title('Thalamus Entropy activity', fontsize=22, y=1.05)
            ax.legend()
            plt.show()
        
        if rec_msn != None:
            # plot the msn layer
            fig, ax = plt.subplots(1,1,figsize=(7, 5))
            # plt.figure(figsize=(7, 5))
            colors = pl.cm.jet(np.linspace(0, 1, len(rec_msn)))
            for i in range(len(rec_msn)):
                ax.plot(rec_msn[i][0], color=colors[i], linewidth=0.5)
            ax.set_xlabel('action location', fontsize=20)
            ax.set_ylabel('Activation', fontsize=20)
            ax.tick_params(axis='x', labelsize=17)
            ax.tick_params(axis='y', labelsize=17)
            ax.set_title('Dynamics Striatum neurons', fontsize=24)
            plt.show()
            
            if show_heat_map != None:
                # imshow msn layer
                plot_th = np.zeros((len(rec_msn), 360))
                for i in range(len(rec_msn)):
                    plot_th[i,:] = rec_msn[i][0].reshape(1,360)
                x_time = np.arange(0,len(rec_msn),1)
                fig2, ax2 = plt.subplots(1,1,figsize=(12,5))
                axins = inset_axes(ax2,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax2.transAxes,
                                    borderpad=0)
                ax2.set_xlabel('time (a.u.)', fontsize=30)
                ax2.set_ylabel('Location', fontsize=30)
                ax2.tick_params(axis='y', labelsize=25)
                ax2.tick_params(axis='x', labelsize=25)
                ax2.set_title('MSN dynamics', fontsize=30, pad=20)
                im = ax2.imshow(plot_th.T, cmap='plasma', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                ax2.autoscale(False)
                cbar = fig2.colorbar(im, cax=axins)
                cbar.ax.tick_params(labelsize=20)
                fig2.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()
        
        if rec_mem != None:
            # plot the mem layer
            fig, ax = plt.subplots(1,1,figsize=(7, 5))
            # plt.figure(figsize=(7, 5))
            colors = pl.cm.jet(np.linspace(0, 1, len(rec_mem)))
            for i in range(len(rec_mem)):
                ax.plot(rec_mem[i][0], color=colors[i], linewidth=0.5)
            ax.set_xlabel('action location', fontsize=20)
            ax.set_ylabel('Activation', fontsize=20)
            ax.tick_params(axis='x', labelsize=17)
            ax.tick_params(axis='y', labelsize=17)
            ax.set_title('dynamics Integration layer', fontsize=24)
            plt.show()
            
            if show_heat_map != None:
                # imshow mem layer
                plot_th = np.zeros((len(rec_mem), 360))
                for i in range(len(rec_mem)):
                    plot_th[i,:] = rec_mem[i][0].reshape(1,360)
                x_time = np.arange(0,len(rec_mem),1)
                fig2, ax2 = plt.subplots(1,1,figsize=(12,5))
                axins = inset_axes(ax2,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax2.transAxes,
                                    borderpad=0)
                ax2.set_xlabel('time (a.u.)', fontsize=30)
                ax2.set_ylabel('Location', fontsize=30)
                ax2.tick_params(axis='y', labelsize=25)
                ax2.tick_params(axis='x', labelsize=25)
                ax2.set_title('Memory dynamics', fontsize=30, pad=20)
                im = ax2.imshow(plot_th.T, cmap='plasma', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                ax2.autoscale(False)
                cbar = fig2.colorbar(im, cax=axins)
                cbar.ax.tick_params(labelsize=20)
                fig2.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()
        
        if rec_th != None:
            # plot the thalamus layer
            fig, ax = plt.subplots(1,1,figsize=(7, 5))
            colors = pl.cm.jet(np.linspace(0, 1, len(rec_th)))
            for i in range(len(rec_th)):
                ax.plot(rec_th[i][0], color=colors[i], linewidth=0.5)
            ax.set_xlabel('action location', fontsize=20)
            ax.set_ylabel('Activation', fontsize=20)
            ax.tick_params(axis='x', labelsize=17)
            ax.tick_params(axis='y', labelsize=17)
            ax.set_title('dynamics thalamus layer', fontsize=24)
            plt.show()
            
            if show_heat_map != None:
                # imshow thalamus layer
                plot_th = np.zeros((len(rec_th), 360))
                for i in range(len(rec_th)):
                    plot_th[i,:] = rec_th[i][0].reshape(1,360)
                x_time = np.arange(0,len(rec_th),1)
                fig2, ax2 = plt.subplots(1,1,figsize=(12,5))
                axins = inset_axes(ax2,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax2.transAxes,
                                    borderpad=0)
                ax2.set_xlabel('time (a.u.)', fontsize=30)
                ax2.set_ylabel('Location', fontsize=30)
                ax2.tick_params(axis='y', labelsize=25)
                ax2.tick_params(axis='x', labelsize=25)
                ax2.set_title('thalamus dynamics', fontsize=30, pad=20)
                im = ax2.imshow(plot_th.T, cmap='plasma', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                ax2.autoscale(False)
                cbar = fig2.colorbar(im, cax=axins)
                cbar.ax.tick_params(labelsize=20)
                fig2.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()
        
        if rec_GP != None:
            # plot the GP layer
            fig, ax = plt.subplots(1,1,figsize=(7, 5))
            # plt.figure(figsize=(7, 5))
            colors = pl.cm.jet(np.linspace(0, 1, len(rec_GP)))
            for i in range(len(rec_GP)):
                ax.plot(rec_GP[i][0], color=colors[i], linewidth=0.5)
            ax.set_xlabel('action location', fontsize=20)
            ax.set_ylabel('Activation', fontsize=20)
            ax.tick_params(axis='x', labelsize=17)
            ax.tick_params(axis='y', labelsize=17)
            # ax.set_xticks(fontsize=17)
            # ax.set_yticks(fontsize=17)
            ax.set_title('dynamics GP layer', fontsize=24)
            plt.show()
            
            if show_heat_map != None:
                # imshow GP layer
                plot_th = np.zeros((len(rec_GP), 360))
                for i in range(len(rec_GP)):
                    plot_th[i,:] = rec_GP[i][0].reshape(1,360)
                x_time = np.arange(0,len(rec_GP),1)
                fig2, ax2 = plt.subplots(1,1,figsize=(12,5))
                axins = inset_axes(ax2,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax2.transAxes,
                                    borderpad=0)
                ax2.set_xlabel('time (a.u.)', fontsize=30)
                ax2.set_ylabel('Location', fontsize=30)
                ax2.tick_params(axis='y', labelsize=25)
                ax2.tick_params(axis='x', labelsize=25)
                ax2.set_title('GP dynamics', fontsize=30, pad=20)
                im = ax2.imshow(plot_th.T, cmap='plasma', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                ax2.autoscale(False)
                cbar = fig2.colorbar(im, cax=axins)
                cbar.ax.tick_params(labelsize=20)
                fig2.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()
        
        if len(rec_output) != None and means != None and test != None and trial !=None:   
            # plot the mem layer
            fig, ax = plt.subplots(1,1,figsize=(7, 5))
            # plt.figure(figsize=(7, 5))
            colors = pl.cm.jet(np.linspace(0, 1, len(rec_output)))
            for i in range(len(rec_output)):
                plt.plot(rec_output[i][0], color=colors[i], linewidth=0.5)
            # red is the real mean value
            ax.axvline(means[trial], linestyle='dashed',
                        color='r', linewidth=5)
            ax.axvline(test, linestyle='dashed', color='blue',
                        linewidth=5)   # blue is the prediction position
            ax.set_xlabel('action location', fontsize=20)
            ax.set_ylabel('Activation', fontsize=20)
            ax.tick_params(axis='x', labelsize=17)
            ax.tick_params(axis='y', labelsize=17)
            ax.set_title('dynamics motor layer', fontsize=24)
            plt.show()
            
            if show_heat_map != None:
                # imshow output layer
                plot_th = np.zeros((len(rec_output), 360))
                for i in range(len(rec_output)):
                    plot_th[i,:] = rec_output[i][0].reshape(1,360)
                x_time = np.arange(0,len(rec_output),1)
                fig2, ax2 = plt.subplots(1,1,figsize=(12,5))
                axins = inset_axes(ax2,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax2.transAxes,
                                    borderpad=0)
                ax2.set_xlabel('time (a.u.)', fontsize=30)
                ax2.set_ylabel('Location', fontsize=30)
                ax2.tick_params(axis='y', labelsize=25)
                ax2.tick_params(axis='x', labelsize=25)
                ax2.set_title('Output dynamics', fontsize=30, pad=20)
                im = ax2.imshow(plot_th.T, cmap='plasma', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                ax2.autoscale(False)
                cbar = fig2.colorbar(im, cax=axins)
                cbar.ax.tick_params(labelsize=20)
                fig2.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()

    else:


        #first plot RNN_dynamics
        if N_rnn!=None and rec_rnn!=None and ax1!=None and fig!=None:
            plot_rnn = np.zeros((len(rec_rnn), N_rnn))
            for i in range(len(rec_rnn)):
                plot_rnn[i,:] = rec_rnn[i][0].reshape(1,N_rnn)
            if trial == 17:
                axins = inset_axes(ax1,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax1.transAxes,
                                    borderpad=0)
            # ax1.set_xlabel('time (a.u.)', fontsize=12)
            if trial == 4:
                # ax1.set_ylabel('Units', fontsize=45)
                ax1.tick_params(axis='y', labelsize=60)
                ax1.set_yticks([0, 60, 119])
            # ax1.tick_params(axis='x', labelsize=15)
                # ax1.set_title('RNN Dynamics', fontsize=55, pad=15)
            else: ax1.set_yticks([])
            im = ax1.imshow(plot_rnn.T[:120], cmap='plasma', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            ax1.autoscale(False)
            if trial == 17:
                cbar = fig.colorbar(im, cax=axins)
                cbar.ax.tick_params(labelsize=55)
                cbar.set_ticks([0, 0.5, 1])
            ax1.set_xticks([])

        
        # second plot
        if rec_msn != None and ax2!=None and fig!=None:
            # imshow msn layer
            plot_th = np.zeros((len(rec_msn), 360))
            for i in range(len(rec_msn)):
                plot_th[i,:] = rec_msn[i][0].reshape(1,360)
            x_time = np.arange(0,len(rec_msn),1)
            # fig2, ax2 = plt.subplots(1,1,figsize=(12,5))
            if trial == 17:
                axins = inset_axes(ax2,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax2.transAxes,
                                    borderpad=0)
            # ax2.set_xlabel('time (a.u.)', fontsize=12)
            # if trial == 4:
            #     ax2.set_ylabel('Location', fontsize=35)
            if trial == 4:
                ax2.tick_params(axis='y', labelsize=60)
                ax2.set_yticks([0, 180, 359])
                # ax2.tick_params(axis='x', labelsize=15)
                # ax2.set_title('Striatum layer Dynamics', fontsize=55, pad=10)
            else: ax2.set_yticks([])
            im = ax2.imshow(plot_th.T, cmap='jet', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            ax2.autoscale(False)
            if trial == 17:
                cbar = fig.colorbar(im, cax=axins)
                cbar.ax.tick_params(labelsize=55)
                cbar.set_ticks([0, 0.5, 1])
            ax2.set_xticks([])


        # Third plot
        if rec_GP != None and ax3!=None and fig!=None:
            # imshow GP layer
            plot_th = np.zeros((len(rec_GP), 360))
            for i in range(len(rec_GP)):
                plot_th[i,:] = rec_GP[i][0].reshape(1,360)
            # fig2, ax2 = plt.subplots(1,1,figsize=(12,5))
            if trial == 17:
                axins = inset_axes(ax3,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax3.transAxes,
                                    borderpad=0)
            # ax3.set_xlabel('time (a.u.)', fontsize=12)
            # if trial == 4:
            #     ax3.set_ylabel('Location', fontsize=45)
            if trial == 4:
                ax3.tick_params(axis='y', labelsize=60)
                ax3.set_yticks([0, 180, 359])
                # ax3.tick_params(axis='x', labelsize=15)
                # ax3.set_title('GP layer Dynamics', fontsize=55, pad=10)
            else: ax3.set_yticks([])
            im = ax3.imshow(plot_th.T, cmap='jet', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            ax3.autoscale(False)
            if trial == 17:
                cbar = fig.colorbar(im, cax=axins)
                cbar.ax.tick_params(labelsize=55)
                cbar.set_ticks([0, 0.5, 1])
            ax3.set_xticks([])

            

        # Fourth plot
        # imshow thalamus layer
        if rec_th != None and ax4!=None and fig!=None:
            plot_th = np.zeros((len(rec_th), 360))
            for i in range(len(rec_th)):
                plot_th[i,:] = rec_th[i][0].reshape(1,360)
            # fig2, ax2 = plt.subplots(1,1,figsize=(12,5))
            if trial == 17:
                axins = inset_axes(ax4,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax4.transAxes,
                                    borderpad=0)
            # ax4.set_xlabel('time (a.u.)', fontsize=12)
            if trial == 4:
                # ax4.set_ylabel('Location', fontsize=35)
                ax4.set_yticks([0, 180, 359])
                ax4.tick_params(axis='y', labelsize=60)
                # ax4.tick_params(axis='x', labelsize=25)
                # ax4.set_title('Thalamus layer Dynamics', fontsize=55, pad=10)
            else: ax4.set_yticks([])
            im = ax4.imshow(plot_th.T, cmap='jet', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            ax4.autoscale(False)
            if trial == 17:
                cbar = fig.colorbar(im, cax=axins)
                cbar.ax.tick_params(labelsize=55)
                cbar.set_ticks([0, 0.5, 1])
            ax4.set_xticks([])

        # Fifth plot 
        if rec_output!=None and ax5!=None and fig!=None:
            plot_th = np.zeros((len(rec_output), 360))
            for i in range(len(rec_output)):
                plot_th[i,:] = rec_output[i][0].reshape(1,360)
            if trial == 17:
                axins = inset_axes(ax5,
                                    width="5%",  # width = 5% of parent_bbox width
                                    height="100%",  # height : 50%
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax5.transAxes,
                                    borderpad=0)
            if trial == 4:
                ax5.set_yticks([0, 180, 359])
                # ax5.set_xlabel('time (a.u.)', fontsize=45)
                # ax5.set_ylabel('Location', fontsize=35)
                ax5.tick_params(axis='y', labelsize=60)
                # ax5.tick_params(axis='x', labelsize=25)
                # ax5.set_title('Motor layer Dynamics', fontsize=55, pad=10)
            else: ax5.set_yticks([])
            ax5.tick_params(axis='x', labelsize=60)
            ax5.set_xticks([0, 100, 300, 500])
            im = ax5.imshow(plot_th.T, cmap='jet', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            ax5.autoscale(False)
            if trial == 17:
                cbar = fig.colorbar(im, cax=axins)
                # cbar.ax.tick_params(labelsize=45)
                # cbar.set_ticks([0, 0.5, 1])

        
        if save_flag == None:
            pass
        else:
            if fig != None:
                plt.subplots_adjust(hspace=0.4, wspace=0.1)  # 0.4 0.5
                fig.savefig('{}.svg'.format(f_name), format = 'svg',dpi=300)
