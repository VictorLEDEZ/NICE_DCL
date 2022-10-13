import time

import torch

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer

# ! I might have to change to decoupled training here (NICE_GAN Training)
if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    opt = TrainOptions().parse()   # get training options
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.

    # create a model given opt.model and other options
    model = create_model(opt)
    print('The number of training images = %d' % dataset_size)

    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1
    times = []
    # * outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0
        # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        visualizer.reset()

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # * inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                # regular setup: load and print networks; create schedulers
                model.setup(opt)
                model.parallelize()
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            # ? update here maybe (optimizing params for every epochs and elements in the dataset)
            model.optimize_parameters()  # * we call this function for every epochs and data
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / \
                batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:  # * Display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # * Print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # * Cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' %
                      (epoch, total_iters))
                # it's useful to occasionally show the experiment name on console
                print(opt.name)
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:  # * Cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch,
              opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        # update learning rates at the end of every epoch.
        model.update_learning_rate()
