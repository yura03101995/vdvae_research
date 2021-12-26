import numpy as np
import imageio
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from apex.optimizers import FusedAdam as AdamW
from data import set_up_data, read_data
from utils import get_cpu_stats_over_ranks
from train_helpers import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema, ConstantMult
from matplotlib import pyplot as plt


def training_step(H, data_input, target, vae, ema_vae, optimizer, iterate):
    t0 = time.time()
    vae.zero_grad()
    stats = vae.forward(data_input, target)
    stats['elbo'].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item()
    distortion_nans = torch.isnan(stats['distortion']).sum()
    rate_nans = torch.isnan(stats['rate']).sum()
    stats.update(dict(rate_nans=0 if rate_nans == 0 else 1, distortion_nans=0 if distortion_nans == 0 else 1))
    stats = get_cpu_stats_over_ranks(stats)

    skipped_updates = 1
    # only update if no rank has a nan and if the grad norm is below a specific threshold
    if stats['distortion_nans'] == 0 and stats['rate_nans'] == 0 and (H.skip_threshold == -1 or grad_norm < H.skip_threshold):
        optimizer.step()
        skipped_updates = 0
        update_ema(vae, ema_vae, H.ema_rate)

    t1 = time.time()
    stats.update(skipped_updates=skipped_updates, iter_time=t1 - t0, grad_norm=grad_norm)
    return stats

def training_primary_step(H, data_input, target, vae, ema_vae, optimizer, iterate, lambda_mult, optimizer_lambda):
    t0 = time.time()
    vae.zero_grad()
    stats = vae.forward(data_input, target)

    copy_lambda_rate = lambda_mult.lambda_rate.detach()
    copy_distortion = stats["distortion"].detach()
    loss_nn = ((1 - copy_lambda_rate) * stats['rate'] + copy_lambda_rate * stats['distortion']).mean()
    loss_lambda = -1. * lambda_mult(copy_distortion - H.gamma)
    loss_nn.backward()
    loss_lambda.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item()
    distortion_nans = torch.isnan(stats['distortion']).sum()
    rate_nans = torch.isnan(stats['rate']).sum()
    stats.update(dict(rate_nans=0 if rate_nans == 0 else 1, distortion_nans=0 if distortion_nans == 0 else 1))
    stats = get_cpu_stats_over_ranks(stats)

    skipped_updates = 1
    # only update if no rank has a nan and if the grad norm is below a specific threshold
    if stats['distortion_nans'] == 0 and stats['rate_nans'] == 0 and (
            H.skip_threshold == -1 or grad_norm < H.skip_threshold):
        optimizer.step()
        optimizer_lambda.step()
        skipped_updates = 0
        update_ema(vae, ema_vae, H.ema_rate)

    lambda_mult.lambda_rate.data.clamp_(0.0001, 0.9999)

    t1 = time.time()
    lambda_rate = lambda_mult.lambda_rate.detach().item()
    stats.update(skipped_updates=skipped_updates, iter_time=t1 - t0, grad_norm=grad_norm,
                 lambda_rate=lambda_rate, d_gamma=(copy_distortion - H.gamma).item())
    return stats


def eval_step(data_input, target, ema_vae):
    with torch.no_grad():
        stats = ema_vae.forward(data_input, target)
        zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(data_input)]
        mb = data_input.shape[0]
        output_samples = ema_vae.forward_samples_set_latents(mb, zs)
        target_samples = ((target.detach().cpu().numpy() * 127.5) + 127.5).clip(0., 255.)
        stats["MSE"] = np.square(output_samples - target_samples).mean()
    stats = get_cpu_stats_over_ranks(stats)
    return stats


def get_sample_for_visualization(data, preprocess_fn, num, dataset):
    for x in DataLoader(data, batch_size=num):
        break
    orig_image = (x[0] * 255.0).to(torch.uint8).permute(0, 2, 3, 1) if dataset == 'ffhq_1024' else x[0]
    preprocessed = preprocess_fn(x)[0]
    return orig_image, preprocessed


def train_loop(H, data_train, data_valid, preprocess_fn, vae, ema_vae, logprint):
    optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_opt(H, vae, logprint)
    train_sampler = DistributedSampler(data_train, num_replicas=H.mpi_size, rank=H.rank)
    viz_batch_original, viz_batch_processed = get_sample_for_visualization(data_valid, preprocess_fn, H.num_images_visualize, H.dataset)
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()
    for epoch in range(starting_epoch, H.num_epochs):
        train_sampler.set_epoch(epoch)
        for x in DataLoader(data_train, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=train_sampler):
            data_input, target = preprocess_fn(x)
            training_stats = training_step(H, data_input, target, vae, ema_vae, optimizer, iterate)
            stats.append(training_stats)
            scheduler.step()
            if iterate % H.iters_per_print == 0 or iters_since_starting in early_evals:
                logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0], epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))

            if iterate % H.iters_per_images == 0 or (iters_since_starting in early_evals and H.dataset != 'ffhq_1024') and H.rank == 0:
                write_images(H, ema_vae, viz_batch_original, viz_batch_processed, f'{H.save_dir}/samples/{H.exp_name}_samples-{iterate}.png', logprint)

            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['elbo']):
                    logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
                    fp = os.path.join(H.save_dir, f'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, vae, ema_vae, optimizer, H)

            if iterate % H.iters_per_ckpt == 0 and H.rank == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae, ema_vae, optimizer, H)

        if epoch % H.epochs_per_eval == 0:
            valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
            logprint(model=H.desc, type='eval_loss', epoch=epoch, step=iterate, **valid_stats)


def train_primary_loop(H, data_train, data_valid, preprocess_fn, vae, ema_vae, logprint):
    optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_opt(H, vae, logprint)

    lambda_mult = ConstantMult().cuda()
    optimizer_lambda = AdamW(lambda_mult.parameters(), weight_decay=H.wd, lr=H.lr, betas=(0.0, 0.999))

    train_sampler = DistributedSampler(data_train, num_replicas=H.mpi_size, rank=H.rank)
    viz_batch_original, viz_batch_processed = get_sample_for_visualization(data_valid, preprocess_fn,
                                                                           H.num_images_visualize, H.dataset)
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()
    for epoch in range(starting_epoch, H.num_epochs):
        train_sampler.set_epoch(epoch)
        for x in DataLoader(data_train, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=train_sampler):
            data_input, target = preprocess_fn(x)
            training_stats = training_primary_step(H, data_input, target, vae, ema_vae, optimizer, iterate, lambda_mult,
                                                   optimizer_lambda)
            stats.append(training_stats)
            scheduler.step()
            if iterate % H.iters_per_print == 0 or iters_since_starting in early_evals:
                logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0], epoch=epoch, step=iterate,
                         **accumulate_stats(stats, H.iters_per_print))

            if iterate % H.iters_per_images == 0 or (
                    iters_since_starting in early_evals and H.dataset != 'ffhq_1024') and H.rank == 0:
                write_images(H, ema_vae, viz_batch_original, viz_batch_processed,
                             f'{H.save_dir}/samples/{H.exp_name}_samples-{iterate}.png', logprint)

            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['elbo']):
                    logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate,
                             **accumulate_stats(stats, H.iters_per_print))
                    fp = os.path.join(H.save_dir, f'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, vae, ema_vae, optimizer, H)

            if iterate % H.iters_per_ckpt == 0 and H.rank == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae, ema_vae, optimizer, H)

        if epoch % H.epochs_per_eval == 0:
            valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
            logprint(model=H.desc, type='eval_loss', epoch=epoch, step=iterate, **valid_stats)


TRAIN_LOOPS = {
    "simple": train_loop,
    "primary": train_primary_loop,
}


def evaluate(H, ema_vae, data_valid, preprocess_fn):
    stats_valid = []
    valid_sampler = DistributedSampler(data_valid, num_replicas=H.mpi_size, rank=H.rank)
    for x in DataLoader(data_valid, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=valid_sampler):
        data_input, target = preprocess_fn(x)
        stats_valid.append(eval_step(data_input, target, ema_vae))
    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(n_batches=len(vals), filtered_elbo=np.mean(finites), **{k: np.mean([a[k] for a in stats_valid]) for k in stats_valid[-1]})
    return stats


def write_images(H, ema_vae, viz_batch_original, viz_batch_processed, fname, logprint):
    zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed)]
    batches = [viz_batch_original.numpy()]
    mb = viz_batch_processed.shape[0]
    lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    for i in lv_points:
        batches.append(ema_vae.forward_samples_set_latents(mb, zs[:i], t=0.1))
    for t in [1.0, 0.9, 0.8, 0.7][:H.num_temperatures_visualize]:
        batches.append(ema_vae.forward_uncond_samples(mb, t=t))
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *viz_batch_processed.shape[1:])).transpose([0, 2, 1, 3, 4]).reshape([n_rows * viz_batch_processed.shape[1], mb * viz_batch_processed.shape[2], 3])
    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


def run_test_eval(H, ema_vae, data_test, preprocess_fn, logprint):
    print('evaluating')
    stats = evaluate(H, ema_vae, data_test, preprocess_fn)
    print('test results')
    for k in stats:
        print(k, stats[k])
    logprint(type='test_loss', **stats)


def run_draw_orig_samples(H):
    trX, vaX, teX, _, _ = read_data(H)
    data = np.concatenate([trX, vaX, teX])
    n_rows = int(np.ceil(np.sqrt(H.n_batch)))
    n_batch = n_rows * n_rows
    h = data.shape[1]
    w = data.shape[2]
    c = data.shape[3]
    rnd_inds = np.random.choice(np.arange(data.shape[0]), size=n_batch, replace=False)
    data = data[rnd_inds].reshape(n_rows, n_rows, h, w, c).transpose([0, 2, 1, 3, 4]).reshape(n_rows * h, n_rows * w, c)
    imageio.imwrite(f'{H.save_dir}/original_samples.png', data)


def run_draw_uncond_samples(H, ema_vae, logprint):
    n_rows = int(np.ceil(np.sqrt(H.n_batch)))
    n_batch = n_rows * n_rows
    samples, logscales, probs = ema_vae.forward_uncond_samples(n_batch, return_logscales_probs=True)
    logscales = logscales.ravel()
    probs = probs.ravel()
    assert probs.shape == logscales.shape

    plt.figure(figsize=(25, 10))
    plt.hist(logscales, weights=probs, bins=300, density=True)
    plt.savefig(f'{H.save_dir}/{H.exp_name}_logscales_distr.png', dpi="figure")

    h = samples.shape[1]
    w = samples.shape[2]
    c = samples.shape[3]
    im = samples.reshape((n_rows, n_rows, h, w, c)).transpose(0, 2, 1, 3, 4).reshape(n_rows * h, n_rows * w, c)
    imageio.imwrite(f'{H.save_dir}/{H.exp_name}_uncond_samples.png', im)


def run_draw_test_samples(H, ema_vae, data_valid, preprocess_fn):
    viz_batch_original, viz_batch_processed = get_sample_for_visualization(data_valid, preprocess_fn, H.num_images_visualize, H.dataset)
    zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed)]
    batches = [viz_batch_original.numpy()]
    mb = viz_batch_processed.shape[0]
    lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    for i in lv_points:
        batches.append(ema_vae.forward_samples_set_latents(mb, zs[:i], t=0.1))
    n_rows = len(batches)
    im = (
        np.concatenate(batches, axis=0)
            .reshape((n_rows, mb, *viz_batch_processed.shape[1:]))
            .transpose([0, 2, 1, 3, 4])
            .reshape([n_rows * viz_batch_processed.shape[1], mb * viz_batch_processed.shape[2], 3])
    )
    imageio.imwrite(f'{H.save_dir}/{H.exp_name}_test_samples.png', im)


def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    if H.draw_orig_samples:
        run_draw_orig_samples(H)
        
    vae, ema_vae = load_vaes(H, logprint)

    if H.draw_uncond_samples:
        run_draw_uncond_samples(H, ema_vae, logprint)

    if H.draw_test_samples:
        run_draw_test_samples(H, ema_vae, data_valid_or_test, preprocess_fn)

    if H.test_eval:
        run_test_eval(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)

    if H.train_model:
        TRAIN_LOOPS[H.vae_type](H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae, logprint)


if __name__ == "__main__":
    main()
