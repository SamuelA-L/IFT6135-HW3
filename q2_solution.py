import torch
from q2_sampler import svhn_sampler
from q2_model import Critic, Generator
from torch import optim
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np


def lp_reg(x, y, critic, device):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    t = torch.rand(x.size()).to(device)
    x_hat = t*x + (1-t)*y
    # x_hat.requires_grad = True
    f_out = critic(x_hat)
    grad_f = torch.autograd.grad(f_out, x_hat, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(f_out))[0]
    grad_norm = torch.norm(grad_f, p=2, dim=-1)
    max_0_grad_squared = torch.nn.functional.relu(grad_norm-1) ** 2

    return torch.mean(max_0_grad_squared)

    # t = torch.rand(x.size(), device=device)
    #
    # x_hat = t*x + (1 - t)*y
    # f_x = critic(x_hat)
    # grad = torch.autograd.grad(f_x, x_hat,
    #                            retain_graph=True,
    #                            create_graph=True,
    #                            grad_outputs=torch.ones_like(f_x))[0]
    #
    # grad_norm = torch.norm(grad, p=2, dim=-1)
    # lp = torch.nn.functional.relu(grad_norm - 1) **2
    # return lp.mean()


def vf_wasserstein_distance(p, q, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    return critic(p).mean() - critic(q).mean()



if __name__ == '__main__':
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 50000 # N training iterations
    n_critic_updates = 5 # N critic updates per generator update
    lp_coeff = 10 # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100
    #
    # train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)
    #
    # generator = Generator(z_dim=z_dim).to(device)
    # critic = Critic().to(device)
    #
    # optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    # optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    #
    # # COMPLETE TRAINING PROCEDURE
    # train_iter = iter(train_loader)
    # valid_iter = iter(valid_loader)
    # test_iter = iter(test_loader)
    # x = torch.randn(train_batch_size, z_dim).to(device)
    #
    # for i in tqdm(range(n_iter)):
    #     generator.train()
    #     critic.train()
    #     for _ in range(n_critic_updates):
    #         try:
    #             data = next(train_iter)[0].to(device)
    #         except Exception:
    #             train_iter = iter(train_loader)
    #             data = next(train_iter)[0].to(device)
    #         #####
    #         # train the critic model here
    #         #####
    #         x = torch.randn(train_batch_size, z_dim).to(device)
    #         gen_x = generator(x)
    #         optim_critic.zero_grad()
    #         loss_critic = -vf_wasserstein_distance(data, gen_x, critic) + lp_coeff * lp_reg(data, gen_x, critic, device)
    #         loss_critic.backward()
    #         optim_critic.step()
    #
    #     #####
    #     # train the generator model here
    #     #####
    #     gen_x = generator(x)
    #     optim_generator.zero_grad()
    #     loss_gen = -torch.mean(critic(gen_x))
    #     loss_gen.backward()
    #     optim_generator.step()
    #
    #
    #     # Save sample images
    #     if i % 100 == 0:
    #         z = torch.randn(64, z_dim, device=device)
    #         imgs = generator(z)
    #         save_image(imgs, f'imgs_{i}.png', normalize=True, value_range=(-1, 1))


    # COMPLETE QUALITATIVE EVALUATION
    # torch.save(generator, 'generator.pt')
    # torch.save(critic, 'critic.pt')

    generator = torch.load('generator.pt')

    z = torch.randn(train_batch_size, z_dim).to(device)
    reference = generator(z)
    save_image(reference, 'img_perturbation/ref.png', normalize=True, value_range=(-1, 1))
    eps = [5, 15, 100]
    for i in range(z.size(1)):
        for e in eps:
            clone = z.clone()
            clone[:, i] += e
            image = generator(clone)
            save_image(image, f'img_perturbation/img_{i}_{e}.png', normalize=True, value_range=(-1, 1))


    x_1 = torch.randn(train_batch_size, z_dim).to(device)
    x_2 = torch.randn(train_batch_size, z_dim).to(device)
    img_1 = generator(x_1)
    img_2 = generator(x_2)
    save_image(img_1, f'latent_space_int/img_ref1.png', normalize=True, value_range=(-1, 1))
    save_image(img_2, f'latent_space_int/img_ref2.png', normalize=True, value_range=(-1, 1))
    save_image(img_1, f'pixel_space_int/img_ref1.png', normalize=True, value_range=(-1, 1))
    save_image(img_2, f'pixel_space_int/img_ref2.png', normalize=True, value_range=(-1, 1))

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for a in alphas:
        x_latent = generator(a*x_1 + (1-a)*x_2)
        save_image(x_latent, f'latent_space_int/img_latent_{a}.png', normalize=True, value_range=(-1, 1))
        x_pixel = a*img_1 + (1-a)*img_2
        save_image(x_pixel, f'pixel_space_int/img_pixel_{a}.png', normalize=True, value_range=(-1, 1))


