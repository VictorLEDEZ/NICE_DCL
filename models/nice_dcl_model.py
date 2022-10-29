import itertools
from pickle import FALSE
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from util.image_pool import ImagePool


class NICEDCLModel(BaseModel):
    """ This class implements DCLGAN model.
    This code is inspired by CUT and CycleGAN.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for DCLGAN """
        parser.add_argument('--DCL_mode', type=str,
                            default="DCL", choices='DCL')
        parser.add_argument('--lambda_GAN', type=float,
                            default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=2.0,
                            help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_IDT', type=float, default=1.0,
                            help='weight for l1 identical loss: (G(X),X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='4,8,12,16',
                            help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float,
                            default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int,
                            default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization.")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for DCLGAN.
        if opt.DCL_mode.lower() == "dcl":
            parser.set_defaults(nce_idt=True, lambda_NCE=2.0)
        else:
            raise ValueError(opt.DCL_mode)

        return parser

    def __init__(self, opt):  # ? Maybe change something here
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['disA', 'gen2B',
                           'NCE1', 'disB', 'gen2A', 'NCE2', 'G']
        visual_names_A = ['real_A', 'fake_A2B']
        visual_names_B = ['real_B', 'fake_B2A']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['idt_B', 'idt_A']
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        # combine visualizations for A and B
        self.visual_names = visual_names_A + visual_names_B

        if self.isTrain:
            self.model_names = ['gen2B', 'netF1',
                                'disA', 'gen2A', 'netF2', 'disB']
        else:  # during test time, only load G
            self.model_names = ['gen2B', 'gen2A']

        # ! ####################################################################
        # ! define networks (both generator and discriminator) #################
        self.gen2B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                       opt.no_antialias_up, self.gpu_ids, opt)
        self.gen2A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                       opt.no_antialias_up, self.gpu_ids, opt)
        self.netF1 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)
        self.netF2 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)
        self.disA = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                      self.gpu_ids, opt)
        self.disB = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                      self.gpu_ids, opt)
        # ! ####################################################################

        if self.isTrain:
            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)
            # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionSim = torch.nn.L1Loss('sum').to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.gen2B.parameters(), self.gen2A.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_G_loss().backward()  # calculate gradients for G
            self.backward_D_A()  # calculate gradients for disA
            self.backward_D_B()  # calculate gradients for disB
            self.optimizer_F = torch.optim.Adam(itertools.chain(
                self.netF1.parameters(), self.netF2.parameters()))
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):  # ? maybe change something here
        # * we call this function for every epochs and data
        # forward
        self.forward()

        # * update D (same structure in DCLGAN) ################################
        self.set_requires_grad(
            [self.disA, self.disB], True)  # * not frozen
        self.optimizer_D.zero_grad()
        self.backward_D_A()  # calculate gradients for disA
        self.backward_D_B()  # calculate graidents for disB
        self.optimizer_D.step()

        # * update G ###########################################################
        self.set_requires_grad([self.disA, self.disB], False)  # * frozen
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):  # ? here => done v1
        # * Here we don't compute fake_B2A2B and fake_A2B2A as there is no Cycle-Consistency Loss
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_A2B = self.gen2B(self.real_A)  # gen2B(A)
        # self.fake_B2A = self.gen2A(self.real_B)  # gen2A(B)

        _, self.real_A_z = self.disA(
            input=self.real_A.detach(), discriminating=FALSE)
        _, self.real_B_z = self.disB(
            input=self.real_B.detach(), discriminating=FALSE)

        self.fake_A2B = self.gen2B(self.real_A_z)
        self.fake_B2A = self.gen2A(self.real_B_z)

        if self.opt.nce_idt:
            # self.idt_A = self.gen2B(self.real_B)
            # self.idt_B = self.gen2A(self.real_A)

            _, self.real_B_z = self.disA(
                input=self.real_B.detach(), discriminating=FALSE)
            _, self.real_A_z = self.disB(
                input=self.real_A.detach(), discriminating=FALSE)

            # TODO this should be of size [1, 3, 256, 256] but we got [1, 3, 24, 24]
            self.idt_A = self.gen2B(self.real_B_z)
            self.idt_B = self.gen2A(self.real_A_z)

    def backward_D_basic(self, netD, real, fake):  # ? here
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real, _ = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake, _ = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator disA"""
        fake_A2B = self.fake_B_pool.query(self.fake_A2B)
        self.loss_disA = self.backward_D_basic(
            self.disA, self.real_B, fake_A2B) * self.opt.lambda_GAN

    def backward_D_B(self):
        """Calculate GAN loss for discriminator disB"""
        fake_B2A = self.fake_A_pool.query(self.fake_B2A)
        self.loss_disB = self.backward_D_basic(
            self.disB, self.real_A, fake_B2A) * self.opt.lambda_GAN

    def compute_G_loss(self):  # ? here => done
        """Calculate GAN and NCE loss for the generator"""
        fake_A2B = self.fake_A2B
        fake_B2A = self.fake_B2A

        # ! GAN LOSS ###########################################################
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fakeB, _ = self.disA(fake_A2B)
            pred_fakeA, _ = self.disB(fake_B2A)
            self.loss_gen2B = self.criterionGAN(
                pred_fakeB, True).mean() * self.opt.lambda_GAN
            self.loss_gen2A = self.criterionGAN(
                pred_fakeA, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_gen2B = 0.0
            self.loss_gen2A = 0.0

        # ! NCE LOSS ###########################################################
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE1 = self.calculate_NCE_loss1(
                self.real_A, self.fake_A2B) * self.opt.lambda_NCE
            self.loss_NCE2 = self.calculate_NCE_loss2(
                self.real_B, self.fake_B2A) * self.opt.lambda_NCE
        else:
            self.loss_NCE1, self.loss_NCE_bd, self.loss_NCE2 = 0.0, 0.0, 0.0
        if self.opt.lambda_NCE > 0.0:

            # ! L1 IDENTICAL LOSS
            # TODO #############################################################
            # TODO RuntimeError: The size of tensor a (24) must match the size of tensor b (256) at non-singleton dimension 3

            # * self.idt_A (This one is the problem) ---------------------------
            # * torch.Size([1, 3, 24, 24])

            # * self.real_B ----------------------------------------------------
            # * torch.Size([1, 3, 256, 256])
            # * 256 is the size of the real image

            self.loss_idt_A = self.criterionIdt(
                self.idt_A, self.real_B) * self.opt.lambda_IDT
            # TODO #############################################################
            self.loss_idt_B = self.criterionIdt(
                self.idt_B, self.real_A) * self.opt.lambda_IDT
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * \
                0.5 + (self.loss_idt_A + self.loss_idt_B) * 0.5

        else:
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5

        # ! FULL OBJECTIVE #####################################################
        self.loss_G = (self.loss_gen2B + self.loss_gen2A) * 0.5 + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss1(self, src, tgt):  # ? here => done
        n_layers = len(self.nce_layers)
        _, tgt_z = self.disB(input=tgt.detach(), discriminating=FALSE)
        _, src_z = self.disA(input=src.detach(), discriminating=FALSE)
        feat_q = self.gen2A(tgt_z, self.nce_layers, encode_only=True)
        feat_k = self.gen2B(src_z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF1(
            feat_k, self.opt.num_patches, None)
        # TODO #################################################################
        # TODO RuntimeError: CUDA error: device-side assert triggered. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING = 1.
        feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, sample_ids)
        # TODO #################################################################
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_NCE_loss2(self, src, tgt):  # ? here => done
        n_layers = len(self.nce_layers)
        _, tgt_z = self.disA(input=tgt.detach(), discriminating=FALSE)
        _, src_z = self.disB(input=src.detach(), discriminating=FALSE)
        feat_q = self.gen2B(tgt_z, self.nce_layers, encode_only=True)
        feat_k = self.gen2A(src_z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF2(
            feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF1(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.gen2B
            D = self.disA
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                _, source_z = D(input=source.detach(), discriminating=FALSE)
                visuals["fake_A2B"] = G(source_z)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals
