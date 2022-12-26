# NICE_DCL: A Combination of NICE-GAN and DCLGAN

NICE-GAN + DCLGAN = NICEDCL

We believe that we could start with the DCLGAN as a basis and use the feature of the NICEGAN consisting of reusing the encoder.
The model is very similar to the DCLGAN model except that we reuse the discriminator’s encoder in the generator. Therefore, there is no more encoder in the generator but only in the discriminator. The architecture is then simplified.

## Structure

- Based on the code of DCLGAN ([repo](https://github.com/JunlinHan/DCLGAN/tree/2727288f252cda192b11da1c8c04aff2997eaa6a), [paper](https://arxiv.org/pdf/2104.07689.pdf))
- To see the implementation of the model, nice_dcl_model.py has been added in the models folder.
- In networks.py, the new generator and discriminator have been added. Both generators and discriminators are the one from NICE-GAN ([repo](https://github.com/alpc91/NICE-GAN-pytorch/tree/1b114f1de95be845629a94db667368528a503a7d), [paper](https://arxiv.org/pdf/2003.00273.pdf)).

## What has been done

- Implemented new Generators and Discriminators.
- Settled the decoupled training.
- Created a custom dataset for NIR to RGB translation (NIR2RGB).
- Adapted PatchNCE loss to fit with the new Generators and Discriminators.

## What has to be done

- Correct the Gradient backpropagation issue.
- Train and test the model in the NIR2RGB dataset.
- Compare it with the other models at the state-of-the-art on that same dataset.
