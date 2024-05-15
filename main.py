import torch
import argparse
import matplotlib.pyplot as plt
from exercise import VoxelMorph
from loaders import get_train_eval_test
from exercise import train, test
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default = 'train', choices =["train", "eval"])
parser.add_argument("--show_output", type=bool, default = False)
parser.add_argument("--save_output_path", type=str, default = 'deformed.pt')
parser.add_argument("--inp_channels", type=int, default = 2)
parser.add_argument("--out_channels", type=int, default = 2)
parser.add_argument("--nb_features", type=list_of_ints, default = [32,64,128])
parser.add_argument("--nb_conv_per_level", type=int, default = 2)
parser.add_argument("--scale_factor", type=int, default = 2)
parser.add_argument("--activation", type=str, default = 'ReLU', choices = ["ReLU", "ELU", "LeakyReLU"])
parser.add_argument("--last_act_function", type=str, default = 'ReLU', choices = ["softmax", "sigmoid", "ReLU", "no_act_function"])
parser.add_argument("--unpooling", type=str, default = 'transp_conv', choices =["transp_conv", "interpolation"])
parser.add_argument("--mode", type=str, default = 'nearest', choices=['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'])
parser.add_argument("--checkpoint_path", type=str, default = 'models/ckpt.ckpt')
parser.add_argument("--max_epochs", type=int, default = 300)
parser.add_argument("--accumulate_grad_batches", type=int, default = 1)
parser.add_argument("--patience", type=int, default = 50)
parser.add_argument("--lr", type=float, default = 0.02)
parser.add_argument("--lam", type=float, default = 0.01)
parser.add_argument("--min_delta", type=float, default = 0.0001)
parser.add_argument("--devices", type=str, default = 1)
parser.add_argument("--accelerator", type=str, default = 'auto', choices = ["auto", "gpu", "cpu"])
args = parser.parse_args()

trainset, evalset, testset = get_train_eval_test()

model = VoxelMorph(inp_channels = args.inp_channels,
            out_channels = args.out_channels,
            nb_features = args.nb_features,
            nb_conv_per_level= args.nb_conv_per_level,
            scale_factor = args.scale_factor,
            activation = args.activation,
            last_act_function = args.last_act_function,
            unpooling = args.unpooling, 
            mode = args.mode)
checkpoint_path = args.checkpoint_path
max_epochs = args.max_epochs
accumulate_grad_batches = args.accumulate_grad_batches
patience = args.patience
lr = args.lr
lam = args.lam
min_delta = args.min_delta
save_output_path = args.save_output_path
accelerator = args.accelerator
devices = args.devices

if args.run == 'train':
    train(model, trainset, evalset, testset, lr,lam,accelerator, devices, checkpoint_path,  max_epochs, accumulate_grad_batches,min_delta,patience)
elif args.run == 'eval':
    loss, deformed = test( model,testset, checkpoint_path,lr,lam, accelerator, devices, max_epochs, accumulate_grad_batches,min_delta, patience)
    print(f"Eval Loss:{loss}")
    torch.save(deformed, args.save_output_path)
    if args.show_output:
        x = next(iter(testset))
        for idx in range(x.shape[0]):
            fig, ax = plt.subplots(1,3, figsize=(8,8))
            ax[0].imshow(x[idx,1].detach())
            ax[1].imshow(x[idx,0].detach())
            ax[2].imshow(deformed[idx,0].detach())
            ax[0].set_title('Mov')
            ax[1].set_title('Fix')
            ax[2].set_title('Deformed')
            plt.show()

else:
    print('run should be "train", "pocket_train" or "eval"')
