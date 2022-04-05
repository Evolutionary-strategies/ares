import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ['TORCH_HOME']=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'model/')
from torchvision.utils import save_image
import torch
import torchvision
import torchvision.transforms as transforms
from third_party.example.cifar10.pytorch_cifar10.models import *
from pytorch_ares.attack_torch import *
ATTACKS = {
    'ba': BoundaryAttack,
    'spsa': SPSA,   
    'nes': NES,
    'nattack':Nattack,
    'evolutionary':Evolutionary,
    'fgsm':FGSM,
    'autoattack':AutoAttack,
    'cw':CW,
    'deepfool':DeepFool,
    'pgd':PGD,
    'bim':BIM,
    'mim':MIM,
}

class Model(nn.Module):
    def __init__(self, requires_grad = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4608, 128)#???
        self.fc2 = nn.Linear(128, 10)
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class cifar10_model(torch.nn.Module):
    def __init__(self,device,model):
        torch.nn.Module.__init__(self)
        self.device=device
        self.model = model
        self.model = self.model.to(self.device)

    def forward(self, x):
        self.mean_torch_c = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(self.device)
        self.std_torch_c = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(self.device)
        x = (x - self.mean_torch_c) / self.std_torch_c
        labels = self.model(x.to(self.device))
        return labels


def test(args):
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    #test/evo/nets/gd_model_max.pt
    #test/evo/nets/nes_model_sigma01.pt
    #test/evo/nets/nes_model_sigma015.pt
    #test/evo/nets/es_model_sigma015_acc073_1.pt

    path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'test/evo/nets/nes_model_sigma01.pt')
   
    model = Model()
    pretrain_dict = torch.load(path, map_location=device)
    model.load_state_dict(pretrain_dict)
    net = cifar10_model(device, model)

    transform = transforms.ToTensor()
    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)
    
    net.eval()
    distortion = 0
    dist= 0
    success_num = 0
    test_num= 0
    
    if args.attack_name == 'mim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, norm=args.norm, decay_factor=args.decay_factor ,stepsize=args.stepsize, data_name="cifar10", steps=args.steps,target=False, device=device, loss=args.loss)
    if args.attack_name == 'bim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, data_name="cifar10", steps=args.steps,target=False, device=device, loss=args.loss)
    if args.attack_name == 'pgd':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, data_name="cifar10", steps=args.steps,target=False, device=device, loss=args.loss)
    if args.attack_name == 'deepfool':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, args.overshoot, args.max_iter, args.norm, False, device)
    if args.attack_name == 'cw':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, device,args.norm, args.target, args.kappa, args.lr, args.init_const, args.max_iter, args.binary_search_steps, "cifar10")
    if args.attack_name == 'autoattack':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net,norm=args.norm,steps=args.steps, query=args.n_queries, eps=args.eps, version=args.version,device=device)
    if args.attack_name == 'fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, eps=args.eps, data_name="cifar10",target=args.target, loss=args.loss, device=device)
    if args.attack_name == 'ba':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, args.spherical_step_eps, args.norm,args.orth_step_factor,args.perp_step_factor, 
                        args.orthogonal_step_eps, args.max_queries, "cifar10", device,False)
    if args.attack_name == 'spsa':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net,norm=args.norm, device=device, eps=args.eps, learning_rate=args.learning_rate, delta=args.delta, spsa_samples=args.spsa_samples, 
                 sample_per_draw=args.sample_per_draw, nb_iter=args.max_queries, data_name="cifar10",early_stop_loss_threshold=None, IsTargeted=args.target)
    if args.attack_name == 'nes':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, nes_samples=args.nes_samples, sample_per_draw=args.nes_per_draw, 
                              p=args.norm, max_queries=args.max_queries, epsilon=args.epsilon, step_size=args.stepsize,
                device=device, data_name="cifar10", search_sigma=0.02, decay=1.0, random_perturb_start=True, target=args.target)
    if args.attack_name == 'nattack':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, eps=args.epsilon, max_queries=args.max_queries, device=device,data_name="cifar10", 
                              distance_metric=args.norm, target=args.target, sample_size=args.sample_size, lr=args.lr, sigma=args.sigma)
    if args.attack_name == 'evolutionary':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net,"cifar10",args.target,device, args.ccov, 
        args.decay_weight, args.max_queries, args.mu, args.sigma, args.maxlen)


    name = attack.__class__.__name__
    for i, (image,labels) in enumerate(test_loader, 1):
        batchsize = image.shape[0]
        image, labels = image.to(device), labels.to(device)
        out = net(image)
        out = torch.argmax(out, dim=1)
        
        adv_image= attack.forward(image, labels, None)
        distortion1 = torch.mean((adv_image-image)**2) / ((1-0)**2)
        distortion +=distortion1
        if args.norm==2:
            dist1 = torch.norm(adv_image - image, p=2)
            dist+=dist1
        elif args.norm==np.inf:
            dist1 = torch.max(torch.abs(adv_image - image)).item()
            dist+=dist1

    
        if i==1:
            filename = "%s_%s_%s.png" %(name, "cifar10", args.norm)
            load_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'test/test_out/', filename)
            save_image( torch.cat([image, adv_image], 0),  load_path, nrow=batchsize, padding=2, normalize=True, 
                        range=(0,1), scale_each=False, pad_value=0)

        
        #print(out_adv.shape)
        out_adv = net(adv_image)
    
        out_adv = torch.argmax(out_adv, dim=1)
        success_num +=(out_adv != labels).sum()
        
        test_num += (out == labels).sum()

        if i % 1 == 0:
            num = i*batchsize
            test_acc = test_num.item() / num
            adv_acc = 1-(success_num.item() / num)
            db_mean_distance = dist / num
            distortion_mean = distortion / num
            print("%s Image No. %d default model accuracy：%.2f %%" %("cifar10", i, test_acc*100 ))
            print("%s %s Image No. %a adversarial accuracy: %.2f %%" %(name, "cifar10", i, adv_acc*100))

    total_num = len(test_loader.dataset)
    final_test_acc = test_num.item() / total_num
    success_num = 1-(success_num.item() / num)
    db_mean_distance = dist / num
    distortion_mean = distortion / num
    print("%s Dataset Classification Accuracy：%.2f %%" %("cifar10", final_test_acc*100))
    print("%s %s Adversarial accuracy：%.2f %%" %(name, "cifar10", success_num*100))
 
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # data preprocess args 
    parser.add_argument("--gpu", type=str, default="2", help="Comma separated list of GPU ids")
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='')
    parser.add_argument('--attack_name', default='evolutionary', help= 'Dataset for this model', choices= ['ba','spsa', 'nes','nattack','evolutionary', 'fgsm', 'autoattack', 'cw','deepfool','pgd','bim', 'mim'])


    """#FGSM
    #parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=[np.inf, 2])
    parser.add_argument('--loss', default='cw', help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'cw'])
    parser.add_argument('--eps', type= float, default=0.01, help='linf: 8/255.0 and l2: 3.0')

    #spsa
    parser.add_argument('--learning_rate', type= float, default=0.006, help='learning_rate for spsa')
    parser.add_argument('--delta', type= float, default=1e-3, help='delta for spsa')
    parser.add_argument('--spsa_samples', type= int, default= 10, help='spsa_samples for spsa')
    parser.add_argument('--sample_per_draw', type= int, default=20, help='spsa_iters for spsa')
    #nes
    parser.add_argument('--epsilon', type= float, default= 16/255.0, help='eps for spsa, 0.05 for linf')
    #parser.add_argument('--stepsize', type= float, default=16/25500.0, help='learning_rate for spsa')
    #parser.add_argument('--max_iter', type= int, default=100, help='max_iter for spsa')
    parser.add_argument('--nes_samples', default= 10, help='nes_samples for nes')
    parser.add_argument('--nes_per_draw', type= int, default=20, help='nes_iters for nes')
    #nattack
    parser.add_argument('--sample_size', type= int, default=100, help='sample_size for nattack')
    #parser.add_argument('--lr', type= float, default= 0.02, help='lr for nattack')
    parser.add_argument('--sigma', type= float, default= 0.1, help='sigma for nattack')
    #Evolutionary
    parser.add_argument('--ccov', type= float, default= 0.001, help='eps for spsa, 0.05 for linf')
    parser.add_argument('--decay_weight', type= float, default=0.99, help='learning_rate for spsa')
    parser.add_argument('--mu', type= float, default=1e-2, help='max_iter for spsa')
    parser.add_argument('--sigmaa', type= float, default= 3e-2, help='nes_samples for nes')
    parser.add_argument('--maxlen', type= int, default=30, help='nes_iters for nes')
    parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])
    #autoattack
    parser.add_argument('--version', default='rand', help= 'version for autoattack', choices= ['standard', 'plus', 'rand'])
    parser.add_argument('--n_queries', default= 5000, help= 'n_queries for square')
    parser.add_argument('--stepsize', type= float, default=8/24000.0, help='linf: eps/steps and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=100, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    #parser.add_argument('--norm', default="Linf", help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=['Linf', 'L2', 'L1'])
    #cw
    parser.add_argument('--norm', default=2, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=[np.inf, 2])
    parser.add_argument('--kappa', type= float, default=0.0)
    parser.add_argument('--lr', type= float, default=0.2)
    parser.add_argument('--init_const', type= float, default=0.01)
    parser.add_argument('--binary_search_steps', type= int, default=4)
    parser.add_argument('--max_iter', type= int, default=200)

    #deepfool
    parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=[np.inf, 2])
    parser.add_argument('--batchsize', default=10, help= 'batchsize for this model')
    parser.add_argument('--overshoot', type= float, default=0.02)
    parser.add_argument('--max_iter', type= int, default=50)

    #pgd
    parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support apgd and deepfool', choices=[np.inf, 2])
    parser.add_argument('--eps', type= float, default=8/255.0, help='linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--stepsize', type= float, default=8/24000.0, help='linf: eps/steps and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=100, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    parser.add_argument('--n_queries', default= 5000, help= 'n_queries for square')
    parser.add_argument('--loss', default='cw', help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'cw'])

    #ba
    parser.add_argument('--norm', default= 2, help='You can choose linf and l2', choices=[np.inf, 1, 2])
    parser.add_argument('--spherical_step_eps', type= float, default=1e-2)
    parser.add_argument('--orthogonal_step_eps', type= float, default=1e-2)
    parser.add_argument('--orth_step_factor', type= float, default=0.97)
    parser.add_argument('--perp_step_factor', type= float, default=0.97)
    parser.add_argument('--max_queries', type= int, default=20000, help='max_queries for black-box attack based on queries')
    #bim
    parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support apgd and deepfool', choices=[np.inf, 2])
    parser.add_argument('--eps', type= float, default=8/255.0, help='linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--stepsize', type= float, default=8/24000.0, help='linf: eps/steps and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=100, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    parser.add_argument('--n_queries', default= 5000, help= 'n_queries for square')
    parser.add_argument('--loss', default='cw', help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'cw'])"""

    #mim
    parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support apgd and deepfool', choices=[np.inf, 2])
    parser.add_argument('--eps', type= float, default=8/255.0, help='linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--stepsize', type= float, default=8/24000.0, help='linf: eps/steps and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=100, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    parser.add_argument('--n_queries', default= 5000, help= 'n_queries for square')
    parser.add_argument('--loss', default='cw', help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'cw'])
    parser.add_argument('--decay_factor', default=1.0, help= 'float between 0 and 2.0')






    args = parser.parse_args()
    

    test(args)
