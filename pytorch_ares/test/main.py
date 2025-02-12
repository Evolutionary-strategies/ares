from imghdr import tests
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

def load_data():

    transform = transforms.ToTensor()
    batch_size = 1
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    """testset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    
    download=True, transform=transform)
    indices = []
    class_units = [0 for i in range(10)] 
    number_per_class = 10 #Change this if you want to train with a bigger portion of the training data

    for i,data in enumerate(testset):
        _, label = data

        if class_units[label] < number_per_class:
            class_units[label] += 1
            indices.append(i)

    test_subset = torch.utils.data.Subset(testset, indices)"""
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    return test_loader
def test(args, syspath):
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    

    path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), syspath)
   
    model = Model()
    pretrain_dict = torch.load(path, map_location=device)
    model.load_state_dict(pretrain_dict)
    net = cifar10_model(device, model)

    test_loader = load_data()
    
    net.eval()
    success_num = 0
    test_num= 0


    attack_class = ATTACKS[args.attack_name]
    if args.attack_name == 'mim':
        attack = attack_class(net, epsilon=args.eps, norm=args.norm, decay_factor=args.decay_factor ,stepsize=args.stepsize, data_name="cifar10", steps=args.steps,target=False, device=device, loss=args.loss)
    if args.attack_name == 'bim':
        attack = attack_class(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, data_name="cifar10", steps=args.steps,target=False, device=device, loss=args.loss)
    if args.attack_name == 'pgd':
        attack = attack_class(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, data_name="cifar10", steps=args.steps,target=False, device=device, loss=args.loss)
    if args.attack_name == 'deepfool':
        attack = attack_class(net, args.overshoot, args.max_iter, args.norm, False, device)
    if args.attack_name == 'cw':
        attack = attack_class(net, device,args.norm, False, args.kappa, args.lr, args.init_const, args.max_iter, args.binary_search_steps, "cifar10")
    if args.attack_name == 'autoattack':
        attack = attack_class(net,norm=args.norm,steps=args.steps, query=args.n_queries, eps=args.eps, version=args.version,device=device)
    if args.attack_name == 'fgsm':
        attack = attack_class(net, p=args.norm, eps=args.eps, data_name="cifar10",target=False, loss=args.loss, device=device)
    if args.attack_name == 'ba':
        attack = attack_class(net, args.spherical_step_eps, args.norm,args.orth_step_factor,args.perp_step_factor, 
                        args.orthogonal_step_eps, args.max_queries, "cifar10", device,False)
    if args.attack_name == 'spsa':
        attack = attack_class(net,norm=args.norm, device=device, eps=args.eps, learning_rate=args.learning_rate, delta=args.delta, spsa_samples=args.spsa_samples, 
                 sample_per_draw=args.sample_per_draw, nb_iter=args.max_queries, data_name="cifar10",early_stop_loss_threshold=None, IsTargeted=False)
    if args.attack_name == 'nes':
        attack = attack_class(net, nes_samples=args.nes_samples, sample_per_draw=args.nes_per_draw, 
                              p=args.norm, max_queries=args.max_queries, epsilon=args.epsilon, step_size=args.stepsize,
                device=device, data_name="cifar10", search_sigma=0.02, decay=1.0, random_perturb_start=True, target=False)
    if args.attack_name == 'nattack':
        attack = attack_class(net, eps=args.epsilon, max_queries=args.max_queries, device=device,data_name="cifar10", 
                              distance_metric=args.norm, target=False, sample_size=args.sample_size, lr=args.lr, sigma=args.sigma)
    if args.attack_name == 'evolutionary':
        attack = attack_class(net,"cifar10",False,device, args.ccov, 
        args.decay_weight, args.max_queries, args.mu, args.sigma, args.maxlen)


    name = attack.__class__.__name__
    for i, (image,labels) in enumerate(test_loader, 1):
        batchsize = image.shape[0]
        image, labels = image.to(device), labels.to(device)
        out = net(image)
        out = torch.argmax(out, dim=1)
        
        adv_image= attack.forward(image, labels, None)
    
        if i==1:
            filename = "%s_%s_%s_NES.png" %(name, "cifar10", args.norm)
            load_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'test/test_out/', filename)
            save_image( torch.cat([adv_image], 0),  load_path, nrow=batchsize, padding=2, normalize=True, 
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
            print("%s Image No. %d default model accuracy: %.2f %%" %("cifar10", i, test_acc*100 ))
            print("%s %s Image No. %a adversarial accuracy: %.2f %%" %(name, "cifar10", i, adv_acc*100))

    total_num = len(test_loader.dataset)
    final_test_acc = test_num.item() / total_num
    success_num = 1-(success_num.item() / num)
    model_acc_str = "%s Dataset Classification Accuracy: %.2f %%\n" %("cifar10", final_test_acc*100)
    adv_acc_str = "%s %s Adversarial accuracy: %.2f %%\n" %(name, "cifar10", success_num*100)
    print(model_acc_str)
    print(adv_acc_str)
    return f"{model_acc_str} {adv_acc_str}"
 
    


if __name__ == "__main__":
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self



    paths = [
        #"test/evo/nets/gd_model054.pt",
        
        "test/evo/nets/es_model_graph.pt",
        
        ]
        #"test/evo/nets/nes_model_sigma01.pt",
        #"test/evo/nets/nes_model_sigma015.pt",
        #"test/evo/nets/es_model_sigma015_acc073_2.pt",

    args = [
        #SCORE BASED BLACK BOX
        #nattack
        AttrDict({
            "logname": "nattack_eps0.03",
            "attack_name":"nattack",
            "gpu":"2",
            "epsilon": 0.03,
            "norm": np.inf,
            "sample_size": 100,
            "lr":0.02,
            "sigma":0.1,
            "max_queries": 1000
        }),
        #SPSA
        AttrDict({
            "logname":"spsa_l2_0.7",
            "attack_name":"spsa",
            "gpu":"2",
            "norm": 2,
            "eps": 0.7,
            "learning_rate":0.01,
            "delta":0.01,
            "spsa_samples":10,
            "sample_per_draw": 20,
            "max_queries": 2000
        }),  
        AttrDict({
            "logname":"spsa_linf_0.03",
            "attack_name":"spsa",
            "gpu":"2",
            "norm": np.inf,
            "eps": 0.03,
            "learning_rate":0.01,
            "delta":0.01,
            "spsa_samples":10,
            "sample_per_draw": 20,
            "max_queries": 2000
        }),
        #NES
        AttrDict({
            "logname": "nes_2_eps1",
            "attack_name":"nes",
            "gpu":"2",
            "epsilon": 1,
            "norm": 2,
            "nes_samples": 10,
            "nes_per_draw":20,
            "max_queries": 2000,
            "stepsize": 1/100
        }),    
        AttrDict({
            "logname": "nes_linf_0.01",
            "attack_name":"nes",
            "gpu":"2",
            "epsilon": 0.01,
            "norm": np.inf,
            "nes_samples": 10,
            "nes_per_draw":20,
            "max_queries": 2000,
            "stepsize": 0.5/100
        }),    
        AttrDict({
            "logname": "nes_linf_0.03",
            "attack_name":"nes",
            "gpu":"2",
            "epsilon": 0.03,
            "norm": np.inf,
            "nes_samples": 10,
            "nes_per_draw":20,
            "max_queries": 2000,
            "stepsize": 0.5/100
        }),  
        #WHITEBOX
        #cw
        AttrDict({
            "logname": "cw_kappa0.1",
            "attack_name":"cw",
            "gpu":"2",
            "norm": 2,
            "lr":0.2,
            "kappa":0.1,
            "init_const":0.01,
            "binary_search_steps": 4,
            "max_iter": 50
        }),
        AttrDict({
            "logname": "cw_kappa0",
            "attack_name":"cw",
            "gpu":"2",
            "norm": 2,
            "lr":0.2,
            "kappa":0,
            "init_const":0.01,
            "binary_search_steps": 4,
            "max_iter": 50
        }),
        #deepfool
        AttrDict({
            "logname": "deepfool_l2_overshoot0.005",
            "attack_name":"deepfool",
            "gpu":"2",
            "norm": 2,
            "overshoot":0.005,
            "max_iter":50,
        }),
        AttrDict({
            "logname": "deepfool_l2_overshoot0.01",
            "attack_name":"deepfool",
            "gpu":"2",
            "norm": 2,
            "overshoot":0.1,
            "max_iter":50,
        }),
        AttrDict({
            "logname": "deepfool_l2_overshoot0.02",
            "attack_name":"deepfool",
            "gpu":"2",
            "norm": 2,
            "overshoot":0.2,
            "max_iter":50,
        }),
        AttrDict({
            "logname": "deepfool_l2_overshoot0.03",
            "attack_name":"deepfool",
            "gpu":"2",
            "norm": 2,
            "overshoot":0.3,
            "max_iter":50,
        }),
        #pgd
        AttrDict({
            "logname": "pgd_ce_l2_0.3",
            "attack_name": "pgd",
            "gpu": "2",
            "norm":2,
            "loss": "ce",
            "eps": 0.3,
            "stepsize": 2.5*0.3/100,
            "steps": 100,
        }),
        AttrDict({
            "logname": "pgd_ce_l2_0.5",
            "attack_name": "pgd",
            "gpu": "2",
            "norm":2,
            "loss": "ce",
            "eps": 0.5,
            "stepsize": 2.5*0.5/100,
            "steps": 100,
        }),
        AttrDict({
            "logname": "pgd_cw_l2_0.3",
            "attack_name": "pgd",
            "gpu": "2",
            "norm":2,
            "loss": "cw",
            "eps": 0.3,
            "stepsize": 2.5*0.3/100,
            "steps": 100,
        }),
        AttrDict({
            "logname": "pgd_cw_l2_0.5",
            "attack_name": "pgd",
            "gpu": "2",
            "norm":2,
            "loss": "cw",
            "eps": 0.5,
            "stepsize": 2.5*0.5/100,
            "steps": 100,
        }),
        AttrDict({
            "logname": "pgd_ce_linf_0.005",
            "attack_name": "pgd",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.005,
            "stepsize": 0.005/100,
            "steps": 100,
        }),
        AttrDict({
            "logname": "pgd_ce_linf_0.01",
            "attack_name": "pgd",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.01,
            "stepsize": 0.01/100,
            "steps": 100,
        }),
        AttrDict({
            "logname": "pgd_ce_linf_0.02",
            "attack_name": "pgd",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.02,
            "stepsize": 0.02/100,
            "steps": 100,
        }),
        AttrDict({
            "logname": "pgd_cw_linf_0.005",
            "attack_name": "pgd",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.005,
            "stepsize": 0.005/100,
            "steps": 100,
        }),
        AttrDict({
            "logname": "pgd_cw_linf_0.01",
            "attack_name": "pgd",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.01,
            "stepsize": 0.01/100,
            "steps": 100,
            }),
        AttrDict({
            "logname": "pgd_cw_linf_0.02",
            "attack_name": "pgd",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.02,
            "stepsize": 0.02/100,
            "steps": 100,
            }),   
        #mim
        AttrDict({
            "logname": "mim_ce_l2_0.3",
            "attack_name": "mim",
            "gpu": "2",
            "norm":2,
            "loss": "ce",
            "eps": 0.3,
            "stepsize": 2.5*0.3/100,
            "steps": 100,
            "decay_factor": 1.0,
        }),
        AttrDict({
            "logname": "mim_ce_l2_0.5",
            "attack_name": "mim",
            "gpu": "2",
            "norm":2,
            "loss": "ce",
            "eps": 0.5,
            "stepsize": 2.5*0.5/100,
            "steps": 100,
            "decay_factor": 1.0,
        }),
        AttrDict({
            "logname": "mim_cw_l2_0.3",
            "attack_name": "mim",
            "gpu": "2",
            "norm":2,
            "loss": "cw",
            "eps": 0.3,
            "stepsize": 2.5*0.3/100,
            "steps": 100,
            "decay_factor": 1.0,
        }),
        AttrDict({
            "logname": "mim_cw_l2_0.5",
            "attack_name": "mim",
            "gpu": "2",
            "norm":2,
            "loss": "cw",
            "eps": 0.5,
            "stepsize": 2.5*0.5/100,
            "steps": 100,
            "decay_factor": 1.0,
        }),
        AttrDict({
            "logname": "mim_ce_linf_0.005",
            "attack_name": "mim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.005,
            "stepsize": 0.005/100,
            "steps": 100,
            "decay_factor": 1.0,
        }),
        AttrDict({
            "logname": "mim_ce_linf_0.01",
            "attack_name": "mim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.01,
            "stepsize": 0.01/100,
            "steps": 100,
            "decay_factor": 1.0,
        }),
        AttrDict({
            "logname": "mim_ce_linf_0.02",
            "attack_name": "mim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.02,
            "stepsize": 0.02/100,
            "steps": 100,
            "decay_factor": 1.0,
        }),
        AttrDict({
            "logname": "mim_cw_linf_0.005",
            "attack_name": "mim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.005,
            "stepsize": 0.005/100,
            "steps": 100,
            "decay_factor": 1.0,
        }),
        AttrDict({
            "logname": "mim_cw_linf_0.01",
            "attack_name": "mim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.01,
            "stepsize": 0.01/100,
            "steps": 100,
            "decay_factor": 1.0,
            }),
        AttrDict({
            "logname": "mim_cw_linf_0.02",
            "attack_name": "mim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.02,
            "stepsize": 0.02/100,
            "steps": 100,
            "decay_factor": 1.0,
            }),   
        #bim
        AttrDict({
            "logname": "bim_ce_l2_0.3",
            "attack_name": "bim",
            "gpu": "2",
            "norm":2,
            "loss": "ce",
            "eps": 0.3,
            "stepsize": 2.5*0.3/100,
            "steps": 100
        }),
        AttrDict({
            "logname": "bim_ce_l2_0.5",
            "attack_name": "bim",
            "gpu": "2",
            "norm":2,
            "loss": "ce",
            "eps": 0.5,
            "stepsize": 2.5*0.5/100,
            "steps": 100
        }),
        AttrDict({
            "logname": "bim_cw_l2_0.3",
            "attack_name": "bim",
            "gpu": "2",
            "norm":2,
            "loss": "cw",
            "eps": 0.3,
            "stepsize": 2.5*0.3/100,
            "steps": 100
        }),
        AttrDict({
            "logname": "bim_cw_l2_0.5",
            "attack_name": "bim",
            "gpu": "2",
            "norm":2,
            "loss": "cw",
            "eps": 0.5,
            "stepsize": 2.5*0.5/100,
            "steps": 100
        }),
        AttrDict({
            "logname": "bim_ce_linf_0.005",
            "attack_name": "bim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.005,
            "stepsize": 0.005/100,
            "steps": 100
        }),
        AttrDict({
            "logname": "bim_ce_linf_0.01",
            "attack_name": "bim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.01,
            "stepsize": 0.01/100,
            "steps": 100
        }),
        AttrDict({
            "logname": "bim_ce_linf_0.02",
            "attack_name": "bim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.02,
            "stepsize": 0.02/100,
            "steps": 100
        }),
        AttrDict({
            "logname": "bim_cw_linf_0.005",
            "attack_name": "bim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.005,
            "stepsize": 0.005/100,
            "steps": 100
        }),
        AttrDict({
            "logname": "bim_cw_linf_0.01",
            "attack_name": "bim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.01,
            "stepsize": 0.01/100,
            "steps": 100
            }),
        AttrDict({
            "logname": "bim_cw_linf_0.02",
            "attack_name": "bim",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.02,
            "stepsize": 0.02/100,
            "steps": 100
            }),   
        #fgsm
        AttrDict({
            "logname": "fgsm_ce_l2_0.3",
            "attack_name": "fgsm",
            "gpu": "2",
            "norm":2,
            "loss": "ce",
            "eps": 0.3
        }),
        AttrDict({
            "logname": "fgsm_ce_l2_0.5",
            "attack_name": "fgsm",
            "gpu": "2",
            "norm":2,
            "loss": "ce",
            "eps": 0.5
        }),
        AttrDict({
            "logname": "fgsm_cw_l2_0.3",
            "attack_name": "fgsm",
            "gpu": "2",
            "norm":2,
            "loss": "cw",
            "eps": 0.3
        }),
        AttrDict({
            "logname": "fgsm_cw_l2_0.5",
            "attack_name": "fgsm",
            "gpu": "2",
            "norm":2,
            "loss": "cw",
            "eps": 0.5
        }),
        AttrDict({
            "logname": "fgsm_ce_linf_0.005",
            "attack_name": "fgsm",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.005
        }),
        AttrDict({
            "logname": "fgsm_ce_linf_0.01",
            "attack_name": "fgsm",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.01
        }),
        AttrDict({
            "logname": "fgsm_ce_linf_0.02",
            "attack_name": "fgsm",
            "gpu": "2",
            "norm":np.inf,
            "loss": "ce",
            "eps": 0.02
        }),
        AttrDict({
            "logname": "fgsm_cw_linf_0.005",
            "attack_name": "fgsm",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.005
        }),
        AttrDict({
            "logname": "fgsm_cw_linf_0.01",
            "attack_name": "fgsm",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.01
            }),
        AttrDict({
            "logname": "fgsm_cw_linf_0.02",
            "attack_name": "fgsm",
            "gpu": "2",
            "norm":np.inf,
            "loss": "cw",
            "eps": 0.02
            }),   

        AttrDict({
            "logname": "nattack_eps0.03",
            "attack_name":"nattack",
            "gpu":"2",
            "epsilon": 0.03,
            "norm": np.inf,
            "sample_size": 100,
            "lr":0.02,
            "sigma":0.1,
            "max_queries": 2000
        }),
        AttrDict({
            "logname": "nattack_eps0.05",
            "attack_name":"nattack",
            "gpu":"2",
            "epsilon": 0.05,
            "norm": np.inf,
            "sample_size": 100,
            "lr":0.02,
            "sigma":0.1,
            "max_queries": 2000
        }),
        AttrDict({
            "logname": "nattack_eps0.1",
            "attack_name":"nattack",
            "gpu":"2",
            "epsilon": 0.1,
            "norm": np.inf,
            "sample_size": 100,
            "lr":0.02,
            "sigma":0.1,
            "max_queries": 2000
        }),
        #SPSA
        AttrDict({
            "logname":"spsa_l2_0.5",
            "attack_name":"spsa",
            "gpu":"2",
            "norm": 2,
            "eps": 0.5,
            "learning_rate":0.01,
            "delta":0.01,
            "spsa_samples":10,
            "sample_per_draw": 20,
            "max_queries": 2000
        }), 
        AttrDict({
            "logname":"spsa_l2_1.0",
            "attack_name":"spsa",
            "gpu":"2",
            "norm": 2,
            "eps": 1.0,
            "learning_rate":0.01,
            "delta":0.01,
            "spsa_samples":10,
            "sample_per_draw": 20,
            "max_queries": 2000
        }), 
        AttrDict({
            "logname":"spsa_linf_0.01",
            "attack_name":"spsa",
            "gpu":"2",
            "norm": np.inf,
            "eps": 0.01,
            "learning_rate":0.01,
            "delta":0.01,
            "spsa_samples":10,
            "sample_per_draw": 20,
            "max_queries": 2000
        }),
        AttrDict({
            "logname":"spsa_linf_0.02",
            "attack_name":"spsa",
            "gpu":"2",
            "norm": np.inf,
            "eps": 0.02,
            "learning_rate":0.01,
            "delta":0.01,
            "spsa_samples":10,
            "sample_per_draw": 20,
            "max_queries": 2000
        }),
        #NES
        AttrDict({
            "logname": "nes_2_eps0.7",
            "attack_name":"nes",
            "gpu":"2",
            "epsilon": 0.7,
            "norm": 2,
            "nes_samples": 10,
            "nes_per_draw":20,
            "max_queries": 2000,
            "stepsize": 1/100
        }),    
        AttrDict({
            "logname": "nes_2_eps1.3",
            "attack_name":"nes",
            "gpu":"2",
            "epsilon": 1.3,
            "norm": 2,
            "nes_samples": 10,
            "nes_per_draw":20,
            "max_queries": 2000,
            "stepsize": 1/100
        }),
        AttrDict({
            "logname": "nes_linf_0.05",
            "attack_name":"nes",
            "gpu":"2",
            "epsilon": 0.05,
            "norm": np.inf,
            "nes_samples": 10,
            "nes_per_draw":20,
            "max_queries": 2000,
            "stepsize": 0.5/100
        }),
    ]
    for path in paths:
        for arg in args:
            f = open("results.txt", "a")
            f.write(arg.logname)
            f.write(test(arg, path))
    
    f.close()
   


        
    
