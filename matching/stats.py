import torch,torchvision
import tqdm

def get_mean_std(gan,gan_layers,discr,discr_layers,gan_mode,discr_mode,dataset,epochs,batch_size,device):
    '''Get the activation statistics from GAN and discr'''
    print("Collecting Dataset Statistics")
    gan_stats_list = []
    discr_stats_list = []
    gan_activs={}
    discr_activs={}
    with torch.no_grad():
        for iteration in tqdm.trange(0, epochs):
            z = dataset[0][iteration*batch_size: (iteration+1)*batch_size ]
            c = dataset[1][iteration*batch_size: (iteration+1)*batch_size ]
            if gan_mode=="biggan":
                img=gan(z,c,1)
                img=(img+1)/2

            
            for layer in gan_layers:
                if iteration==0:
                    gan_activs[layer]=[]    
                gan_activation=gan._retrieve_retained(layer,clear=True)
                gan_activs[layer].append(gan_activation)

            #clip condition here
            img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
            img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img)
            
            for layer in discr_layers:
                if iteration==0:
                    discr_activs[layer]=[]
                discr_activation=discr._retrieve_retained(layer,clear=True)
                discr_activs[layer].append(discr_activs)
        print("finished iterating for stats")

        
        final_gan_stats={}
        for layer in gan_layers:
            layer_activ_list=gan_activs[layer]
            concat_activs=torch.cat(layer_activ_list,dim=0)
            concat_activs=concat_activs.permute(1,0,2,3).contiguous()
            concat_activs=concat_activs.view(concat_activs.shape[0],-1)
            final_gan_stats[layer]=torch.mean(concat_activs,dim=-1,dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                    torch.std(concat_activs,dim=-1,dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)

        final_discr_stats={}
        for layer in discr_layers:
            layer_activ_list=discr_activs[layer]
            concat_activs=torch.cat(layer_activ_list,dim=0)
            concat_activs=concat_activs.permute(1,0,2,3).contiguous()
            concat_activs=concat_activs.view(concat_activs.shape[0],-1)
            final_discr_stats[layer]=torch.mean(concat_activs,dim=-1,dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                    torch.std(concat_activs,dim=-1,dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
        
        return final_gan_stats,final_discr_stats