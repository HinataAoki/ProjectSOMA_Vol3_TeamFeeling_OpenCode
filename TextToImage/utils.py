import clip
from dall_e import map_pixels, unmap_pixels, load_model
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import kornia
import PIL
import os, io, sys
import random
import imageio
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from google.colab import output
import requests
import cv2

def generate_image_from_text(generated_text):


    # 4.CLIPのモデル化
    # ! pip install ftfy regex
    model, preprocess = clip.load('ViT-B/32', jit=True)  
    model = model.eval()  

    # 5.DALL-Eのモデル化
    dec = load_model("https://cdn.openai.com/dall-e/decoder.pkl", 'cuda')

    # 初期設定
    im_shape = [512, 512, 3]
    sideX, sideY, channels = im_shape
    target_image_size = sideX
    tau_value = 2.
    
    img_dir = "output"
    if not os.path.exists(img_dir):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(img_dir)

    # 正規化と回転設定
    nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    augs = kornia.augmentation.RandomRotation(30).cuda()

    # テキスト入力
    text_input = generated_text

    # テキストを特徴ベクトルに変換
    token = clip.tokenize(text_input)  
    text_v = model.encode_text(token.cuda()).detach().clone()

    # 最適化手法の設定
    latent = Pars().cuda()  
    param = [latent.normu]  
    optimizer = torch.optim.Adam([{'params': param, 'lr': .01}]) 
    
    # 【チェック】パラメータから画像生成
    with torch.no_grad():
        out = unmap_pixels(torch.sigmoid(dec(latent())[:, :3].float()))
        displ(out.cpu()[0])

    print('latent().shape = ', latent().shape)
    print('dec(latent()).shape = ', dec(latent()).shape)
    print('out.shape = ', out.shape)

    # 学習ループ
    for iteration in range(1001):

        # --- 順伝播 ---
        # パラメータから画像を生成
        out = unmap_pixels(torch.sigmoid(dec(latent())[:, :3].float()))
        # 画像をランダム切り出し・回転  
        into = augment(out)
        # 画像を正規化
        into = nom((into))
        # 画像から特徴ベクトルを取得
        image_v = model.encode_image(into)
        # テキストと画像の特徴ベクトルのCOS類似度を計算 
        loss = -torch.cosine_similarity(text_v, image_v).mean()  

        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        # 学習率の調整
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*1.005
            g['lr'] = min(g['lr'], .12)

        # ログ表示      
        if iteration % 1 == 0:
            with torch.no_grad():
                # 生成画像の表示・保存
                out = unmap_pixels(torch.sigmoid(dec(latent())[:, :3]).float())  ###
                displ(out.cpu()[0], iteration)  ###​

            # データ表示
            print('iter = ',iteration)
            for g in optimizer.param_groups:
                print('lr = ', g['lr'])
            print('tau_value = ', tau_value)
            print('loss = ',loss.item())
            print('\n')
    # encoder(for mp4)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # output file name, encoder, fps, size(fit to image size)
    video = cv2.VideoWriter('video.mp4',fourcc, 20.0, (512, 512))

    if not video.isOpened():
        print("can't be opened")
        sys.exit()

    for i in range(0, 1000+1, 2):
        # hoge0000.png, hoge0001.png,..., hoge0090.png
        img = cv2.imread('output/output%d.png' % i)
        # can't read image, escape
        if img is None:
            print("can't read")
            break

        # add
        video.write(img)
        print(i)

    video.release()
    print('written')

# 画像表示・保存
def displ(img, num=0):
    display.clear_output(True)
    img = np.array(img)[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    imageio.imwrite('output/output%s.png'%num, np.array(img))
    return display.Image('output/output%s.png'%num)

# 画像のランダム切り出し
def augment(out, cutn=12):
    p_s = []
    for ch in range(cutn):
        sizey = int(torch.zeros(1,).uniform_(.5, .99)*sideY)
        sizex = int(torch.zeros(1,).uniform_(.5, .99)*sideX)
        offsetx = torch.randint(0, sideX - sizex, ())
        offsety = torch.randint(0, sideY - sizey, ())
        apper = out[:, :, offsetx:offsetx + sizex, offsety:offsety + sizey]
        apper = apper + .1*torch.rand(1,1,1,1).cuda()*torch.randn_like(apper, requires_grad=True)
        apper = torch.nn.functional.interpolate(apper, (224,224), mode='bilinear')
        p_s.append(apper)
    into = augs(torch.cat(p_s, 0))
    return into

# パラメータの設定
class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        hots = torch.nn.functional.one_hot((torch.arange(0, 8192).to(torch.int64)), num_classes=8192)
        rng = torch.zeros(1, 64*64, 8192).uniform_()
        for i in range(64*64):
            rng[0,i] = hots[[np.random.randint(8191)]]
        rng = rng.permute(0, 2, 1)
        self.normu = torch.nn.Parameter(rng.cuda().view(1, 8192, 64*64))
        
    def forward(self):
        tau_value = 2.0      
        normu = torch.nn.functional.gumbel_softmax(self.normu.reshape(1,64*64,8192), dim=1, tau=tau_value).view(1, 8192, 64, 64)
        return normu

