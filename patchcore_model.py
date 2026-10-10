import json, random
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

MEAN=torch.tensor([.485,.456,.406]).view(1,3,1,1)
STD=torch.tensor([.229,.224,.225]).view(1,3,1,1)

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT).eval()
        for p in self.net.parameters(): p.requires_grad=False
        self.out={}
        mods=dict(self.net.named_modules())
        mods['layer2'].register_forward_hook(lambda m,i,o:self.out.__setitem__('layer2',o))
        mods['layer3'].register_forward_hook(lambda m,i,o:self.out.__setitem__('layer3',o))
    @torch.inference_mode()
    def forward(self,x):
        self.out.clear(); self.net(x)
        return [self.out['layer2'],self.out['layer3']]

class SmartPatchCore:
    def __init__(self, image_size=256, coreset_ratio=.10, neighbors=9,
                 threshold_percentile=99.5, device=None):
        self.image_size=image_size; self.coreset_ratio=coreset_ratio
        self.neighbors=neighbors; self.threshold_percentile=threshold_percentile
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.extractor=FeatureExtractor().to(self.device)
        self.memory_bank=None; self.nn=None; self.threshold=None

    def _embed(self,img):
        from pathlib import Path
        from PIL import Image

        if isinstance(img, (str, Path)):
            img = Image.open(img)
        img = img.convert("RGB").resize(
            (self.image_size, self.image_size)
        )
        x=torch.from_numpy(np.asarray(img)).float().permute(2,0,1)/255.
        x=((x.unsqueeze(0)-MEAN)/STD).to(self.device)
        with torch.inference_mode(): fs=self.extractor(x)
        h=max(f.shape[-2] for f in fs); w=max(f.shape[-1] for f in fs)
        fs=[F.interpolate(f,(h,w),mode='bilinear',align_corners=False) if f.shape[-2:]!=(h,w) else f for f in fs]
        e=torch.cat(fs,1).squeeze(0).permute(1,2,0).reshape(-1,fs[0].shape[1]+fs[1].shape[1])
        return F.normalize(e,p=2,dim=1).cpu().numpy(),(h,w)

    def fit(self, paths, seed=42):
        seed_everything(seed); paths=list(paths)
        all_e=[]
        for i,p in enumerate(paths,1):
            all_e.append(self._embed(Image.open(p))[0])
            print(f'\rExtracting {i}/{len(paths)}',end='')
        print()
        e=np.vstack(all_e).astype('float32')
        n=max(1,int(len(e)*self.coreset_ratio))
        idx=np.random.default_rng(seed).choice(len(e),n,replace=False)
        self.memory_bank=e[idx]
        self._build_nn()
        scores=[self._score(Image.open(p))[0] for p in paths]
        self.threshold=float(np.percentile(scores,self.threshold_percentile))
        return self

    def _build_nn(self):
        self.nn=NearestNeighbors(n_neighbors=min(self.neighbors,len(self.memory_bank)),
                                 metric='euclidean',n_jobs=-1).fit(self.memory_bank)

    def _score(self, img):
        import numpy as np
        from pathlib import Path
        from sklearn.neighbors import NearestNeighbors
        
        if self.nn is None or self.memory_bank is None:
            if self.memory_bank is None:
                mem_path = Path("models/trained_model/memory_bank.npy")
                if mem_path.exists():
                    self.memory_bank = np.load(mem_path).astype('float32')
            
            self.nn = NearestNeighbors(n_neighbors=self.neighbors, algorithm='auto', metric='minkowski')
            if self.memory_bank is not None:
                self.nn.fit(self.memory_bank)

        patch_tokens, grid = self._embed(img)
        
        if hasattr(patch_tokens, 'cpu'):
            p = patch_tokens.cpu().numpy()
        else:
            p = np.array(patch_tokens)
        
        d, _ = self.nn.kneighbors(p, n_neighbors=min(self.neighbors, len(self.memory_bank)))
        anomaly_scores = np.mean(d, axis=1)
        
        max_idx = np.argmax(anomaly_scores)
        score = anomaly_scores[max_idx]
        
        H, W = grid
        amap = anomaly_scores.reshape(H, W)
        return score, amap
    def predict(self, img):
        score, amap = self._score(img)
        thresh = self.threshold if self.threshold is not None else 5.0
        is_anomaly = bool(score > thresh)
        return is_anomaly, float(score)

    def save(self,folder):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'memory_bank.npy',self.memory_bank)
        (folder/'metadata.json').write_text(json.dumps({
            'image_size':self.image_size,'coreset_ratio':self.coreset_ratio,
            'neighbors':self.neighbors,'threshold_percentile':self.threshold_percentile,
            'threshold':self.threshold},indent=2),encoding='utf8')

    @classmethod
    def load(cls, folder, device=None):
        folder = Path(folder)
        m = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        
        # إنشاء الكائن بالخصائص المحفوظة
        x = cls(
            image_size=m.get('image_size', 256),
            coreset_ratio=m.get('coreset_ratio', 0.1),
            neighbors=m.get('neighbors', 9),
            threshold_percentile=m.get('threshold_percentile', 99.5),
            device=device
        )
        
        # تحميل الذاكرة
        x.memory_bank = np.load(folder / 'memory_bank.npy').astype('float32')
        
        # إعادة بناء نموذج الجار الأقرب مباشرة لضمان عدم ظهور خطأ NoneType
        from sklearn.neighbors import NearestNeighbors
        x.nn = NearestNeighbors(n_neighbors=x.neighbors, algorithm='auto', metric='minkowski')
        x.nn.fit(x.memory_bank)
        
        x.threshold = m.get('threshold', None)
        return x
