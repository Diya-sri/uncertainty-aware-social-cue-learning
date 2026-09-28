"""Measure prediction stability under realistic camera corruptions.

Uses the locked test split and writes reports/robustness.json. This is not training and
must never be used to tune on test results.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np

ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from emotion_model import EmotionPredictor, crop_face_like_app

LABELS=["anger","disgust","fear","happiness","neutral","sadness","surprise"]
def corrupt(x,kind):
    if kind=="dark": return np.clip(x.astype(np.float32)*.45,0,255).astype(np.uint8)
    if kind=="bright": return np.clip(x.astype(np.float32)*1.45+20,0,255).astype(np.uint8)
    if kind=="blur": return cv2.GaussianBlur(x,(11,11),0)
    if kind=="jpeg":
        ok,b=cv2.imencode('.jpg',x,[cv2.IMWRITE_JPEG_QUALITY,25]); return cv2.imdecode(b,cv2.IMREAD_COLOR) if ok else x
    if kind=="occlusion":
        y=x.copy();h,w=y.shape[:2];y[h//2:3*h//4,w//5:4*w//5]=0;return y
    return x
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/split/test');ap.add_argument('--model',default='models/emotion_model.npz');ap.add_argument('--out',default='reports/robustness.json');args=ap.parse_args()
    pred=EmotionPredictor(args.model); result={}
    for kind in ["clean","dark","bright","blur","jpeg","occlusion"]:
        right=answered=total=0
        for yi,label in enumerate(LABELS):
            for p in sorted((Path(args.data)/label).glob('*')):
                im=cv2.imread(str(p));
                if im is None: continue
                im=corrupt(im,kind);box=(0,0,im.shape[1],im.shape[0]);batch=np.stack([pred._prep(im,box)]);prob=np.asarray(pred._model(batch))[0];total+=1
                if prob.max()>=pred.unsure_below: answered+=1;right+=int(LABELS[int(prob.argmax())]==label)
        result[kind]={"images":total,"coverage":round(answered/max(total,1),4),"answered_accuracy":round(right/max(answered,1),4)}
        print(kind,result[kind])
    Path(args.out).write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
