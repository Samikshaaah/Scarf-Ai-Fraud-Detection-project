import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import asyncio
import numpy as np
import json
import os

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

app = FastAPI()

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontened.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE = "experiment_history.json"

# ================= MODEL =================

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.projection = nn.Linear(128, 64)

    def forward(self, x):
        return self.projection(self.backbone(x))

def contrastive_loss(z1, z2, temp=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.matmul(z1, z2.T) / temp
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(sim, labels)

def corrupt(x, rate=0.4):
    batch_size = x.size(0)
    corrupted = x.clone()
    mask_features = torch.rand_like(x) < (rate * 0.25)
    corrupted = torch.where(mask_features, torch.zeros_like(x), corrupted)
    noise_mask = torch.rand_like(x) < (rate * 0.25)
    corrupted = torch.where(noise_mask, corrupted + torch.randn_like(x) * 0.1, corrupted)
    swap_mask = torch.rand_like(x) < (rate * 0.25)
    corrupted = torch.where(swap_mask, x[torch.randperm(batch_size)], corrupted)
    mix_mask = torch.rand_like(x) < (rate * 0.25)
    mixed = 0.5 * x + 0.5 * x[torch.randperm(batch_size)]
    corrupted = torch.where(mix_mask, mixed, corrupted)
    return corrupted

def fraud_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None and len(np.unique(y_true)) > 1 else 0.0
    return acc, precision, recall, f1, auc

# ================= TRAIN =================

@app.post("/train/")
async def train(file: UploadFile):
    print(f"[train] received file: {file.filename}")
    try:
        df = pd.read_csv(file.file)
        df = df.select_dtypes(include=["number"])
    except Exception as e:
        print(f"[train] error reading CSV: {e}")
        raise

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = y - y.min()  # 0-indexed

    # Stratified split to preserve fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

    # Class weights to handle fraud imbalance
    classes, counts = np.unique(y_train, return_counts=True)
    weights = torch.tensor(len(y_train) / (len(classes) * counts), dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=min(64, len(X_train) // 4),
        shuffle=True
    )

    async def stream():
        try:
            model = Encoder(X_train.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

            # -------- Contrastive Pretraining --------
            for epoch in range(50):
                epoch_loss = 0
                for batch_X, _ in train_loader:
                    z1 = model(corrupt(batch_X))
                    z2 = model(corrupt(batch_X))
                    loss = contrastive_loss(z1, z2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                scheduler.step(avg_loss)
                print(f"[pretrain] epoch {epoch} loss={avg_loss:.4f}")
                yield f"data: LOSS:{avg_loss}\n\n"
                await asyncio.sleep(0.03)

            # -------- Supervised Fine-tuning --------
            n_classes = len(classes)
            classifier = nn.Linear(128, n_classes)
            optimizer = torch.optim.Adam(
                list(model.backbone.parameters()) + list(classifier.parameters()), lr=1e-3
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            loss_fn = nn.CrossEntropyLoss(weight=weights)

            for epoch in range(50):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    preds = classifier(model.backbone(batch_X))
                    loss = loss_fn(preds, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                scheduler.step(avg_loss)
                print(f"[finetune] epoch {epoch} loss={avg_loss:.4f}")
                yield f"data: LOSS:{avg_loss}\n\n"
                await asyncio.sleep(0.03)

            # -------- Evaluate --------
            model.eval()
            with torch.no_grad():
                logits = classifier(model.backbone(X_test_t))
                probs  = torch.softmax(logits, dim=1)
                preds  = torch.argmax(logits, dim=1).numpy()
                # Use fraud class (1) probability for AUC
                fraud_prob = probs[:, 1].numpy() if n_classes == 2 else None

            scarf_acc, scarf_prec, scarf_rec, scarf_f1, scarf_auc = fraud_metrics(y_test, preds, fraud_prob)

            torch.save({'model': model.state_dict(), 'classifier': classifier.state_dict(), 'scaler': scaler}, 'scarf_model.pth')

            # -------- Baseline --------
            baseline = LogisticRegression(max_iter=1000, class_weight='balanced')
            baseline.fit(X_train, y_train)
            bl_preds = baseline.predict(X_test)
            bl_prob  = baseline.predict_proba(X_test)[:, 1] if n_classes == 2 else None
            bl_acc, bl_prec, bl_rec, bl_f1, bl_auc = fraud_metrics(y_test, bl_preds, bl_prob)

            # -------- Save History --------
            try:
                history = json.load(open(HISTORY_FILE)) if os.path.exists(HISTORY_FILE) else []
            except:
                history = []

            history.append({
                "dataset": file.filename,
                "scarf_accuracy": float(scarf_acc),
                "scarf_precision": float(scarf_prec),
                "scarf_recall": float(scarf_rec),
                "scarf_f1": float(scarf_f1),
                "scarf_auc": float(scarf_auc),
                "baseline_accuracy": float(bl_acc),
                "baseline_f1": float(bl_f1),
                "baseline_auc": float(bl_auc),
            })
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=4)

            yield (
                f"data: DONE:{scarf_acc}:{bl_acc}:{len(X_train)}:{len(X_test)}"
                f":{scarf_prec}:{scarf_rec}:{scarf_f1}:{scarf_auc}"
                f":{bl_prec}:{bl_rec}:{bl_f1}:{bl_auc}\n\n"
            )
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[train] error: {e}")
            yield f"data: ERROR:{e}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ================= PREDICT API =================

@app.post("/predict/")
async def predict(file: UploadFile):
    if not os.path.exists('scarf_model.pth'):
        return {"error": "No trained model found. Please train first."}

    try:
        df = pd.read_csv(file.file)
        df_numeric = df.select_dtypes(include=["number"])

        # Drop last column if it looks like a label (named 'Class', 'label', 'fraud', etc.)
        last_col = df_numeric.columns[-1].lower()
        if last_col in ['class', 'label', 'fraud', 'target', 'is_fraud']:
            feature_df = df_numeric.iloc[:, :-1]
        else:
            feature_df = df_numeric

        checkpoint = torch.load('scarf_model.pth', map_location='cpu', weights_only=False)
        scaler = checkpoint['scaler']
        n_features = scaler.n_features_in_

        # Align columns to what model was trained on
        X = feature_df.iloc[:, :n_features].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        n_classes = checkpoint['classifier']['weight'].shape[0]
        model = Encoder(n_features)
        model.load_state_dict(checkpoint['model'])
        classifier = nn.Linear(128, n_classes)
        classifier.load_state_dict(checkpoint['classifier'])

        model.eval()
        classifier.eval()
        with torch.no_grad():
            logits = classifier(model.backbone(X_tensor))
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1).numpy()
            fraud_probs = probs[:, 1].numpy() if n_classes == 2 else probs.max(dim=1).values.numpy()

        # Compute mean & std of training data from scaler for anomaly explanation
        train_mean = scaler.mean_
        train_std  = np.sqrt(scaler.var_)
        feature_names = list(feature_df.columns[:n_features]) if hasattr(feature_df, 'columns') else [f"Feature_{i+1}" for i in range(n_features)]

        results = []
        for i, (pred, prob) in enumerate(zip(preds, fraud_probs)):
            observations = None
            if pred == 1:
                row_vals = X[i]  # original unscaled values
                deviations = np.abs((row_vals - train_mean) / (train_std + 1e-8))
                top_idx = deviations.argsort()[-5:][::-1]  # top 5 most anomalous features
                flags = []
                for idx in top_idx:
                    dev = deviations[idx]
                    val = float(row_vals[idx])
                    fname = feature_names[idx]
                    direction = "unusually high" if row_vals[idx] > train_mean[idx] else "unusually low"
                    flags.append({
                        "feature": fname,
                        "value": round(val, 4),
                        "deviation": round(float(dev), 2),
                        "direction": direction,
                        "reason": f"{fname} is {direction} ({round(val,4)}) — {round(float(dev),1)}x away from normal"
                    })
                fraud_pct = round(float(prob) * 100, 2)
                risk = "🔴 Critical" if fraud_pct >= 85 else "🟠 High" if fraud_pct >= 60 else "🟡 Medium"
                observations = {
                    "risk_level": risk,
                    "fraud_probability": fraud_pct,
                    "summary": f"This transaction triggered {len(flags)} anomalous signals compared to normal transactions.",
                    "flags": flags
                }
            results.append({
                "row": i + 1,
                "prediction": "🚨 FRAUD" if pred == 1 else "✅ Legit",
                "confidence": round(float(prob if pred == 1 else 1 - prob) * 100, 2),
                "fraud_probability": round(float(prob) * 100, 2),
                "observations": observations
            })

        fraud_count = int(sum(preds == 1))
        return {
            "total": len(results),
            "fraud_count": fraud_count,
            "legit_count": len(results) - fraud_count,
            "results": results
        }
    except Exception as e:
        print(f"[predict] error: {e}")
        return {"error": str(e)}


# ================= HISTORY API =================

@app.get("/history/")
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
