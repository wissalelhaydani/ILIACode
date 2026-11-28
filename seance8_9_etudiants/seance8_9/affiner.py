import json
import math
import random
import re
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MOTEUR, SORTIE = sys.argv[1], sys.argv[2]
DONNEES = sys.argv[3] if len(sys.argv) > 3 else "paires.json"
EPOQUES = int(sys.argv[4]) if len(sys.argv) > 4 else 3
FRACTION_VAL = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1   # 0 = pas de validation (tout sert à entraîner)
RANG, LR, ACCUM, LONG_MAX = 16, 2e-4, 8, 2048

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)
moteur = AutoModelForCausalLM.from_pretrained(MOTEUR, torch_dtype=torch.bfloat16, device_map="cuda")
moteur.config.use_cache = False
moteur.gradient_checkpointing_enable()
moteur.enable_input_require_grads()

# Changement 1 : rang 16, alpha 32 (facteur d'échelle 2), dropout, et TOUTES les matrices linéaires
moteur = get_peft_model(moteur, LoraConfig(
    r=RANG, lora_alpha=2 * RANG, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"))
entraines = [p for p in moteur.parameters() if p.requires_grad]
for p in entraines:
    p.data = p.data.float()
print("paramètres entraînés", sum(p.numel() for p in entraines))


def exemple(paire):
    demande = tokeniseur.apply_chat_template(
        [{"role": "user", "content": paire["demande"]}], tokenize=False, add_generation_prompt=True)
    ids_demande = tokeniseur(demande)["input_ids"]
    ids_reponse = tokeniseur(paire["reponse"] + "<|im_end|>")["input_ids"]   # la fin <|im_end|> fait partie de la cible
    ids = (ids_demande + ids_reponse)[:LONG_MAX]
    cibles = ([-100] * len(ids_demande) + ids_reponse)[:LONG_MAX]
    return torch.tensor([ids]).cuda(), torch.tensor([cibles]).cuda()


def groupe(paire):
    m = re.search(r"def (\w+)\(", paire["demande"] + paire["reponse"])
    return m.group(1) if m else paire["demande"]


@torch.no_grad()
def perte_validation(jeu):
    moteur.eval()
    total = sum(moteur(input_ids=i, labels=c).loss.item() for i, c in map(exemple, jeu))
    moteur.train()
    return total / len(jeu)


random.seed(0)
paires = json.load(open(DONNEES, encoding="utf-8"))

# Changement 2 : la validation est faite PAR FONCTION (jamais la même fonction en train et en val)
noms = sorted({groupe(p) for p in paires})
random.shuffle(noms)
noms_val = set(noms[:int(len(noms) * FRACTION_VAL)])
val = [p for p in paires if groupe(p) in noms_val]
train = [p for p in paires if groupe(p) not in noms_val]
print(f"entraînement {len(train)} paires, validation {len(val)} paires")

adam = torch.optim.AdamW(entraines, lr=LR, weight_decay=0.01)
pas_total = math.ceil(len(train) / ACCUM) * EPOQUES
# Changement 3 : warmup + décroissance cosinus du learning rate
planning = get_cosine_schedule_with_warmup(adam, int(0.1 * pas_total), pas_total)

meilleure, meilleurs_poids = float("inf"), None
moteur.train()
for epoque in range(EPOQUES):
    random.shuffle(train)
    pertes = []
    for k, paire in enumerate(train, 1):
        ids, cibles = exemple(paire)
        perte = moteur(input_ids=ids, labels=cibles).loss
        (perte / ACCUM).backward()              # Changement 4 : batch effectif de 8 (accumulation de gradient)
        pertes.append(perte.item())
        if k % ACCUM == 0 or k == len(train):
            torch.nn.utils.clip_grad_norm_(entraines, 1.0)   # Changement 5 : clipping du gradient
            adam.step()
            planning.step()
            adam.zero_grad()
    message = f"époque {epoque + 1}, perte train {sum(pertes) / len(pertes):.3f}"
    if val:
        pv = perte_validation(val)
        message += f", perte val {pv:.3f}"
        if pv < meilleure:                       # Changement 6 : on garde la meilleure époque
            meilleure = pv
            meilleurs_poids = [p.detach().clone() for p in entraines]
    print(message)

if meilleurs_poids is not None:
    for p, w in zip(entraines, meilleurs_poids):
        p.data.copy_(w)
for p in entraines:
    p.data = p.data.to(torch.float16)
moteur.save_pretrained(SORTIE)
print("adaptateur enregistré dans le dossier", SORTIE)