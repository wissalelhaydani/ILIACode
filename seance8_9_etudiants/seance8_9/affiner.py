import json
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MOTEUR, SORTIE = sys.argv[1], sys.argv[2]

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)
moteur = AutoModelForCausalLM.from_pretrained(MOTEUR, torch_dtype=torch.bfloat16, device_map="cuda")
moteur.gradient_checkpointing_enable() #mnhsboux les gradients de k v q
moteur.enable_input_require_grads()

moteur = get_peft_model(moteur, LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_rslora=True,
    task_type="CAUSAL_LM"
))
entraines = [p for p in moteur.parameters() if p.requires_grad] # p sont des tenseur ( a et b)
for p in entraines:
    p.data = p.data.float() #p.data acceder au valeur numerique lhsab ce fait par cpu dakxi bax .float likatparsi l 32
print("paramètres entraînés", sum(p.numel() for p in entraines))


def exemple(paire):

    demande = tokeniseur.apply_chat_template(
        [
            {
                "role": "user",
                "content": paire["demande"]
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    ids_demande = tokeniseur(
        demande,
        add_special_tokens=False
    )["input_ids"]

    ids_reponse = tokeniseur(
        paire["reponse"] + "<|im_end|>",
        add_special_tokens=False
    )["input_ids"]

    ids = ids_demande + ids_reponse

    cibles = (
        [-100] * len(ids_demande)
        + ids_reponse
    )

    return (
        torch.tensor([ids]).cuda(),
        torch.tensor([cibles]).cuda()
    )

#on laisse la demander tokeniser pour savoir quoi fire flmasque

# Phase d'entraînement
paires = json.load(open("paires.json", encoding="utf-8"))

adam = torch.optim.AdamW(
    entraines,
    lr=1e-4,
    weight_decay=0.01
)

random.seed(0)
torch.manual_seed(0)

moteur.train()
moteur.config.use_cache = False

for epoque in range(4):

    random.shuffle(paires)
    pertes = []

    for paire in paires:

        adam.zero_grad()

        ids, cibles = exemple(paire)

        perte = moteur(
            input_ids=ids,
            labels=cibles
        ).loss

        perte.backward()

        # Éviter les gradients trop importants
        torch.nn.utils.clip_grad_norm_(
            entraines,
            max_norm=1.0
        )

        adam.step()

        pertes.append(perte.item())

    print(
        f"Époque {epoque + 1}, "
        f"perte moyenne : "
        f"{sum(pertes) / len(pertes):.3f}"
    )

for p in entraines:
    p.data = p.data.to(torch.float16) #stockage en 16 save les param lidartlihom pretaine
moteur.save_pretrained(SORTIE)
print("adaptateur enregistré dans le dossier", SORTIE)
