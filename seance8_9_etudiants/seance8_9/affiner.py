import json
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MOTEUR, SORTIE = sys.argv[1], sys.argv[2]

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)
moteur = AutoModelForCausalLM.from_pretrained(MOTEUR, torch_dtype=torch.bfloat16, device_map="cuda") #AutoModelForCausalLM est une classe de la bibliothèque Hugging Face Transformers qui permet de charger automatiquement un modèle de langage (LLM) destiné à générer du texte.
moteur.gradient_checkpointing_enable()#stop les gradients des matrice Q K V
moteur.enable_input_require_grads()#stop les gradients des embeddings E

LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_rslora=True,  # stabilise les rangs plus élevés
    task_type="CAUSAL_LM")
entraines = [p for p in moteur.parameters() if p.requires_grad] #selectionne les paramètres du modèle qui sont entraînables (c'est-à-dire ceux pour lesquels les gradients sont calculés pendant l'entraînement). Ces paramètres sont généralement ceux qui ont été ajoutés ou modifiés par la technique LoRA (Low-Rank Adaptation), qui est utilisée pour adapter le modèle pré-entraîné à une tâche spécifique sans avoir à réentraîner tous les paramètres du modèle.
for p in entraines:
    p.data = p.data.float()
print("paramètres entraînés", sum(p.numel() for p in entraines))


def exemple(paire):
    demande = tokeniseur.apply_chat_template(
        [{"role": "user", "content": paire["demande"]}], tokenize=False, add_generation_prompt=True)
    ids_demande = tokeniseur(demande)["input_ids"]
    ids = tokeniseur(demande + paire["reponse"] + "<|im_end|>")["input_ids"]
    cibles = [-100] * len(ids_demande) + ids[len(ids_demande):]
    return torch.tensor([ids]).cuda(), torch.tensor([cibles]).cuda()
#cette fonction applique le gabarit de chat du tokeniseur sur la demande de la paire, puis elle convertit la demande et la réponse en identifiants de tokens. -100 est utilisé pour indiquer que les tokens correspondants à la demande ne doivent pas contribuer à la perte lors de l'entraînement, tandis que les tokens de la réponse sont utilisés comme cibles pour l'entraînement du modèle. On transforme les ids et les cibles en tenseurs PyTorch et on les déplace sur le GPU pour l'entraînement.

paires = json.load(open("paires.json", encoding="utf-8"))
adam = torch.optim.AdamW(entraines, lr=2e-4)
random.seed(0)
for epoque in range(2):
    random.shuffle(paires)
    pertes = []
    for paire in paires:
        ids, cibles = exemple(paire)
        perte = moteur(input_ids=ids, labels=cibles).loss #cette methode pour faire le masque car cibles = -100 -100 ... (reponse)200 150 ... donc caalcule lerreur de prediction du modele sur la reponse et pas sur la demande
        perte.backward()
        adam.step() #step() met à jour les paramètres du modèle en fonction des gradients calculés lors de la rétropropagation. Cela ajuste les poids du modèle pour réduire l'erreur de prédiction sur les données d'entraînement.
        adam.zero_grad() #suppression des gradients accumulés dans les paramètres du modèle. Cela est nécessaire pour éviter que les gradients des itérations précédentes n'affectent les mises à jour des paramètres lors de l'itération suivante.
        pertes.append(perte.item()) #perte.item() renvoie la valeur scalaire de la perte sous forme de nombre flottant. Cette valeur est ajoutée à la liste pertes pour suivre l'évolution de la perte au cours de l'entraînement.
    print(f"époque {epoque + 1}, perte moyenne {sum(pertes) / len(pertes):.3f}")

for p in entraines:
    p.data = p.data.to(torch.float16)
moteur.save_pretrained(SORTIE)
print("adaptateur enregistré dans le dossier", SORTIE)