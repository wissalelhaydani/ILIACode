import json
import os
import re
import urllib.request
from pathlib import Path

ADRESSE = "http://127.0.0.1:8000"
FIN = 151645
LEXIQUE = set(Path(__file__).with_name("lexique.txt").read_text(encoding="utf-8").split())


def lire_reglages(chemin="./reglages.json"):
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier)


def plier(mot):
    return mot


def canonicaliser(texte):
    return texte


def demander(texte, type_demande, modele="moteur", n_max=400):
    reglage = lire_reglages()[type_demande]
    corps = {"model": modele,
        "messages": [{"role": "user", "content": canonicaliser(texte)}],
        "max_tokens": n_max,
        "temperature": reglage["temperature"], #L'application choisit les paramètres (temperature et top_p) en fonction de la tâche demandée : génération de fonction, génération de tests ou explication.(dans le fichier reglages.json)
        "top_p": reglage["p_noyau"],      
        "seed": 0,
        "stop_token_ids": [FIN]} #quand il decide de s'arreter va donne fin = 151645
    requete = urllib.request.Request(
        ADRESSE + "/v1/chat/completions", data=json.dumps(corps).encode("utf-8"), 
        headers={"Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ["CLE_ILIACODE"]})
    with urllib.request.urlopen(requete, timeout=300) as reponse: #300 on attend la requete est reçue au serveur 
        choix = json.loads(reponse.read())["choices"][0]
    return choix["message"]["content"], choix["finish_reason"] == "stop"
#lorsque jai une demande je appliquer la fonction demander et dans la reponse je recupere le contenu et la raison (le finish reason) pour savoir si le moteur sait a finir ou bien il atteind le nombre maximal de token