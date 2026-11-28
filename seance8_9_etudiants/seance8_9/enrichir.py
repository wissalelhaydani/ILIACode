import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from mesurer import LANCER_TESTS, maison, tourne

random.seed(0)

# (signature, description, solution, assertions, explication)
NOUVELLES = [
    ("est_vide(texte)", "renvoie True si le texte est vide ou ne contient que des espaces",
     '''def est_vide(texte):
    return texte.strip() == ""''',
     '''assert est_vide("")
assert est_vide("   ")
assert not est_vide(" a ")''',
     "La fonction retire les espaces autour du texte puis regarde s'il reste quelque chose. Elle renvoie vrai quand il ne reste rien."),
    ("titre_valide(titre)", "renvoie True si le titre, sans espaces autour, fait entre 3 et 80 caractères",
     '''def titre_valide(titre):
    return 3 <= len(titre.strip()) <= 80''',
     '''assert titre_valide("Bug")
assert not titre_valide("ab")
assert titre_valide("a" * 80)
assert not titre_valide("a" * 81)''',
     "La fonction mesure le titre sans les espaces du début et de la fin. Elle l'accepte seulement s'il fait entre 3 et 80 caractères."),
    ("statut_suivant(statut)", "renvoie le statut suivant dans l'ordre ouvert, en_cours, ferme, et ferme reste ferme",
     '''def statut_suivant(statut):
    ordre = ["ouvert", "en_cours", "ferme"]
    return ordre[min(ordre.index(statut) + 1, 2)]''',
     '''assert statut_suivant("ouvert") == "en_cours"
assert statut_suivant("en_cours") == "ferme"
assert statut_suivant("ferme") == "ferme"''',
     "La fonction cherche la position du statut dans l'ordre ouvert, en_cours, ferme, et renvoie le suivant. Après ferme, elle reste sur ferme."),
    ("somme_priorites(tickets)", "renvoie la somme des priorités des tickets",
     '''def somme_priorites(tickets):
    return sum(t["priorite"] for t in tickets)''',
     '''assert somme_priorites([]) == 0
assert somme_priorites([{"priorite": 1}, {"priorite": 3}]) == 4''',
     "La fonction additionne la priorité de chaque ticket de la liste. Une liste vide donne 0."),
    ("priorite_max(tickets)", "renvoie la plus haute priorité, c'est-à-dire le plus petit nombre, ou None si la liste est vide",
     '''def priorite_max(tickets):
    if not tickets:
        return None
    return min(t["priorite"] for t in tickets)''',
     '''assert priorite_max([]) is None
assert priorite_max([{"priorite": 3}, {"priorite": 1}, {"priorite": 2}]) == 1''',
     "La fonction renvoie None pour une liste vide. Sinon elle prend le plus petit numéro de priorité, qui est la priorité la plus haute."),
    ("titres_ouverts(tickets)", 'renvoie la liste des titres des tickets dont le statut est "ouvert"',
     '''def titres_ouverts(tickets):
    return [t["titre"] for t in tickets if t["statut"] == "ouvert"]''',
     '''assert titres_ouverts([]) == []
assert titres_ouverts([{"titre": "A", "statut": "ouvert"}, {"titre": "B", "statut": "ferme"}]) == ["A"]
assert titres_ouverts([{"titre": "C", "statut": "ferme"}]) == []''',
     "La fonction garde les tickets ouverts et renvoie seulement leurs titres, dans l'ordre de la liste."),
    ("dernier_ticket(tickets)", "renvoie le ticket qui a le plus grand identifiant, ou None si la liste est vide",
     '''def dernier_ticket(tickets):
    if not tickets:
        return None
    return max(tickets, key=lambda t: t["id"])''',
     '''assert dernier_ticket([]) is None
assert dernier_ticket([{"id": 2}, {"id": 9}, {"id": 4}]) == {"id": 9}''',
     "La fonction compare les tickets selon leur identifiant et renvoie celui qui a le plus grand. Une liste vide donne None."),
    ("compter_assignes(tickets, utilisateur)", "renvoie le nombre de tickets assignés à l'utilisateur donné",
     '''def compter_assignes(tickets, utilisateur):
    return sum(1 for t in tickets if t["assigne"] == utilisateur)''',
     '''assert compter_assignes([], "sara") == 0
assert compter_assignes([{"assigne": "sara"}, {"assigne": "omar"}, {"assigne": "sara"}], "sara") == 2
assert compter_assignes([{"assigne": "omar"}], "sara") == 0''',
     "La fonction compte les tickets dont le champ assigne vaut l'utilisateur donné."),
    ("ticket_sans_assigne(ticket)", "renvoie True si le ticket n'a personne d'assigné, c'est-à-dire si le champ assigne est absent ou vaut None",
     '''def ticket_sans_assigne(ticket):
    return ticket.get("assigne") is None''',
     '''assert ticket_sans_assigne({})
assert ticket_sans_assigne({"assigne": None})
assert not ticket_sans_assigne({"assigne": "sara"})''',
     "La fonction lit le champ assigne sans erreur s'il manque. Elle renvoie vrai quand il est absent ou vaut None."),
    ("majuscules_titre(titre)", "renvoie le titre en majuscules, sans espaces au début ni à la fin",
     '''def majuscules_titre(titre):
    return titre.strip().upper()''',
     '''assert majuscules_titre("  bug login ") == "BUG LOGIN"
assert majuscules_titre("   ") == ""''',
     "La fonction retire les espaces autour du titre puis met toutes ses lettres en majuscules."),
    ("mots_cles(titre)", "renvoie la liste des mots du titre en minuscules",
     '''def mots_cles(titre):
    return titre.lower().split()''',
     '''assert mots_cles("Bug LOGIN page") == ["bug", "login", "page"]
assert mots_cles("") == []''',
     "La fonction met le titre en minuscules puis le découpe sur les espaces. Un titre vide donne une liste vide."),
    ("remplacer_espaces(texte)", "renvoie le texte où chaque espace est remplacé par un tiret bas",
     '''def remplacer_espaces(texte):
    return texte.replace(" ", "_")''',
     '''assert remplacer_espaces("bug login") == "bug_login"
assert remplacer_espaces("") == ""
assert remplacer_espaces("a  b") == "a__b"''',
     "La fonction remplace chaque espace du texte par un tiret bas. Deux espaces de suite donnent deux tirets bas."),
    ("est_numerique(texte)", "renvoie True si le texte est fait uniquement de chiffres et n'est pas vide",
     '''def est_numerique(texte):
    return len(texte) > 0 and texte.isdigit()''',
     '''assert est_numerique("123")
assert not est_numerique("")
assert not est_numerique("12a")''',
     "La fonction vérifie que le texte n'est pas vide et que tous ses caractères sont des chiffres."),
    ("nombre_commentaires(ticket)", "renvoie le nombre de commentaires du ticket, ou 0 si le champ commentaires est absent",
     '''def nombre_commentaires(ticket):
    return len(ticket.get("commentaires", []))''',
     '''assert nombre_commentaires({}) == 0
assert nombre_commentaires({"commentaires": ["a", "b"]}) == 2''',
     "La fonction compte les éléments de la liste des commentaires. Si le ticket n'a pas ce champ, elle renvoie 0."),
    ("a_commentaires(ticket)", "renvoie True si le ticket a au moins un commentaire",
     '''def a_commentaires(ticket):
    return len(ticket.get("commentaires", [])) > 0''',
     '''assert not a_commentaires({})
assert not a_commentaires({"commentaires": []})
assert a_commentaires({"commentaires": ["vu"]})''',
     "La fonction renvoie vrai quand la liste des commentaires contient au moins un élément. Un ticket sans commentaires donne faux."),
    ("jours_restants(echeance, aujourd_hui)", "renvoie le nombre de jours entre la date du jour et l'échéance, au format AAAA-MM-JJ, négatif si l'échéance est passée",
     '''from datetime import date

def jours_restants(echeance, aujourd_hui):
    return (date.fromisoformat(echeance) - date.fromisoformat(aujourd_hui)).days''',
     '''assert jours_restants("2026-11-20", "2026-11-16") == 4
assert jours_restants("2026-11-10", "2026-11-16") == -6''',
     "La fonction lit les deux dates, soustrait la date du jour à l'échéance et renvoie l'écart en jours. Le résultat est négatif si l'échéance est passée."),
    ("annee_de(date_iso)", "renvoie l'année d'une date AAAA-MM-JJ sous forme d'entier",
     '''def annee_de(date_iso):
    return int(date_iso.split("-")[0])''',
     '''assert annee_de("2026-11-16") == 2026
assert annee_de("1999-01-01") == 1999''',
     "La fonction découpe la date sur les tirets et convertit le premier morceau en entier."),
    ("est_weekend(date_iso)", "renvoie True si la date AAAA-MM-JJ tombe un samedi ou un dimanche",
     '''from datetime import date

def est_weekend(date_iso):
    return date.fromisoformat(date_iso).weekday() >= 5''',
     '''assert est_weekend("2026-11-28")
assert est_weekend("2026-11-29")
assert not est_weekend("2026-11-30")''',
     "La fonction calcule le jour de la semaine, de 0 pour lundi à 6 pour dimanche. Elle renvoie vrai pour 5 et 6, donc samedi et dimanche."),
    ("ajouter_jours(date_iso, n)", "renvoie la date AAAA-MM-JJ obtenue en ajoutant n jours à la date donnée",
     '''from datetime import date, timedelta

def ajouter_jours(date_iso, n):
    return (date.fromisoformat(date_iso) + timedelta(days=n)).isoformat()''',
     '''assert ajouter_jours("2026-11-16", 5) == "2026-11-21"
assert ajouter_jours("2026-12-30", 3) == "2027-01-02"''',
     "La fonction lit la date, y ajoute n jours, puis la réécrit au format AAAA-MM-JJ. Elle gère les changements de mois et d'année."),
    ("plus_grand_ecart(nombres)", "renvoie l'écart entre le plus grand et le plus petit nombre, ou 0 si la liste est vide",
     '''def plus_grand_ecart(nombres):
    if not nombres:
        return 0
    return max(nombres) - min(nombres)''',
     '''assert plus_grand_ecart([]) == 0
assert plus_grand_ecart([4, 9, 1]) == 8
assert plus_grand_ecart([5]) == 0''',
     "La fonction soustrait le plus petit nombre au plus grand. Une liste vide donne 0."),
    ("tickets_par_utilisateur(tickets)", "renvoie un dictionnaire qui associe chaque utilisateur au nombre de tickets qui lui sont assignés",
     '''def tickets_par_utilisateur(tickets):
    comptes = {}
    for t in tickets:
        comptes[t["assigne"]] = comptes.get(t["assigne"], 0) + 1
    return comptes''',
     '''assert tickets_par_utilisateur([]) == {}
assert tickets_par_utilisateur([{"assigne": "sara"}, {"assigne": "omar"}, {"assigne": "sara"}]) == {"sara": 2, "omar": 1}''',
     "La fonction parcourt les tickets et ajoute un au compteur de la personne assignée. Elle renvoie un dictionnaire utilisateur vers nombre de tickets."),
    ("fusionner_etiquettes(a, b)", "renvoie la liste des étiquettes de a suivies de celles de b qui n'y sont pas déjà",
     '''def fusionner_etiquettes(a, b):
    return a + [e for e in b if e not in a]''',
     '''assert fusionner_etiquettes(["bug"], ["bug", "ui"]) == ["bug", "ui"]
assert fusionner_etiquettes([], ["a"]) == ["a"]
assert fusionner_etiquettes(["a"], []) == ["a"]''',
     "La fonction garde toutes les étiquettes de a, puis ajoute celles de b qui n'y figurent pas encore. Il n'y a donc pas de doublons venant de b."),
    ("a_etiquette(ticket, etiquette)", "renvoie True si l'étiquette est dans la liste etiquettes du ticket",
     '''def a_etiquette(ticket, etiquette):
    return etiquette in ticket.get("etiquettes", [])''',
     '''assert a_etiquette({"etiquettes": ["bug"]}, "bug")
assert not a_etiquette({"etiquettes": ["bug"]}, "ui")
assert not a_etiquette({}, "bug")''',
     "La fonction cherche l'étiquette dans la liste des étiquettes du ticket. Un ticket sans étiquettes donne faux."),
    ("retirer_etiquette(ticket, etiquette)", "retire l'étiquette du ticket si elle y est, et renvoie le ticket",
     '''def retirer_etiquette(ticket, etiquette):
    if etiquette in ticket.get("etiquettes", []):
        ticket["etiquettes"].remove(etiquette)
    return ticket''',
     '''assert retirer_etiquette({"etiquettes": ["bug", "ui"]}, "bug") == {"etiquettes": ["ui"]}
assert retirer_etiquette({}, "bug") == {}
assert retirer_etiquette({"etiquettes": ["ui"]}, "bug") == {"etiquettes": ["ui"]}''',
     "La fonction enlève l'étiquette de la liste si elle y figure, sinon elle ne change rien. Elle renvoie toujours le ticket."),
    ("est_palindrome(mot)", "renvoie True si le mot est identique lu à l'envers, sans tenir compte des majuscules",
     '''def est_palindrome(mot):
    mot = mot.lower()
    return mot == mot[::-1]''',
     '''assert est_palindrome("Radar")
assert not est_palindrome("ticket")
assert est_palindrome("")''',
     "La fonction met le mot en minuscules et le compare à lui-même retourné. Elle renvoie vrai s'ils sont égaux."),
    ("borner(valeur, minimum, maximum)", "renvoie la valeur ramenée entre minimum et maximum",
     '''def borner(valeur, minimum, maximum):
    return max(minimum, min(valeur, maximum))''',
     '''assert borner(5, 1, 3) == 3
assert borner(0, 1, 3) == 1
assert borner(2, 1, 3) == 2''',
     "La fonction coupe la valeur au maximum si elle le dépasse, et la relève au minimum si elle est en dessous. Sinon elle la laisse telle quelle."),
    ("pair_ou_impair(n)", 'renvoie "pair" si n est pair et "impair" sinon',
     '''def pair_ou_impair(n):
    return "pair" if n % 2 == 0 else "impair"''',
     '''assert pair_ou_impair(4) == "pair"
assert pair_ou_impair(7) == "impair"
assert pair_ou_impair(0) == "pair"''',
     "La fonction regarde le reste de la division de n par 2. Un reste de 0 donne pair, sinon impair."),
    ("total_heures(durees_minutes)", "renvoie le total d'une liste de durées en minutes, converti en heures entières",
     '''def total_heures(durees_minutes):
    return sum(durees_minutes) // 60''',
     '''assert total_heures([30, 45, 50]) == 2
assert total_heures([]) == 0
assert total_heures([59]) == 0''',
     "La fonction additionne les durées en minutes puis divise par 60 en gardant seulement la partie entière."),
    ("commence_par_majuscule(titre)", "renvoie True si le titre commence par une majuscule",
     '''def commence_par_majuscule(titre):
    return titre[:1].isupper()''',
     '''assert commence_par_majuscule("Bug")
assert not commence_par_majuscule("bug")
assert not commence_par_majuscule("")''',
     "La fonction regarde la première lettre du titre. Un titre vide donne faux."),
    ("longueur_moyenne_titres(tickets)", "renvoie la longueur moyenne des titres, arrondie à l'entier, ou 0 si la liste est vide",
     '''def longueur_moyenne_titres(tickets):
    if not tickets:
        return 0
    return round(sum(len(t["titre"]) for t in tickets) / len(tickets))''',
     '''assert longueur_moyenne_titres([]) == 0
assert longueur_moyenne_titres([{"titre": "abcd"}, {"titre": "ab"}]) == 3''',
     "La fonction additionne la longueur de chaque titre, divise par le nombre de tickets et arrondit. Une liste vide donne 0."),
    ("ticket_ouvert(ticket)", 'renvoie True si le statut du ticket est "ouvert"',
     '''def ticket_ouvert(ticket):
    return ticket["statut"] == "ouvert"''',
     '''assert ticket_ouvert({"statut": "ouvert"})
assert not ticket_ouvert({"statut": "ferme"})
assert not ticket_ouvert({"statut": "en_cours"})''',
     "La fonction compare le statut du ticket à \"ouvert\" et renvoie le résultat."),
    ("echeance_depassee(echeance, aujourd_hui)", "renvoie True si la date d'échéance est strictement avant la date du jour, au format AAAA-MM-JJ",
     '''def echeance_depassee(echeance, aujourd_hui):
    return echeance < aujourd_hui''',
     '''assert echeance_depassee("2026-11-10", "2026-11-16")
assert not echeance_depassee("2026-11-16", "2026-11-16")
assert not echeance_depassee("2026-11-20", "2026-11-16")''',
     "La fonction compare les deux dates écrites en AAAA-MM-JJ, ce qui suit l'ordre du calendrier. Le jour même n'est pas encore dépassé."),
    ("enlever_prefixe(titre)", "renvoie le titre sans le préfixe \"[BUG] \" s'il le contient au début",
     '''def enlever_prefixe(titre):
    return titre.removeprefix("[BUG] ")''',
     '''assert enlever_prefixe("[BUG] login") == "login"
assert enlever_prefixe("login") == "login"''',
     "La fonction retire le préfixe [BUG] suivi d'un espace au début du titre. Si le titre ne commence pas ainsi, il reste inchangé."),
]

SWAPS = [(" > ", " >= "), (" >= ", " > "), (" < ", " <= "), (" <= ", " < "), (" == ", " != "),
         (" and ", " or "), (" or ", " and "), (" + 1", " - 1"), (" not in ", " in "), ("[0]", "[-1]")]


def mutants(code):
    sortie = []
    for a, b in SWAPS:
        for m in re.finditer(re.escape(a), code):
            sortie.append(code[:m.start()] + b + code[m.end():])
    return sortie


def code_bloc(texte):
    m = re.search(r"```python\n(.*?)```", texte, re.S)
    return m.group(1) if m else ""


jeu = json.load(open("jeu_test.json", encoding="utf-8"))
interdits = {m for d in jeu for m in re.findall(r"\b([A-Za-z_]+)\(", d["demande"])}
paires = json.load(open("paires.json", encoding="utf-8"))


def depuis_nouvelle(sig, desc, sol, ver, expl):
    nom = sig.split("(")[0]
    if nom in interdits:
        return []
    if not tourne(sol + "\n" + ver):
        print("solution fausse, ignorée :", nom)
        return []
    tests = "\n\n".join(f"def test_cas_{i}():\n    {ligne}" for i, ligne in enumerate(ver.splitlines(), 1))
    if not tourne(sol + "\n" + tests + LANCER_TESTS):
        print("tests faux, ignorés :", nom)
        return []
    tues = sum(1 for m in mutants(sol) if not tourne(m + "\n" + tests + LANCER_TESTS))
    if tues == 0 and mutants(sol):
        print("tests trop faibles (aucun mutant détecté) :", nom)
    bloc = f"```python\n{sol}\n```"
    rep_f, rep_t = f"```python\n{sol}\n```", f"```python\n{tests}\n```"
    sortie = []
    for debut in [f"Écris la fonction {sig} qui {desc}.", f"Code la fonction Python {sig}, elle {desc}.",
                  f"kteb liya la fonction {sig} qui {desc}.", f"3afak dir liya la fonction {sig} qui {desc}."]:
        sortie.append({"type": "fonction", "demande": debut, "reponse": rep_f})
    for debut in [f"Écris les tests de {nom}.\n", f"kteb liya les tests dyal {nom}\n", f"Donne-moi des tests pour {nom}.\n"]:
        sortie.append({"type": "tests", "demande": debut + bloc, "reponse": rep_t})
    for debut in [f"Explique ce que fait {nom}.\n", f"chrah liya chno kadir had l code\n", f"Que fait {nom} ?\n"]:
        sortie.append({"type": "explication", "demande": debut + bloc, "reponse": expl})
    return sortie


def depuis_existantes(paires):
    sortie = []
    for p in paires:
        d = p["demande"]
        if p["type"] == "fonction" and d.startswith("Écris la fonction "):
            reste = d[len("Écris la fonction "):]
            for debut in ["kteb liya la fonction ", "3afak dir liya la fonction ", "Implémente en Python la fonction "]:
                sortie.append({**p, "demande": debut + reste})
        elif p["type"] == "tests" and d.startswith("Écris les tests de "):
            nom, bloc = d[len("Écris les tests de "):].split(".\n", 1)
            sortie.append({**p, "demande": f"Donne-moi des tests pour {nom}.\n{bloc}"})
            sortie.append({**p, "demande": f"bghit les tests dyal {nom}\n{bloc}"})
        elif p["type"] == "explication" and d.startswith("Explique ce que fait "):
            nom, bloc = d[len("Explique ce que fait "):].split(".\n", 1)
            sortie.append({**p, "demande": f"chrah liya chno kadir had l code\n{bloc}"})
            sortie.append({**p, "demande": f"Que fait {nom} ?\n{bloc}"})
            sortie.append({**p, "demande": f"chno kat3mel {nom} ?\n{bloc}"})
    return sortie


def verifier(p):
    if not maison({"type": p["type"]}, p["reponse"]):
        return "format"
    if p["type"] == "tests" and not tourne(code_bloc(p["demande"]) + "\n" + code_bloc(p["reponse"]) + LANCER_TESTS):
        return "tests_echouent"
    return None


toutes = paires + depuis_existantes(paires)
for args in NOUVELLES:
    toutes += depuis_nouvelle(*args)

vues, uniques = set(), []
for p in toutes:
    if p["demande"] not in vues:
        vues.add(p["demande"])
        uniques.append(p)

with ThreadPoolExecutor(8) as lot:
    verdicts = list(lot.map(verifier, uniques))
bonnes = [p for p, v in zip(uniques, verdicts) if v is None]
rejets = [(p["demande"][:50], v) for p, v in zip(uniques, verdicts) if v]
print("rejetées :", rejets)

comptes = {}
for p in bonnes:
    comptes[p["type"]] = comptes.get(p["type"], 0) + 1
json.dump(bonnes, open("paires_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(paires), "paires au départ →", len(bonnes), "dans paires_v2.json", comptes)