import csv
from pathlib import Path

import timm
import torch
import torch_directml
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset


repo_ordner = Path(__file__).resolve().parent.parent

CHECKPOINT_ID = "timm/vit_base_patch16_224.augreg_in21k_ft_in1k"
CHECKPOINT_REVISION = "2ec9fb3d7bb664aac471ac44582c94d18de33780"
CHECKPOINT_DATEINAME = "linear_probing_20261116_184948_182532.pt"

KLASSEN_ZU_LABELS = {
    "Orange": 0,
    "Mandarine": 1,
    "Clementine": 2,
    "Apfel": 3,
    "Pfirsich": 4,
    "Nektarine": 5,
    "Birne": 6,
    "Banane": 7,
}
LABELS_ZU_KLASSEN = {label: klasse for klasse, label in KLASSEN_ZU_LABELS.items()}

ERWARTETE_VALIDIERUNGSBILDER = 6309
ERWARTETE_TREFFER = 6296
ERWARTETE_FEHLER = 13


class BildDataset(Dataset):
    def __init__(self, bildpfade, labels, vorverarbeitung):
        if len(bildpfade) != len(labels):
            raise ValueError(
                f"Anzahl Bildpfade ({len(bildpfade)}) und Labels "
                f"({len(labels)}) stimmt nicht ueberein."
            )
        self.bildpfade = bildpfade
        self.labels = labels
        self.vorverarbeitung = vorverarbeitung

    def __len__(self):
        return len(self.bildpfade)

    def __getitem__(self, index):
        with Image.open(self.bildpfade[index]) as bild:
            bild_tensor = self.vorverarbeitung(bild.convert("RGB"))
        return bild_tensor, self.labels[index], str(self.bildpfade[index])


validierung_zeilen = (
    repo_ordner / "configs" / "validierung.txt"
).read_text(encoding="utf-8").splitlines()
validierungspfade_pro_klasse = {klasse: [] for klasse in KLASSEN_ZU_LABELS}
for zeile in validierung_zeilen:
    klasse = Path(zeile).parent.name
    if klasse in KLASSEN_ZU_LABELS:
        bildpfad = repo_ordner / zeile
        if bildpfad not in validierungspfade_pro_klasse[klasse]:
            validierungspfade_pro_klasse[klasse].append(bildpfad)

fehlende_validierungsklassen = [
    klasse
    for klasse, bildpfade in validierungspfade_pro_klasse.items()
    if not bildpfade
]
if fehlende_validierungsklassen:
    raise ValueError(
        "Keine Validierungsbilder fuer folgende Klassen gefunden: "
        f"{fehlende_validierungsklassen}"
    )

validierung_bildpfade = []
validierung_labels = []
for bild_index in range(max(map(len, validierungspfade_pro_klasse.values()))):
    for klasse, label in KLASSEN_ZU_LABELS.items():
        if bild_index < len(validierungspfade_pro_klasse[klasse]):
            validierung_bildpfade.append(
                validierungspfade_pro_klasse[klasse][bild_index]
            )
            validierung_labels.append(label)


speicherpfad = repo_ordner / "models" / CHECKPOINT_DATEINAME
if not speicherpfad.is_file():
    raise FileNotFoundError(f"Checkpoint nicht gefunden: {speicherpfad}")

gespeichert = torch.load(
    speicherpfad,
    map_location="cpu",
    weights_only=True,
)


if gespeichert.get("checkpoint_id") != CHECKPOINT_ID:
    raise RuntimeError(
        "Checkpoint-ID stimmt nicht mit dem Trainingsstand ueberein: "
        f"{gespeichert.get('checkpoint_id')!r}"
    )
if gespeichert.get("checkpoint_revision") != CHECKPOINT_REVISION:
    raise RuntimeError(
        "Checkpoint-Revision stimmt nicht mit dem Trainingsstand ueberein: "
        f"{gespeichert.get('checkpoint_revision')!r}"
    )
if gespeichert.get("klassen_zu_labels") != KLASSEN_ZU_LABELS:
    raise RuntimeError(
        "Klassen-Zuordnung im Checkpoint stimmt nicht mit dem Auswertungscode ueberein."
    )


modell = timm.create_model(
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    pretrained=True,
    pretrained_cfg_overlay={
        "hf_hub_id": f"{CHECKPOINT_ID}@{CHECKPOINT_REVISION}",
    },
)

for parameter in modell.parameters():
    parameter.requires_grad = False

modell.reset_classifier(num_classes=len(KLASSEN_ZU_LABELS))
geraete_index = next(
    index
    for index in range(torch_directml.device_count())
    if "RX 6700 XT" in torch_directml.device_name(index)
)
geraet = torch_directml.device(geraete_index)
modell = modell.to(geraet)
print('Modellgerät:', torch_directml.device_name(geraete_index).rstrip('\x00'), next(modell.parameters()).device)
head = modell.get_classifier()
if not isinstance(head, torch.nn.Linear):
    raise TypeError(f"Erwartet wurde ein linearer Klassifikationskopf, erhalten: {type(head)}")
if (head.in_features, head.out_features, head.bias is not None) != (768, 8, True):
    raise RuntimeError(
        "Unerwartete Kopfkonfiguration: "
        f"in_features={head.in_features}, out_features={head.out_features}, "
        f"bias={head.bias is not None}"
    )

head.load_state_dict(gespeichert["head_state_dict"])
modell.eval()


vorverarbeitung_config = resolve_model_data_config(modell)
if vorverarbeitung_config != gespeichert["vorverarbeitung"]:
    raise RuntimeError("Vorverarbeitung stimmt nicht mit dem Trainingsstand überein.")
vorverarbeitung = create_transform(**vorverarbeitung_config, is_training=False)

validierungs_dataset = BildDataset(
    validierung_bildpfade,
    validierung_labels,
    vorverarbeitung,
)
batchgroesse = 8
validierungs_dataloader = DataLoader(
    validierungs_dataset,
    batch_size=batchgroesse,
    shuffle=False,
)


anzahl_klassen = len(KLASSEN_ZU_LABELS)
bildzahl_pro_klasse = [0] * anzahl_klassen
richtige_vorhersagen_pro_klasse = [0] * anzahl_klassen
verwechslungsmatrix = [[0] * anzahl_klassen for _ in range(anzahl_klassen)]
fehlerliste = []

with torch.no_grad():
    for batch, batch_labels, batch_pfade in validierungs_dataloader:
        batch = batch.to(geraet)
        batch_labels = batch_labels.to(geraet)
        logits = modell(batch)
        vorhergesagte_labels = logits.argmax(dim=1)

        for tatsaechliches_label, vorhergesagtes_label, bildpfad in zip(
            batch_labels.cpu().tolist(),
            vorhergesagte_labels.cpu().tolist(),
            batch_pfade,
        ):
            bildzahl_pro_klasse[tatsaechliches_label] += 1
            verwechslungsmatrix[tatsaechliches_label][vorhergesagtes_label] += 1
            if tatsaechliches_label == vorhergesagtes_label:
                richtige_vorhersagen_pro_klasse[tatsaechliches_label] += 1
            else:
                fehlerliste.append(
                    {
                        "bildpfad": Path(bildpfad).relative_to(repo_ordner).as_posix(),
                        "tatsaechlich": LABELS_ZU_KLASSEN[tatsaechliches_label],
                        "vorhergesagt": LABELS_ZU_KLASSEN[vorhergesagtes_label],
                    }
                )

klassen_statistik = [
    {
        "klasse": LABELS_ZU_KLASSEN[label],
        "bildzahl": bildzahl_pro_klasse[label],
        "richtige_vorhersagen": richtige_vorhersagen_pro_klasse[label],
        "trefferquote": (
            richtige_vorhersagen_pro_klasse[label] / bildzahl_pro_klasse[label]
        ),
    }
    for label in range(anzahl_klassen)
]
validierung_anzahl = sum(bildzahl_pro_klasse)
validierung_richtig = sum(richtige_vorhersagen_pro_klasse)

print("Checkpoint:", speicherpfad)
print("Validierungsbilder:", validierung_anzahl)
print("Richtige Vorhersagen:", validierung_richtig)
print("Fehler:", len(fehlerliste))
print(f"Trefferquote: {validierung_richtig / validierung_anzahl:.2%}")
print("\nTrefferquote pro Klasse:")
for statistik in klassen_statistik:
    print(f'{statistik['klasse']}: {statistik['richtige_vorhersagen']}/{statistik['bildzahl']} ({statistik['trefferquote']:.2%})')

print("\nFehlerliste (Zeilen=Original, Spalten=Vorhergesagt):")
print(" " * 14 + " ".join(f"{klasse:>12}" for klasse in LABELS_ZU_KLASSEN.values()))
for label, zeile in enumerate(verwechslungsmatrix):
    print(f'{LABELS_ZU_KLASSEN[label]:>14} ' + ' '.join((f'{wert:>12}' for wert in zeile)))

print("\nFalsch erkannte Bilder:")
for fehler in fehlerliste:
    print(f'Original: {fehler['tatsaechlich']} | Vorhergesagt: {fehler['vorhergesagt']} | Bild: {fehler['bildpfad']}')


if (
    validierung_anzahl != ERWARTETE_VALIDIERUNGSBILDER
    or validierung_richtig != ERWARTETE_TREFFER
    or len(fehlerliste) != ERWARTETE_FEHLER
):
    raise RuntimeError(
        "Wiederholbarkeit fehlgeschlagen: erwartet wurden "
        f"{ERWARTETE_TREFFER} Treffer und {ERWARTETE_FEHLER} Fehler bei "
        f"{ERWARTETE_VALIDIERUNGSBILDER} Bildern, erhalten wurden "
        f"{validierung_richtig} Treffer und {len(fehlerliste)} Fehler bei "
        f"{validierung_anzahl} Bildern."
    )


ergebnisordner = repo_ordner / "results"
ergebnisordner.mkdir(parents=True, exist_ok=True)
ergebnisdatei = ergebnisordner / f"auswertung_{speicherpfad.stem}.txt"

with ergebnisdatei.open("w", encoding="utf-8") as datei:
    print("Checkpoint:", speicherpfad.name, file=datei)
    print("Validierungsbilder:", validierung_anzahl, file=datei)
    print("Richtige Vorhersagen:", validierung_richtig, file=datei)
    print("Fehler:", len(fehlerliste), file=datei)
    print(f"Trefferquote: {validierung_richtig / validierung_anzahl:.2%}", file=datei)

    print("\nTrefferquote pro Klasse:", file=datei)
    for statistik in klassen_statistik:
        print(f'{statistik['klasse']}: {statistik['richtige_vorhersagen']}/{statistik['bildzahl']} ({statistik['trefferquote']:.2%})', file=datei)

    print("\nFalsch erkannte Bilder:", file=datei)
    for fehler in fehlerliste:
        print(f'Original: {fehler['tatsaechlich']} | Vorhergesagt: {fehler['vorhergesagt']} | Bild: {fehler['bildpfad']}', file=datei)

matrix_csv = ergebnisordner / f"auswertung_{speicherpfad.stem}_fehlerliste.csv"
with matrix_csv.open("w", newline="", encoding="utf-8") as datei:
    writer = csv.writer(datei)
    writer.writerow(["Original / Vorhergesagt", *LABELS_ZU_KLASSEN.values()])
    for label, zeile in enumerate(verwechslungsmatrix):
        writer.writerow([LABELS_ZU_KLASSEN[label], *zeile])

print("\nErgebnisse gespeichert:")
print(ergebnisdatei)
print(matrix_csv)
