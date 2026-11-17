from datetime import datetime
from pathlib import Path

import timm
import torch
import torch_directml

from peft import LoraConfig, get_peft_model
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset

repo_ordner = Path(__file__).resolve().parent.parent

CHECKPOINT_ID = "timm/vit_base_patch16_224.augreg_in21k_ft_in1k"
CHECKPOINT_REVISION = "2ec9fb3d7bb664aac471ac44582c94d18de33780"

SEED = 32427

LORA_RANG = 4
LORA_ALPHA = 2 * LORA_RANG
LORA_DROPOUT = 0.05
LORA_ZIELMODULE = ["attn.qkv", "attn.proj"]

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
        return bild_tensor, self.labels[index]


training_zeilen = (
    repo_ordner / "configs" / "training.txt"
).read_text(encoding="utf-8").splitlines()
bildpfade_pro_klasse = {klasse: [] for klasse in KLASSEN_ZU_LABELS}
for zeile in training_zeilen:
    klasse = Path(zeile).parent.name
    if klasse in KLASSEN_ZU_LABELS:
        bildpfad = repo_ordner / zeile
        if bildpfad not in bildpfade_pro_klasse[klasse]:
            bildpfade_pro_klasse[klasse].append(bildpfad)

fehlende_klassen = [
    klasse
    for klasse, bildpfade in bildpfade_pro_klasse.items()
    if not bildpfade
]
if fehlende_klassen:
    raise ValueError(
        f"Keine Trainingsbilder fuer folgende Klassen gefunden: {fehlende_klassen}"
    )

bildpfade = []
labels = []
for bild_index in range(max(map(len, bildpfade_pro_klasse.values()))):
    for klasse, label in KLASSEN_ZU_LABELS.items():
        if bild_index < len(bildpfade_pro_klasse[klasse]):
            bildpfade.append(bildpfade_pro_klasse[klasse][bild_index])
            labels.append(label)

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

modell = timm.create_model(
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    pretrained=True,
    pretrained_cfg_overlay={
        "hf_hub_id": f"{CHECKPOINT_ID}@{CHECKPOINT_REVISION}",
    },
)

torch.manual_seed(SEED)
modell.reset_classifier(num_classes=len(KLASSEN_ZU_LABELS))
vorverarbeitung_config = resolve_model_data_config(modell)
vorverarbeitung = create_transform(**vorverarbeitung_config, is_training=False)
trainings_dataset = BildDataset(bildpfade, labels, vorverarbeitung)
validierungs_dataset = BildDataset(
    validierung_bildpfade,
    validierung_labels,
    vorverarbeitung,
)

anzahl_lora_parameter_pro_rang = sum(
    modul.in_features + modul.out_features
    for name, modul in modell.named_modules()
    if isinstance(modul, torch.nn.Linear)
    and any(name.endswith(zielmodul) for zielmodul in LORA_ZIELMODULE)
)
anzahl_kopfparameter = sum(
    parameter.numel() for parameter in modell.get_classifier().parameters()
)
erwartete_anzahl_trainierbarer_parameter = (
    LORA_RANG * anzahl_lora_parameter_pro_rang + anzahl_kopfparameter
)

lora_config = LoraConfig(
    r=LORA_RANG,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_ZIELMODULE,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    modules_to_save=["head"]
)
modell = get_peft_model(modell, lora_config)

geraete_index = next(
    index
    for index in range(torch_directml.device_count())
    if "RX 6700 XT" in torch_directml.device_name(index)
)
geraet = torch_directml.device(geraete_index)
modell = modell.to(geraet)
geraetename = torch_directml.device_name(geraete_index).rstrip("\x00")
modellgeraet = next(modell.parameters()).device
print(f"Modellgerät: {geraetename}")
print(f"Modellparametergerät: {modellgeraet}")

lora_modulnamen = []
for name, modul in modell.named_modules():
    if hasattr(modul, "lora_A"):
        lora_modulnamen.append(name)
        print(f"LoRA Adaptermodul: {name}")

print(f"Anzahl LoRA Adaptermodule: {len(lora_modulnamen)}")

if len(lora_modulnamen) != 24:
    raise RuntimeError(
        f"Erwartet wurden 24 LoRA Adaptermodule, gefunden wurden {len(lora_modulnamen)}."
    )
if any("patch_embed.proj" in name for name in lora_modulnamen):
    raise RuntimeError("patch_embed.proj darf keinen LoRA Adapter enthalten.")

for block in range(12):
    for zielmodul in ("qkv", "proj"):
        suffix = f"blocks.{block}.attn.{zielmodul}"
        treffer = [name for name in lora_modulnamen if name.endswith(suffix)]
        if len(treffer) != 1:
            raise RuntimeError(
                f"Erwartet wurde genau ein Adapter fuer {suffix}, "
                f"gefunden wurden {len(treffer)}: {treffer}"
            )

trainierbare_parameter = [
    (name, parameter)
    for name, parameter in modell.named_parameters()
    if parameter.requires_grad
]
trainierbare_namen = [name for name, _ in trainierbare_parameter]
anzahl_trainierbarer_parameter = sum(
    parameter.numel() for _, parameter in trainierbare_parameter
)

for name in trainierbare_namen:
    print(f"Trainierbarer Parameter: {name}")

print(f"Anzahl trainierbarer Parameter: {anzahl_trainierbarer_parameter}")
print(f"Erwartete Anzahl trainierbarer Parameter: {erwartete_anzahl_trainierbarer_parameter}")

if anzahl_trainierbarer_parameter != erwartete_anzahl_trainierbarer_parameter:
    raise RuntimeError(
        "Erwartet wurden "
        f"{erwartete_anzahl_trainierbarer_parameter} trainierbare Parameter, "
        f"gefunden wurden {anzahl_trainierbarer_parameter}."
    )
