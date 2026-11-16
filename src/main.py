from datetime import datetime
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
print(
    "Modellgerät:",
    torch_directml.device_name(geraete_index).rstrip("\x00"),
    next(modell.parameters()).device,
)
head = modell.get_classifier()
if not isinstance(head, torch.nn.Linear):
    raise TypeError(f"Erwartet wurde ein linearer Klassifikationskopf, erhalten: {type(head)}")
if (head.in_features, head.out_features, head.bias is not None) != (768, 8, True):
    raise RuntimeError(
        "Unerwartete Kopfkonfiguration: "
        f"in_features={head.in_features}, out_features={head.out_features}, "
        f"bias={head.bias is not None}"
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
if trainierbare_namen != ["head.weight", "head.bias"]:
    raise RuntimeError(
        f"Unerwartete trainierbare Parameter: {trainierbare_namen}"
    )
if anzahl_trainierbarer_parameter != 6152:
    raise RuntimeError(
        f"Erwartet wurden 6152 trainierbare Parameter, "
        f"gefunden wurden {anzahl_trainierbarer_parameter}."
    )

optimierer = torch.optim.SGD(
    (parameter for _, parameter in trainierbare_parameter),
    lr=0.01,
)
verlustfunktion = torch.nn.CrossEntropyLoss()

modell.eval()

vorverarbeitung_config = resolve_model_data_config(modell)
vorverarbeitung = create_transform(**vorverarbeitung_config, is_training=False)
trainings_dataset = BildDataset(bildpfade, labels, vorverarbeitung)
validierungs_dataset = BildDataset(
    validierung_bildpfade,
    validierung_labels,
    vorverarbeitung,
)
batchgroesse = 8
trainings_dataloader = DataLoader(
    trainings_dataset,
    batch_size=batchgroesse,
    shuffle=True,
)
validierungs_dataloader = DataLoader(
    validierungs_dataset,
    batch_size=batchgroesse,
    shuffle=False,
)

head_vorher = {
    name: parameter.detach().clone()
    for name, parameter in trainierbare_parameter
}
trainingsverlust_summe = 0.0
training_richtig = 0
training_anzahl = 0
for batch, batch_labels in trainings_dataloader:
    batch = batch.to(geraet)
    batch_labels = batch_labels.to(geraet)

    optimierer.zero_grad(set_to_none=True)
    logits = modell(batch)
    erwartete_ausgabeform = (len(batch), len(KLASSEN_ZU_LABELS))
    if tuple(logits.shape) != erwartete_ausgabeform:
        raise RuntimeError(
            f"Unerwartete Ausgabeform {tuple(logits.shape)}; "
            f"erwartet wird {erwartete_ausgabeform}."
        )

    verlust = verlustfunktion(logits, batch_labels)
    if verlust.ndim != 0 or not torch.isfinite(verlust):
        raise RuntimeError(f"Ungueltiger Verlust: {verlust}")

    verlust.backward()
    if any(parameter.grad is None for _, parameter in trainierbare_parameter):
        raise RuntimeError(
            "Mindestens ein trainierbarer Kopfparameter hat keinen Gradienten."
        )
    if any(
        parameter.grad is not None
        for parameter in modell.parameters()
        if not parameter.requires_grad
    ):
        raise RuntimeError(
            "Ein eingefrorener ViT-Parameter hat unerwartet einen Gradienten."
        )

    optimierer.step()
    anzahl = batch_labels.size(0)
    trainingsverlust_summe += verlust.item() * anzahl
    training_richtig += (
        logits.argmax(dim=1) == batch_labels
    ).sum().item()
    training_anzahl += anzahl

ausgabeform_letzter_trainingsbatch = tuple(logits.shape)
head_danach = {
    name: not torch.equal(head_vorher[name], parameter.detach())
    for name, parameter in trainierbare_parameter
}
if not all(head_danach.values()):
    raise RuntimeError(f"Nicht alle Headparameter wurden aktualisiert: {head_danach}")

modell.eval()
validierungsverlust_summe = 0.0
validierung_richtig = 0
validierung_anzahl = 0

with torch.no_grad():
    for batch, batch_labels in validierungs_dataloader:
        batch = batch.to(geraet)
        batch_labels = batch_labels.to(geraet)

        logits = modell(batch)
        verlust = verlustfunktion(logits, batch_labels)

        anzahl = batch_labels.size(0)
        validierungsverlust_summe += verlust.item() * anzahl
        validierung_richtig += (
            logits.argmax(dim=1) == batch_labels
        ).sum().item()
        validierung_anzahl += anzahl

print("Trainingsbilder:", training_anzahl)
print("Validierungsbilder:", validierung_anzahl)
print("Trainingsbatches:", len(trainings_dataloader))
print(f"Trainingsverlust: {trainingsverlust_summe / training_anzahl:.4f}")
print(f"Trainingstrefferquote: {training_richtig / training_anzahl:.2%}")
print(f"Validierungsverlust: {validierungsverlust_summe / validierung_anzahl:.4f}")
print(f"Validierungstrefferquote: {validierung_richtig / validierung_anzahl:.2%}")
print("Ausgabeform letzter Trainingsbatch:", ausgabeform_letzter_trainingsbatch)
print("Trainierbare Parameter:", anzahl_trainierbarer_parameter)
print("Head aktualisiert:", all(head_danach.values()))
speicherordner = repo_ordner / "models"
speicherordner.mkdir(parents=True, exist_ok=True)

zeitpunkt = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
speicherpfad = speicherordner / f"linear_probing_{zeitpunkt}.pt"

torch.save(
    {
        "head_state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in head.state_dict().items()
        },
        "optimizer_state_dict": optimierer.state_dict(),
        "epoche": 1,
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "klassen_zu_labels": KLASSEN_ZU_LABELS,
        "vorverarbeitung": vorverarbeitung_config,
        "batchgroesse": batchgroesse,
        "optimierer": "SGD",
        "geraet": str(next(modell.parameters()).device),
        "trainingsverlust": trainingsverlust_summe / training_anzahl,
        "trainingstrefferquote": training_richtig / training_anzahl,
        "validierungsverlust": validierungsverlust_summe / validierung_anzahl,
        "validierungstrefferquote": validierung_richtig / validierung_anzahl,
    },
    speicherpfad,
)

print("Trainingsstand gespeichert:", speicherpfad)

gespeichert = torch.load(
    speicherpfad,
    map_location="cpu",
    weights_only=True,
)

for name, parameter in head.state_dict().items():
    if not torch.equal(
        parameter.detach().cpu(),
        gespeichert["head_state_dict"][name],
    ):
        raise RuntimeError(f"Gespeicherte Gewichte stimmen nicht: {name}")

print("Gespeicherte Kopfgewichte erfolgreich geprüft.")
